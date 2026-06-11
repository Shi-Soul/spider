"""Compare standalone G1 WBC rollout against tracking_bfm play rollout.

Run this with the tracking_bfm virtualenv so mjlab/rsl_rl are importable:

  /home/bai/MPC-RL/tracking_bfm/.venv/bin/python \
    -m spider.tasks.g1_wbc.compare_tracking_bfm ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from spider.tasks.g1_wbc.constants import (
    DECIMATION,
    MUJOCO_BODY_NAMES,
    OBS_DIM,
    PHYSICS_DT,
    POLICY_DT,
)
from spider.tasks.g1_wbc.motion import load_motion
from spider.tasks.g1_wbc.obs import G1WbcObservationBuilder
from spider.tasks.g1_wbc.policy import load_wbc_actor, resolve_checkpoint_path
from spider.tasks.g1_wbc.rollout import (
    G1WbcMujocoWarpEnv,
    WbcRolloutConfig,
    default_joint_pos_tensor,
    joint_actuator_specs,
)


TRACKING_BFM_ROOT = Path(__file__).resolve().parents[4] / "tracking_bfm"
if str(TRACKING_BFM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(TRACKING_BFM_ROOT / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.mdp.multi_commands import MotionCommand  # noqa: E402


TASK_ID = "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop"
OBS_TERM_SLICES = {
    "command": (0, 58),
    "ref_limb_ee_pose_b": (58, 238),
    "motion_ref_ang_vel": (238, 241),
    "robot_limb_ee_pose_b": (241, 421),
    "projected_gravity": (421, 436),
    "base_ang_vel": (436, 451),
    "joint_pos": (451, 596),
    "joint_vel": (596, 741),
    "actions": (741, 886),
}


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    checkpoint = resolve_checkpoint_path(args.checkpoint)
    motion_path = Path(args.motion).expanduser().resolve()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tracking_trace = _run_tracking_bfm_trace(
        motion_path=motion_path,
        motion_type=args.motion_type,
        checkpoint=checkpoint,
        device=str(device),
        steps=args.steps,
        seed=args.seed,
        disable_randomization=args.disable_randomization,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spider_trace = _run_spider_trace(
        motion_path=motion_path,
        motion_type=args.motion_type,
        checkpoint=args.checkpoint,
        device=str(device),
        steps=args.steps,
        clip_actions=tracking_trace["clip_actions"],
    )

    payload = _compare_trace(tracking_trace, spider_trace)
    payload["config"] = {
        "motion": str(motion_path),
        "motion_type": args.motion_type,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "steps": args.steps,
        "seed": args.seed,
        "disable_randomization": args.disable_randomization,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_tracking_bfm_trace(
    *,
    motion_path: Path,
    motion_type: str,
    checkpoint: Path,
    device: str,
    steps: int,
    seed: int,
    disable_randomization: bool,
) -> dict[str, torch.Tensor | float | dict[str, object]]:
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(seed)
    env_cfg.decimation = DECIMATION
    env_cfg.sim.mujoco.timestep = PHYSICS_DT
    env_cfg.commands["motion"].motion_file = str(motion_path)
    env_cfg.commands["motion"].motion_type = motion_type
    env_cfg.commands["motion"].history_steps = 0
    env_cfg.commands["motion"].future_steps = 1
    env_cfg.observations["actor"].terms["ref_limb_ee_pose_b"].history_length = 5
    env_cfg.observations["actor"].terms["robot_limb_ee_pose_b"].history_length = 5
    env_cfg.observations["actor"].terms["projected_gravity"].history_length = 5
    env_cfg.observations["actor"].terms["base_ang_vel"].history_length = 5
    env_cfg.observations["actor"].terms["joint_pos"].history_length = 5
    env_cfg.observations["actor"].terms["joint_vel"].history_length = 5
    env_cfg.observations["actor"].terms["actions"].history_length = 5
    env_cfg.terminations = {}
    if disable_randomization:
        env_cfg.events = {
            name: event
            for name, event in env_cfg.events.items()
            if name == "reset_scene_to_default"
        }

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    unwrapped = runner.env.unwrapped
    command = unwrapped.command_manager.get_term("motion")
    if not isinstance(command, MotionCommand):
        raise TypeError(f"Expected MultiMotionCommand, got {type(command)}")
    robot = unwrapped.scene["robot"]
    _reset_tracking_bfm_to_frame(unwrapped, command, frame=0)
    obs = TensorDict(
        unwrapped.observation_manager.compute(update_history=False),
        batch_size=[unwrapped.num_envs],
    )
    body_ids = torch.tensor(
        robot.find_bodies(MUJOCO_BODY_NAMES, preserve_order=True)[0],
        dtype=torch.long,
        device=device,
    )
    action_term = unwrapped.action_manager.get_term("joint_pos")
    clip_actions = runner.env.clip_actions

    obs_trace = []
    action_trace = []
    ctrl_trace = []
    qpos_trace = [_clone(unwrapped.sim.data.qpos)]
    qvel_trace = [_clone(unwrapped.sim.data.qvel)]
    body_pos_trace = [_clone(robot.data.body_link_pos_w[:, body_ids])]
    body_quat_trace = [_clone(robot.data.body_link_quat_w[:, body_ids])]
    time_step_trace = [_clone(command.time_steps)]

    with torch.inference_mode():
        for _ in range(int(steps)):
            actor_obs = obs["actor"]
            if actor_obs.shape[-1] != OBS_DIM:
                raise RuntimeError(f"Expected tracking_bfm obs dim {OBS_DIM}, got {actor_obs.shape[-1]}")
            action = policy(obs)
            if clip_actions is not None:
                action = torch.clamp(action, -clip_actions, clip_actions)
            ctrl = _tracking_bfm_action_to_control(action, action_term)
            obs_trace.append(_clone(actor_obs))
            action_trace.append(_clone(action))
            ctrl_trace.append(_clone(ctrl))
            obs, _, _, _ = runner.env.step(action.to(device))
            qpos_trace.append(_clone(unwrapped.sim.data.qpos))
            qvel_trace.append(_clone(unwrapped.sim.data.qvel))
            body_pos_trace.append(_clone(robot.data.body_link_pos_w[:, body_ids]))
            body_quat_trace.append(_clone(robot.data.body_link_quat_w[:, body_ids]))
            time_step_trace.append(_clone(command.time_steps))

    runner.env.close()
    return {
        "obs": torch.stack(obs_trace),
        "actions": torch.stack(action_trace),
        "controls": torch.stack(ctrl_trace),
        "qpos": torch.stack(qpos_trace),
        "qvel": torch.stack(qvel_trace),
        "body_pos_w": torch.stack(body_pos_trace),
        "body_quat_w": torch.stack(body_quat_trace),
        "time_steps": torch.stack(time_step_trace),
        "dt": POLICY_DT,
        "clip_actions": clip_actions,
    }


def _tracking_bfm_action_to_control(action: torch.Tensor, action_term) -> torch.Tensor:
    ctrl = action * action_term.scale + action_term.offset
    encoder_bias = action_term._entity.data.encoder_bias[:, action_term.target_ids]
    return (ctrl - encoder_bias).detach().clone()


def _reset_tracking_bfm_to_frame(
    env,
    command: MotionCommand,
    *,
    frame: int,
) -> None:
    device = command.time_steps.device
    env_ids = torch.arange(env.num_envs, device=device)
    command.motion_idx[:] = 0
    command.motion_length[:] = command.motion.file_lengths[0]
    command.time_steps[:] = int(frame)

    root_pos = command.body_pos_w[:, 0].clone()
    root_quat = command.body_quat_w[:, 0].clone()
    root_lin_vel = command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = command.body_ang_vel_w[:, 0].clone()
    joint_pos = command.joint_pos.clone()
    joint_vel = command.joint_vel.clone()

    soft_limits = command.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    command.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    command.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
        env_ids=env_ids,
    )
    command.robot.clear_state(env_ids=env_ids)
    env.observation_manager.reset(env_ids)
    env.action_manager.reset(env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    _update_command_relative_body_poses(command)
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)


def _update_command_relative_body_poses(command: MotionCommand) -> None:
    anchor_pos_w = command.anchor_pos_w[:, None, :]
    anchor_quat_w = command.anchor_quat_w[:, None, :]
    robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :]
    robot_anchor_quat_w = command.robot_anchor_quat_w[:, None, :]
    delta_pos_w = robot_anchor_pos_w.repeat(1, len(command.cfg.body_names), 1)
    delta_pos_w[..., 2] = anchor_pos_w[..., 2]
    delta_ori_w = _yaw_quat(
        _quat_mul(robot_anchor_quat_w, _quat_inv(anchor_quat_w))
    ).repeat(1, len(command.cfg.body_names), 1)
    command.body_quat_relative_w = _quat_mul(delta_ori_w, command.body_quat_w)
    command.body_pos_relative_w = delta_pos_w + _quat_apply(
        delta_ori_w,
        command.body_pos_w - anchor_pos_w,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1) / q.square().sum(
        dim=-1,
        keepdim=True,
    ).clamp(min=1.0e-9)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    from spider.tasks.g1_wbc.math_utils import quat_mul

    return quat_mul(q1, q2)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from spider.tasks.g1_wbc.math_utils import quat_apply

    return quat_apply(q, v)


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = q.unbind(-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(0.5 * yaw)
    out[..., 3] = torch.sin(0.5 * yaw)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1.0e-9)


def _run_spider_trace(
    *,
    motion_path: Path,
    motion_type: str,
    checkpoint: str | Path,
    device: str,
    steps: int,
    clip_actions: float | None,
) -> dict[str, torch.Tensor | float | dict[str, object]]:
    motion = load_motion(motion_path, motion_type=motion_type, device=device)
    actor = load_wbc_actor(checkpoint, device=device)
    config = WbcRolloutConfig(device=device, max_steps=steps, ref_offset=0)
    motion = motion.to(device)
    actor = actor.to(device)
    actor.eval()
    env = G1WbcMujocoWarpEnv(config)
    env.reset(motion.qpos()[0], motion.qvel()[0])
    obs_builder = G1WbcObservationBuilder(
        motion=motion,
        num_envs=config.num_envs,
        default_joint_pos=env.default_joint_pos,
        device=device,
    )
    last_action = torch.zeros(config.num_envs, env.action_scale.numel(), device=device)

    obs_trace = []
    action_trace = []
    ctrl_trace = []
    qpos_trace = []
    qvel_trace = []
    body_pos_trace = []
    body_quat_trace = []
    ref_indices = []

    state = env.robot_state()
    qpos_trace.append(state.qpos.detach().clone())
    qvel_trace.append(state.qvel.detach().clone())
    body_pos_trace.append(state.body_pos_w.detach().clone())
    body_quat_trace.append(state.body_quat_w.detach().clone())
    ref_indices.append(torch.zeros(config.num_envs, dtype=torch.long, device=device))

    total_steps = min(motion.num_frames, int(steps))
    with torch.inference_mode():
        for step_idx in range(total_steps):
            ref_idx_scalar = min(
                max(step_idx + int(config.ref_offset), 0), motion.num_frames - 1
            )
            ref_idx = torch.full(
                (config.num_envs,), ref_idx_scalar, dtype=torch.long, device=device
            )
            obs = obs_builder.compute(state, ref_idx, last_action)
            action = actor(obs)
            if clip_actions is not None:
                action = torch.clamp(action, -clip_actions, clip_actions)
            ctrl = env.action_to_control(action)
            env.step_control(ctrl)

            obs_trace.append(obs.detach().clone())
            action_trace.append(action.detach().clone())
            ctrl_trace.append(ctrl.detach().clone())
            ref_indices.append(ref_idx)
            last_action = action

            state = env.robot_state()
            qpos_trace.append(state.qpos.detach().clone())
            qvel_trace.append(state.qvel.detach().clone())
            body_pos_trace.append(state.body_pos_w.detach().clone())
            body_quat_trace.append(state.body_quat_w.detach().clone())

    return {
        "obs": torch.stack(obs_trace).detach().cpu(),
        "actions": torch.stack(action_trace).detach().cpu(),
        "controls": torch.stack(ctrl_trace).detach().cpu(),
        "qpos": torch.stack(qpos_trace).detach().cpu(),
        "qvel": torch.stack(qvel_trace).detach().cpu(),
        "body_pos_w": torch.stack(body_pos_trace).detach().cpu(),
        "body_quat_w": torch.stack(body_quat_trace).detach().cpu(),
        "ref_indices": torch.stack(ref_indices).detach().cpu(),
        "dt": POLICY_DT,
        "default_joint_pos": default_joint_pos_tensor("cpu"),
        "action_scale": joint_actuator_specs("cpu")["action_scale"],
    }


def _compare_trace(
    tracking: dict[str, torch.Tensor | float | dict[str, object]],
    spider: dict[str, torch.Tensor | float | dict[str, object]],
) -> dict[str, object]:
    out: dict[str, object] = {
        "tracking_dt": float(tracking["dt"]),
        "spider_dt": float(spider["dt"]),
    }
    for name in ("obs", "actions", "controls", "qpos", "qvel", "body_pos_w", "body_quat_w"):
        out[name] = _diff_stats(
            torch.as_tensor(tracking[name]).cpu(),
            torch.as_tensor(spider[name]).cpu(),
        )
    out["obs_terms"] = {
        name: _diff_stats(
            torch.as_tensor(tracking["obs"])[..., start:end].cpu(),
            torch.as_tensor(spider["obs"])[..., start:end].cpu(),
        )
        for name, (start, end) in OBS_TERM_SLICES.items()
    }
    out["tracking_time_steps"] = torch.as_tensor(tracking["time_steps"]).cpu().view(-1).tolist()
    out["spider_ref_indices"] = torch.as_tensor(spider["ref_indices"]).cpu().view(-1).tolist()
    return out


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "compatible": False}
    diff = torch.nan_to_num((a.float() - b.float()).abs())
    return {
        "compatible": True,
        "shape": list(a.shape),
        "max": float(diff.max().item()) if diff.numel() else 0.0,
        "mean": float(diff.mean().item()) if diff.numel() else 0.0,
        "p95": float(torch.quantile(diff.reshape(-1), 0.95).item()) if diff.numel() else 0.0,
    }


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--motion-type", choices=("mujoco", "isaaclab"), required=True)
    parser.add_argument("--checkpoint", default="bc")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only deterministic reset events in tracking_bfm env.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
