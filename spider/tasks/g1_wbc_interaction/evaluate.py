"""CLI for G1 WBC interaction MPC+RL rollouts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spider.config import Config, build_sampling_mpc_config
from spider.optimizers.receding import run_sampling_receding_mpc, sampling_mpc_metadata
from spider.simulators.g1_wbc_interaction import (
    G1WbcInteractionSamplingTask,
    InteractionRolloutConfig,
)
from spider.tasks.g1_wbc.constants import ACTION_DIM, POLICY_DT, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.policy import load_wbc_actor, resolve_checkpoint_path
from spider.tasks.g1_wbc.rollout import RolloutResult
from spider.tasks.g1_wbc_interaction.metrics import (
    InteractionScoreWeights,
    compute_interaction_rollout_metrics,
    compute_retarget_object_metrics,
)
from spider.tasks.g1_wbc_interaction.motion import load_interaction_motion, qvel_from_full_qpos
from spider.tasks.g1_wbc_interaction.render import render_interaction_comparison_video
from spider.tasks.g1_wbc_interaction.rollout import (
    G1WbcInteractionMujocoWarpEnv,
    command_from_full_qpos_trajectory,
    load_interaction_model_and_layout,
    run_interaction_command_rollout,
)


def main() -> None:
    args = _parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    model, layout = load_interaction_model_and_layout(args.model_path)
    source_dt = _resolve_source_dt(args.motion, args.source_dt)
    motion = load_interaction_motion(
        args.motion,
        model=model,
        layout=layout,
        device=device,
        source_dt=source_dt,
    )
    raw_ref_qpos = _load_raw_reference_qpos(args.motion, layout)
    checkpoint_path = None
    actor = None
    if args.method != "static_qpos":
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        actor = load_wbc_actor(args.checkpoint, device=device)
    rollout_config = InteractionRolloutConfig(
        model_path=args.model_path,
        device=device,
        num_envs=args.num_envs if args.method == "no_mpc" else 1,
        max_steps=args.max_steps,
        ref_offset=args.ref_offset,
        nconmax_per_env=args.nconmax_per_env,
        njmax_per_env=args.njmax_per_env,
        sync_after_step=args.sync_after_step,
        forward_after_step=args.forward_after_step,
        use_cuda_graph=args.use_cuda_graph,
    )

    mpc_payload = None
    spider_config = None
    result = None
    if args.method == "static_qpos":
        qpos = _load_saved_full_qpos(args.saved_qpos, layout, device=device)
        qvel = qvel_from_full_qpos(qpos, layout, dt=POLICY_DT)
        rollout = _static_rollout(qpos, qvel, motion, rollout_config)
    elif args.method == "no_mpc":
        assert actor is not None
        command_qpos = motion.full_state_qpos()[:, None, :]
        command_qvel = motion.full_state_qvel()[:, None, :]
        if rollout_config.num_envs > 1:
            command_qpos = command_qpos.expand(-1, rollout_config.num_envs, -1)
            command_qvel = command_qvel.expand(-1, rollout_config.num_envs, -1)
        command = command_from_full_qpos_trajectory(
            motion,
            command_qpos,
            rollout_config,
            full_qvel_trajectory=command_qvel,
            preserve_template_first=False,
        )
        rollout = run_interaction_command_rollout(
            command,
            actor,
            rollout_config,
            initial_qpos=motion.full_state_qpos()[0],
            initial_qvel=motion.full_state_qvel()[0],
        )
    else:
        assert actor is not None
        spider_config = _build_sampling_config(args)
        task = G1WbcInteractionSamplingTask(
            motion,
            actor,
            rollout_config,
            score_weights=_score_weights_from_args(args),
        )
        total_steps = motion.num_frames - 1
        if args.max_steps is not None:
            total_steps = min(total_steps, int(args.max_steps))
        receding_result = run_sampling_receding_mpc(
            spider_config,
            task,
            total_steps=total_steps,
        )
        result = task.build_result(
            receding_result.controls,
            receding_result.infos,
            total_steps=total_steps,
        )
        rollout = result.rollout
        mpc_payload = {
            **sampling_mpc_metadata(spider_config),
            "history": _jsonable_infos(receding_result.infos),
            "final_scores_mean": _safe_tensor_stat(result.scores, "mean"),
            "final_scores_max": _safe_tensor_stat(result.scores, "max"),
            "num_windows": result.num_windows,
            "score_weights": _score_weights_from_args(args).__dict__,
        }

    metrics = compute_interaction_rollout_metrics(motion, rollout, layout=layout)
    rollout_qpos_np = _cpu_np(rollout.qpos[1:, 0])
    rollout_times = (np.arange(rollout_qpos_np.shape[0], dtype=np.float64) + 1.0) * POLICY_DT
    rollout_ref_qpos = _reference_qpos_at_times(
        raw_ref_qpos,
        layout,
        source_dt,
        rollout_times,
    )
    metrics.update(
        compute_retarget_object_metrics(
            rollout_qpos_np,
            rollout_ref_qpos,
            layout,
            pos_threshold=args.object_pos_threshold,
            quat_threshold=args.object_quat_threshold,
        )
    )
    payload: dict[str, Any] = {
        "method": args.method,
        "motion": str(Path(args.motion).expanduser().resolve()),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "source_dt": source_dt,
        "policy_dt": POLICY_DT,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": device,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "objects": list(layout.object_names),
        "metrics": metrics,
    }
    baseline_metrics = None
    baseline_qpos_np = None
    baseline_video_qpos = None
    baseline_time_grid = None
    if args.baseline is not None:
        baseline_qpos_np, baseline_times = _load_saved_full_trajectory(
            args.baseline,
            layout,
            fallback_dt=args.baseline_dt,
        )
        baseline_time_grid = _load_saved_time_grid(
            args.baseline,
            layout,
            fallback_dt=args.baseline_dt,
        )
        baseline_ref_qpos_full = _reference_qpos_at_times(
            raw_ref_qpos,
            layout,
            source_dt,
            baseline_times,
        )
        retarget_grid_arrays = _retarget_style_arrays(
            rollout,
            layout=layout,
            time_grid=baseline_time_grid,
        )
        retarget_grid_qpos = _flatten_qpos_np(retarget_grid_arrays["qpos"], layout.nq)
        retarget_grid_metrics = compute_retarget_object_metrics(
            retarget_grid_qpos,
            baseline_ref_qpos_full,
            layout,
            pos_threshold=args.object_pos_threshold,
            quat_threshold=args.object_quat_threshold,
        )
        baseline_full_metrics = compute_retarget_object_metrics(
            baseline_qpos_np,
            baseline_ref_qpos_full,
            layout,
            pos_threshold=args.object_pos_threshold,
            quat_threshold=args.object_quat_threshold,
        )
        baseline_matched_qpos = _qpos_at_times(
            baseline_qpos_np,
            baseline_times,
            layout,
            rollout_times,
        )
        baseline_video_qpos = baseline_matched_qpos
        baseline_metrics = compute_retarget_object_metrics(
            baseline_matched_qpos,
            rollout_ref_qpos,
            layout,
            pos_threshold=args.object_pos_threshold,
            quat_threshold=args.object_quat_threshold,
        )
        payload["baseline"] = {
            "path": str(Path(args.baseline).expanduser().resolve()),
            "metrics": baseline_metrics,
            "full_metrics": baseline_full_metrics,
        }
        payload["retarget_grid"] = {
            "time_shape": list(baseline_time_grid.shape),
            "metrics": retarget_grid_metrics,
        }
        payload["comparison"] = _comparison(metrics, baseline_metrics)
        payload["retarget_grid_comparison"] = _comparison(
            retarget_grid_metrics,
            baseline_full_metrics,
        )
    if mpc_payload is not None:
        payload["mpc"] = mpc_payload
    print(json.dumps(_jsonify(payload), indent=2, sort_keys=True))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(_jsonify(payload), indent=2, sort_keys=True) + "\n"
        )
        if args.save_rollout:
            _save_rollout_npz(output_dir / "rollout.npz", rollout)
            window_steps = None
            if spider_config is not None:
                window_steps = int(spider_config.ctrl_steps)
            _save_retarget_style(
                output_dir / "trajectory_mjwp.npz",
                rollout,
                layout=layout,
                time_grid=baseline_time_grid,
                window_steps=window_steps,
            )
            _save_retarget_style(
                output_dir / "trajectory_mpc_rl_object.npz",
                rollout,
                layout=layout,
                time_grid=baseline_time_grid,
                window_steps=window_steps,
            )
            if result is not None:
                np.savez_compressed(
                    output_dir / "mpc_command.npz",
                    refined_qpos=_cpu_np(result.refined_qpos),
                    refined_qvel=_cpu_np(result.refined_qvel),
                    command_qpos_trajectory=_cpu_np(result.command_qpos),
                    candidate_scores=_cpu_np(result.scores),
                    controls=_cpu_np(result.controls),
                    executed_controls=_cpu_np(result.executed_controls),
                )
        if args.save_video:
            render_interaction_comparison_video(
                model_path=args.model_path,
                rollout_qpos=rollout_qpos_np,
                reference_qpos=rollout_ref_qpos,
                baseline_qpos=baseline_video_qpos,
                out_path=output_dir / "visualization_mpc_rl_object.mp4",
                fps=args.video_fps,
                width=args.video_width,
                height=args.video_height,
                camera=args.video_camera,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="Full-state reference npz.")
    parser.add_argument("--model-path", required=True, help="Complete interaction MuJoCo XML.")
    parser.add_argument("--baseline", default=None, help="Existing retarget baseline npz.")
    parser.add_argument(
        "--source-dt",
        type=float,
        default=None,
        help="Reference qpos timestep. Defaults to retarget-style 1/30s when absent.",
    )
    parser.add_argument("--baseline-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--checkpoint", default="bc")
    parser.add_argument(
        "--method",
        default="g1_wbc_interaction",
        choices=("no_mpc", "g1_wbc_interaction", "static_qpos"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ref-offset", type=int, default=0)
    parser.add_argument("--nconmax-per-env", type=int, default=InteractionRolloutConfig.nconmax_per_env)
    parser.add_argument("--njmax-per-env", type=int, default=InteractionRolloutConfig.njmax_per_env)
    parser.add_argument("--sync-after-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--forward-after-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-rollout", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera", default="auto")
    parser.add_argument("--saved-qpos", default=None)
    parser.add_argument("--mpc-samples", type=int, default=None)
    parser.add_argument("--mpc-rollout-batch-size", type=int, default=0)
    parser.add_argument("--mpc-iterations", type=int, default=None)
    parser.add_argument("--mpc-planning-horizon-steps", type=int, default=None)
    parser.add_argument("--mpc-control-steps", type=int, default=None)
    parser.add_argument("--mpc-knot-count", type=int, default=None)
    parser.add_argument("--mpc-temperature", type=float, default=None)
    parser.add_argument("--mpc-control-update-mode", choices=("weighted_mean", "best"), default="weighted_mean")
    parser.add_argument("--mpc-first-ctrl-noise-scale", type=float, default=0.5)
    parser.add_argument("--mpc-last-ctrl-noise-scale", type=float, default=1.0)
    parser.add_argument("--mpc-final-noise-scale", type=float, default=0.1)
    parser.add_argument("--mpc-root-pos-sigma", type=float, default=None)
    parser.add_argument("--mpc-root-rot-sigma", type=float, default=None)
    parser.add_argument("--mpc-joint-sigma", type=float, default=None)
    parser.add_argument("--mpc-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--object-pos-weight", type=float, default=35.0)
    parser.add_argument("--object-rot-weight", type=float, default=4.0)
    parser.add_argument("--object-final-pos-weight", type=float, default=80.0)
    parser.add_argument("--object-final-rot-weight", type=float, default=8.0)
    parser.add_argument("--object-vel-weight", type=float, default=0.5)
    parser.add_argument("--robot-contact-mismatch-weight", type=float, default=4.0)
    parser.add_argument("--robot-contact-switch-weight", type=float, default=2.0)
    parser.add_argument("--robot-contact-false-positive-weight", type=float, default=0.0)
    parser.add_argument("--robot-contact-false-negative-weight", type=float, default=0.0)
    parser.add_argument("--robot-contact-force-excess-weight", type=float, default=0.0)
    parser.add_argument("--robot-contact-force-delta-weight", type=float, default=0.0)
    parser.add_argument("--robot-bad-floor-contact-weight", type=float, default=0.0)
    parser.add_argument("--robot-bad-floor-force-excess-weight", type=float, default=0.0)
    parser.add_argument("--robot-ee-global-pos-weight", type=float, default=2.5)
    parser.add_argument("--robot-ee-global-rot-weight", type=float, default=0.0)
    parser.add_argument("--robot-ee-local-pos-weight", type=float, default=1.5)
    parser.add_argument("--robot-ee-local-rot-weight", type=float, default=0.0)
    parser.add_argument("--robot-hand-global-pos-weight", type=float, default=0.0)
    parser.add_argument("--robot-hand-global-rot-weight", type=float, default=0.0)
    parser.add_argument("--robot-hand-local-pos-weight", type=float, default=0.0)
    parser.add_argument("--robot-hand-local-rot-weight", type=float, default=0.0)
    parser.add_argument("--robot-body-global-pos-weight", type=float, default=0.0)
    parser.add_argument("--robot-body-global-rot-weight", type=float, default=0.0)
    parser.add_argument("--robot-body-local-pos-weight", type=float, default=0.0)
    parser.add_argument("--robot-body-local-rot-weight", type=float, default=0.0)
    parser.add_argument("--robot-root-pos-weight", type=float, default=1.0)
    parser.add_argument("--robot-root-rot-weight", type=float, default=0.4)
    parser.add_argument("--robot-joint-pos-weight", type=float, default=0.15)
    parser.add_argument("--robot-action-delta-weight", type=float, default=0.0)
    parser.add_argument("--robot-joint-acc-weight", type=float, default=0.0)
    parser.add_argument("--robot-joint-jerk-weight", type=float, default=0.0)
    parser.add_argument("--control-delta-weight", type=float, default=0.04)
    parser.add_argument("--object-pos-threshold", type=float, default=0.1)
    parser.add_argument("--object-quat-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_sampling_config(args: argparse.Namespace) -> Config:
    for attr in (
        "mpc_samples",
        "mpc_iterations",
        "mpc_planning_horizon_steps",
        "mpc_control_steps",
        "mpc_knot_count",
        "mpc_temperature",
        "mpc_root_pos_sigma",
        "mpc_root_rot_sigma",
        "mpc_joint_sigma",
    ):
        if getattr(args, attr) is None:
            raise ValueError(f"--{attr.replace('_', '-')} is required.")
    return build_sampling_mpc_config(
        robot_type="g1",
        embodiment_type="humanoid",
        simulator="g1_wbc_interaction",
        device=args.device,
        num_samples=int(args.mpc_samples),
        rollout_batch_size=int(args.mpc_rollout_batch_size),
        max_num_iterations=int(args.mpc_iterations),
        horizon_steps=int(args.mpc_planning_horizon_steps),
        ctrl_steps=int(args.mpc_control_steps),
        knot_count=int(args.mpc_knot_count),
        temperature=float(args.mpc_temperature),
        control_update_mode=args.mpc_control_update_mode,
        sim_dt=POLICY_DT,
        nq=QPOS_DIM,
        nv=QVEL_DIM,
        nu=QPOS_DIM - 1,
        pos_noise_scale=float(args.mpc_root_pos_sigma),
        rot_noise_scale=float(args.mpc_root_rot_sigma),
        joint_noise_scale=float(args.mpc_joint_sigma),
        first_ctrl_noise_scale=float(args.mpc_first_ctrl_noise_scale),
        last_ctrl_noise_scale=float(args.mpc_last_ctrl_noise_scale),
        final_noise_scale=float(args.mpc_final_noise_scale),
        use_torch_compile=bool(args.mpc_torch_compile),
        seed=int(args.seed),
    )


def _score_weights_from_args(args: argparse.Namespace) -> InteractionScoreWeights:
    return InteractionScoreWeights(
        robot_contact_mismatch=float(args.robot_contact_mismatch_weight),
        robot_contact_switch=float(args.robot_contact_switch_weight),
        robot_contact_false_positive=float(args.robot_contact_false_positive_weight),
        robot_contact_false_negative=float(args.robot_contact_false_negative_weight),
        robot_contact_force_excess=float(args.robot_contact_force_excess_weight),
        robot_contact_force_delta=float(args.robot_contact_force_delta_weight),
        robot_bad_floor_contact=float(args.robot_bad_floor_contact_weight),
        robot_bad_floor_force_excess=float(args.robot_bad_floor_force_excess_weight),
        robot_ee_global_pos=float(args.robot_ee_global_pos_weight),
        robot_ee_global_rot=float(args.robot_ee_global_rot_weight),
        robot_ee_local_pos=float(args.robot_ee_local_pos_weight),
        robot_ee_local_rot=float(args.robot_ee_local_rot_weight),
        robot_hand_global_pos=float(args.robot_hand_global_pos_weight),
        robot_hand_global_rot=float(args.robot_hand_global_rot_weight),
        robot_hand_local_pos=float(args.robot_hand_local_pos_weight),
        robot_hand_local_rot=float(args.robot_hand_local_rot_weight),
        robot_body_global_pos=float(args.robot_body_global_pos_weight),
        robot_body_global_rot=float(args.robot_body_global_rot_weight),
        robot_body_local_pos=float(args.robot_body_local_pos_weight),
        robot_body_local_rot=float(args.robot_body_local_rot_weight),
        robot_root_pos=float(args.robot_root_pos_weight),
        robot_root_rot=float(args.robot_root_rot_weight),
        robot_joint_pos=float(args.robot_joint_pos_weight),
        robot_action_delta=float(args.robot_action_delta_weight),
        robot_joint_acc=float(args.robot_joint_acc_weight),
        robot_joint_jerk=float(args.robot_joint_jerk_weight),
        control_delta=float(args.control_delta_weight),
        object_pos=float(args.object_pos_weight),
        object_rot=float(args.object_rot_weight),
        object_final_pos=float(args.object_final_pos_weight),
        object_final_rot=float(args.object_final_rot_weight),
        object_vel=float(args.object_vel_weight),
    )


def _static_rollout(qpos: torch.Tensor, qvel: torch.Tensor, motion, config):
    del motion
    device = torch.device(config.device)
    qpos = qpos.to(device, dtype=torch.float32)
    qvel = qvel.to(device, dtype=torch.float32)
    if qpos.ndim == 2:
        qpos = qpos[:, None, :]
    if qvel.ndim == 2:
        qvel = qvel[:, None, :]
    total_steps = qpos.shape[0] - 1
    if config.max_steps is not None:
        total_steps = min(total_steps, int(config.max_steps))
    qpos = qpos[: total_steps + 1]
    qvel = qvel[: total_steps + 1]
    env = G1WbcInteractionMujocoWarpEnv(
        replace(
            config,
            num_envs=int(qpos.shape[1]),
            max_steps=total_steps,
            use_cuda_graph=False,
        )
    )
    body_pos_w = []
    body_quat_w = []
    body_lin_vel_w = []
    body_ang_vel_w = []
    contact_indicator = []
    contact_force = []
    floor_contact_indicator = []
    floor_contact_force = []
    with torch.inference_mode():
        for frame_idx in range(total_steps + 1):
            env.reset(qpos[frame_idx], qvel[frame_idx])
            state = env.robot_state()
            floor_contact, floor_force = env.floor_contact()
            body_pos_w.append(state.body_pos_w.detach().clone())
            body_quat_w.append(state.body_quat_w.detach().clone())
            body_lin_vel_w.append(state.body_lin_vel_w.detach().clone())
            body_ang_vel_w.append(state.body_ang_vel_w.detach().clone())
            contact_indicator.append(floor_contact[:, :2].detach().clone())
            contact_force.append(floor_force[:, :2].detach().clone())
            floor_contact_indicator.append(floor_contact.detach().clone())
            floor_contact_force.append(floor_force.detach().clone())
    actions = torch.zeros(
        total_steps,
        qpos.shape[1],
        ACTION_DIM,
        dtype=torch.float32,
        device=device,
    )
    controls = env.layout.robot_qpos(qpos[:-1])[:, :, 7:].contiguous()
    ref_indices = torch.arange(total_steps + 1, dtype=torch.long, device=device)
    ref_indices = ref_indices[:, None].expand(total_steps + 1, qpos.shape[1])
    return RolloutResult(
        qpos=qpos,
        qvel=qvel,
        body_pos_w=torch.stack(body_pos_w, dim=0),
        body_quat_w=torch.stack(body_quat_w, dim=0),
        body_lin_vel_w=torch.stack(body_lin_vel_w, dim=0),
        body_ang_vel_w=torch.stack(body_ang_vel_w, dim=0),
        actions=actions,
        controls=controls,
        contact_indicator=torch.stack(contact_indicator, dim=0),
        contact_force=torch.stack(contact_force, dim=0),
        floor_contact_indicator=torch.stack(floor_contact_indicator, dim=0),
        floor_contact_force=torch.stack(floor_contact_force, dim=0),
        ref_indices=ref_indices,
    )


def _load_saved_full_qpos(path: str | None, layout, *, device: str) -> torch.Tensor:
    if path is None:
        raise ValueError("--method static_qpos requires --saved-qpos.")
    qpos_path = Path(path).expanduser().resolve()
    with np.load(qpos_path) as data:
        for key in ("qpos", "refined_qpos", "command_qpos_trajectory"):
            if key in data.files:
                qpos = data[key]
                break
        else:
            raise ValueError(f"{qpos_path} is missing qpos/refined_qpos/command_qpos_trajectory.")
    if qpos.ndim == 3 and qpos.shape[1] == 1:
        qpos = qpos[:, 0]
    if qpos.ndim == 3:
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.ndim != 2 or qpos.shape[-1] != layout.nq:
        raise ValueError(f"Expected qpos shape (T,{layout.nq}), got {qpos.shape}.")
    return torch.tensor(qpos, dtype=torch.float32, device=device)


def _load_saved_full_trajectory(
    path: str | Path,
    layout,
    *,
    fallback_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    traj_path = Path(path).expanduser().resolve()
    with np.load(traj_path) as data:
        for key in ("qpos", "refined_qpos", "command_qpos_trajectory"):
            if key in data.files:
                qpos = np.asarray(data[key])
                break
        else:
            raise ValueError(f"{traj_path} is missing qpos/refined_qpos/command_qpos_trajectory.")
        if "time" in data.files:
            time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
        else:
            count = int(np.prod(qpos.shape[:-1])) if qpos.ndim > 2 else int(qpos.shape[0])
            time = np.arange(count, dtype=np.float64) * float(fallback_dt)
    qpos = _flatten_qpos_np(qpos, layout.nq)
    if time.shape[0] != qpos.shape[0]:
        time = np.arange(qpos.shape[0], dtype=np.float64) * float(fallback_dt)
    return qpos, time


def _load_saved_time_grid(
    path: str | Path,
    layout,
    *,
    fallback_dt: float,
) -> np.ndarray:
    traj_path = Path(path).expanduser().resolve()
    with np.load(traj_path) as data:
        for key in ("qpos", "refined_qpos", "command_qpos_trajectory"):
            if key in data.files:
                qpos_shape = tuple(np.asarray(data[key]).shape)
                break
        else:
            raise ValueError(
                f"{traj_path} is missing qpos/refined_qpos/command_qpos_trajectory."
            )
        if not qpos_shape or qpos_shape[-1] != layout.nq:
            raise ValueError(f"Expected qpos last dim {layout.nq}, got {qpos_shape}.")
        time_shape = qpos_shape[:-1]
        count = int(np.prod(time_shape))
        if "time" in data.files:
            time = np.asarray(data["time"], dtype=np.float64)
        else:
            time = np.arange(count, dtype=np.float64) * float(fallback_dt)
        if time.shape == time_shape:
            return np.ascontiguousarray(time)
        if time.size != count:
            raise ValueError(
                f"Expected {count} timestamps for {traj_path}, got shape {time.shape}."
            )
        return np.ascontiguousarray(time.reshape(time_shape))


def _comparison(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key in (
        "obj_pos_err",
        "obj_quat_err",
        "obj_pos_err_mean",
        "obj_quat_err_mean",
        "add_auc10_mean",
        "add_auc10",
    ):
        if key in metrics and key in baseline:
            out[f"{key}_delta_vs_baseline"] = float(metrics[key]) - float(baseline[key])
    return out


def _save_rollout_npz(path: Path, rollout) -> None:
    arrays = {
        "qpos": _cpu_np(rollout.qpos),
        "qvel": _cpu_np(rollout.qvel),
        "actions": _cpu_np(rollout.actions),
        "controls": _cpu_np(rollout.controls),
        "ref_indices": _cpu_np(rollout.ref_indices),
        "dt": np.array(rollout.dt, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)


def _save_retarget_style(
    path: Path,
    rollout,
    *,
    layout,
    time_grid: np.ndarray | None = None,
    window_steps: int | None = None,
) -> None:
    np.savez_compressed(
        path,
        **_retarget_style_arrays(
            rollout,
            layout=layout,
            time_grid=time_grid,
            window_steps=window_steps,
        ),
    )


def _retarget_style_arrays(
    rollout,
    *,
    layout,
    time_grid: np.ndarray | None = None,
    window_steps: int | None = None,
) -> dict[str, np.ndarray]:
    if time_grid is not None:
        time = np.asarray(time_grid, dtype=np.float64)
        flat_time = time.reshape(-1)
        native_qpos = rollout.qpos[:, 0].detach().cpu().numpy()
        native_qvel = rollout.qvel[:, 0].detach().cpu().numpy()
        native_time = np.arange(native_qpos.shape[0], dtype=np.float64) * float(POLICY_DT)
        qpos = _qpos_at_times(native_qpos, native_time, layout, flat_time)
        qvel = _array_at_times(native_qvel, native_time, flat_time)
        ctrl = _array_at_times(
            rollout.controls[:, 0].detach().cpu().numpy(),
            (np.arange(rollout.controls.shape[0], dtype=np.float64) + 1.0)
            * float(POLICY_DT),
            flat_time,
        )
        shape = time.shape
        return {
            "qpos": qpos.reshape(shape + (layout.nq,)),
            "qvel": qvel.reshape(shape + (layout.nv,)),
            "time": time,
            "ctrl": ctrl.reshape(shape + (ctrl.shape[-1],)),
        }

    qpos = rollout.qpos[1:, 0].detach().cpu().numpy()
    qvel = rollout.qvel[1:, 0].detach().cpu().numpy()
    ctrl = rollout.controls[:, 0].detach().cpu().numpy()
    time = (np.arange(qpos.shape[0], dtype=np.float64) + 1.0) * float(POLICY_DT)
    if window_steps is not None and window_steps > 0 and qpos.shape[0] % window_steps == 0:
        windows = qpos.shape[0] // int(window_steps)
        qpos = qpos.reshape(windows, int(window_steps), qpos.shape[-1])
        qvel = qvel.reshape(windows, int(window_steps), qvel.shape[-1])
        ctrl = ctrl.reshape(windows, int(window_steps), ctrl.shape[-1])
        time = time.reshape(windows, int(window_steps))
    return {"qpos": qpos, "qvel": qvel, "time": time, "ctrl": ctrl}


def _resolve_source_dt(motion_path: str | Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    path = Path(motion_path).expanduser().resolve()
    with np.load(path) as data:
        if "dt" in data.files:
            return float(np.asarray(data["dt"]).item())
        if "fps" in data.files:
            return 1.0 / float(np.asarray(data["fps"]).item())
    for config_path in _candidate_config_paths(path):
        text = config_path.read_text()
        match = re.search(r"(?m)^ref_dt:\s*([0-9.eE+-]+)\s*$", text)
        if match:
            return float(match.group(1))
    return 1.0 / 30.0


def _candidate_config_paths(motion_path: Path) -> list[Path]:
    candidates = [motion_path.parent / "config.yaml"]
    task_dir = motion_path.parent.parent
    if task_dir.is_dir():
        candidates.extend(sorted(task_dir.glob("*/config.yaml")))
    return [path for path in candidates if path.is_file()]


def _load_raw_reference_qpos(path: str | Path, layout) -> np.ndarray:
    ref_path = Path(path).expanduser().resolve()
    with np.load(ref_path) as data:
        if "qpos" not in data.files:
            raise ValueError(f"Reference motion {ref_path} is missing qpos.")
        return _flatten_qpos_np(np.asarray(data["qpos"]), layout.nq)


def _reference_qpos_at_times(
    raw_qpos: np.ndarray,
    layout,
    source_dt: float,
    times: np.ndarray,
) -> np.ndarray:
    raw_qpos = _flatten_qpos_np(raw_qpos, layout.nq)
    times = np.asarray(times, dtype=np.float64)
    if raw_qpos.shape[0] <= 1:
        return np.repeat(raw_qpos[:1], times.shape[0], axis=0)
    src_dt = float(source_dt)
    src_duration = (raw_qpos.shape[0] - 1) * src_dt
    u = np.clip(times, 0.0, src_duration) / src_dt
    i0 = np.floor(u).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, raw_qpos.shape[0] - 1)
    alpha = (u - i0.astype(np.float64))[:, None]
    out = raw_qpos[i0] * (1.0 - alpha) + raw_qpos[i1] * alpha
    for qpos_adr in _freejoint_qpos_addrs(layout):
        out[:, qpos_adr + 3 : qpos_adr + 7] = _slerp_np(
            raw_qpos[i0, qpos_adr + 3 : qpos_adr + 7],
            raw_qpos[i1, qpos_adr + 3 : qpos_adr + 7],
            alpha,
        )
    return out


def _qpos_at_times(
    qpos: np.ndarray,
    qpos_times: np.ndarray,
    layout,
    times: np.ndarray,
) -> np.ndarray:
    qpos = _flatten_qpos_np(qpos, layout.nq)
    qpos_times = np.asarray(qpos_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if qpos_times.shape[0] != qpos.shape[0]:
        raise ValueError(
            f"Expected {qpos.shape[0]} qpos timestamps, got {qpos_times.shape[0]}."
        )
    if qpos.shape[0] <= 1:
        return np.repeat(qpos[:1], times.shape[0], axis=0)
    query = np.clip(times, float(qpos_times[0]), float(qpos_times[-1]))
    i1 = np.searchsorted(qpos_times, query, side="left")
    i1 = np.clip(i1, 1, qpos.shape[0] - 1)
    i0 = i1 - 1
    denom = np.maximum(qpos_times[i1] - qpos_times[i0], 1.0e-12)
    alpha = ((query - qpos_times[i0]) / denom)[:, None]
    out = qpos[i0] * (1.0 - alpha) + qpos[i1] * alpha
    for qpos_adr in _freejoint_qpos_addrs(layout):
        out[:, qpos_adr + 3 : qpos_adr + 7] = _slerp_np(
            qpos[i0, qpos_adr + 3 : qpos_adr + 7],
            qpos[i1, qpos_adr + 3 : qpos_adr + 7],
            alpha,
        )
    return out


def _array_at_times(
    values: np.ndarray,
    value_times: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values)
    value_times = np.asarray(value_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"Expected at least 2D time-major values, got {values.shape}.")
    if value_times.shape[0] != values.shape[0]:
        raise ValueError(
            f"Expected {values.shape[0]} value timestamps, got {value_times.shape[0]}."
        )
    if values.shape[0] <= 1:
        return np.repeat(values[:1], times.shape[0], axis=0)
    query = np.clip(times, float(value_times[0]), float(value_times[-1]))
    i1 = np.searchsorted(value_times, query, side="left")
    i1 = np.clip(i1, 1, values.shape[0] - 1)
    i0 = i1 - 1
    denom = np.maximum(value_times[i1] - value_times[i0], 1.0e-12)
    alpha = ((query - value_times[i0]) / denom).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[i0] * (1.0 - alpha) + values[i1] * alpha


def _slerp_np(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat_np(q0)
    q1 = _normalize_quat_np(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    small = dot > 0.9995
    theta0 = np.arccos(dot)
    sin_theta0 = np.maximum(np.sin(theta0), 1.0e-8)
    theta = theta0 * alpha
    s0 = np.sin(theta0 - theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    out = s0 * q0 + s1 * q1
    lerp = q0 + alpha * (q1 - q0)
    out = np.where(small, lerp, out)
    return _normalize_quat_np(out)


def _normalize_quat_np(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-8)


def _freejoint_qpos_addrs(layout) -> list[int]:
    addrs = [int(layout.root_qpos_adr)]
    addrs.extend(
        int(obj.pose_qpos_adr)
        for obj in layout.objects
        if obj.has_freejoint_pose and obj.pose_qpos_adr is not None
    )
    return addrs


def _flatten_qpos_np(qpos: np.ndarray, nq: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 3 and qpos.shape[1] == 1:
        qpos = qpos[:, 0]
    elif qpos.ndim == 3:
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.ndim != 2 or qpos.shape[-1] != nq:
        raise ValueError(f"Expected qpos shape (T,{nq}) or (...,{nq}), got {qpos.shape}.")
    return qpos


def _jsonable_infos(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_jsonify(info) for info in infos]


def _safe_tensor_stat(value: torch.Tensor, stat: str) -> float | None:
    if value.numel() == 0:
        return None
    if stat == "mean":
        return float(value.float().mean().detach().cpu().item())
    if stat == "max":
        return float(value.float().max().detach().cpu().item())
    raise ValueError(stat)


def _cpu_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


if __name__ == "__main__":
    main()
