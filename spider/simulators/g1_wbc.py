"""G1 WBC task adapter for SPIDER's generic sampling optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import mujoco

from spider.config import Config
from spider.tasks.g1_wbc.constants import MUJOCO_JOINT_NAMES, POLICY_DT, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.math_utils import (
    normalize,
    quat_from_axis_angle,
    quat_mul,
)
from spider.tasks.g1_wbc.metrics import compute_rollout_metrics, compute_rollout_scores
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    command_batch_from_qpos_trajectory,
    load_wbc_model,
    run_command_rollout,
)

G1WbcObjective = Literal["g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global"]

REWARD_WEIGHT_PRESETS: dict[G1WbcObjective, dict[str, float]] = {
    "g1_wbc_joint_global": {
        "bad_floor_contact": 45.0,
        "bad_floor_force_excess": 10.0,
        "contact_switch": 12.0,
        "contact_force_delta": 2.5,
        "contact_false_positive": 1.5,
        "contact_false_negative": 0.4,
        "control_delta": 1.8,
        "action_delta": 0.6,
        "joint_acc": 0.006,
        "joint_jerk": 0.0012,
        "body_global_pos_error": 4.0,
        "body_global_rot_error": 0.8,
        "ee_global_pos_error": 1.5,
        "ee_global_rot_error": 0.3,
    },
    "g1_wbc_joint": {
        "bad_floor_contact": 35.0,
        "bad_floor_force_excess": 8.0,
        "contact_switch": 10.0,
        "contact_force_delta": 2.0,
        "contact_false_positive": 0.8,
        "contact_false_negative": 0.3,
        "control_delta": 1.6,
        "action_delta": 0.5,
        "joint_acc": 0.006,
        "joint_jerk": 0.0012,
        "body_local_pos_error": 26.0,
        "body_local_rot_error": 3.0,
        "joint_pos_error": 2.1,
        "ee_local_pos_error": 6.0,
        "ee_local_rot_error": 1.2,
        "body_global_pos_error": 0.8,
        "body_global_rot_error": 0.2,
        "ee_global_pos_error": 0.3,
    },
    "g1_wbc_ee": {
        "bad_floor_contact": 35.0,
        "bad_floor_force_excess": 8.0,
        "contact_switch": 10.0,
        "contact_force_delta": 2.0,
        "contact_false_positive": 0.5,
        "contact_false_negative": 0.2,
        "control_delta": 2.0,
        "action_delta": 0.6,
        "joint_acc": 0.006,
        "joint_jerk": 0.0015,
        "hand_global_pos_error": 35.0,
        "hand_global_rot_error": 3.0,
        "hand_local_pos_error": 8.0,
        "hand_local_rot_error": 1.5,
        "ee_global_pos_error": 2.0,
        "ee_global_rot_error": 0.4,
        "body_global_pos_error": 0.8,
        "body_local_pos_error": 0.8,
    },
}


@dataclass
class G1WbcSpiderResult:
    command: G1CommandBatch
    rollout: RolloutResult
    refined_qpos: torch.Tensor
    controls: torch.Tensor
    infos: list[dict[str, Any]]
    scores: torch.Tensor
    num_windows: int = 0


def load_reward_weights(path: str | Path, mode: G1WbcObjective) -> dict[str, float]:
    raw = json.loads(Path(path).expanduser().read_text())
    if mode in raw and isinstance(raw[mode], dict):
        raw = raw[mode]
    if not isinstance(raw, dict):
        raise ValueError(f"Reward weight file must contain a JSON object: {path}")
    return {str(key): float(value) for key, value in raw.items()}


def reward_weights_for(
    mode: G1WbcObjective,
    weights: dict[str, float] | None,
) -> dict[str, float]:
    return weights or REWARD_WEIGHT_PRESETS[mode]


class G1WbcSamplingTask:
    """Task adapter consumed by SPIDER's generic sampling optimizer."""

    def __init__(
        self,
        motion: G1Motion,
        actor: WbcActor,
        rollout_config: WbcRolloutConfig,
        *,
        mode: G1WbcObjective,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        self.device = torch.device(rollout_config.device)
        self.motion = motion.to(self.device)
        self.actor = actor.to(self.device).eval()
        self.rollout_config = rollout_config
        self.mode = mode
        self.reward_weights = reward_weights
        self.joint_low, self.joint_high = _joint_limits(rollout_config, self.device)
        self.current_qpos = self.motion.qpos()[0].detach().clone()
        self.current_qvel = self.motion.qvel()[0].detach().clone()
        self.current_last_action: torch.Tensor | None = None
        self.current_history_state = None
        self._qpos_trace: list[torch.Tensor] = []
        self._qvel_trace: list[torch.Tensor] = []
        self._body_pos_trace: list[torch.Tensor] = []
        self._body_quat_trace: list[torch.Tensor] = []
        self._body_lin_vel_trace: list[torch.Tensor] = []
        self._body_ang_vel_trace: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._controls: list[torch.Tensor] = []
        self._contact_indicator: list[torch.Tensor] = []
        self._contact_force: list[torch.Tensor] = []
        self._floor_contact_indicator: list[torch.Tensor] = []
        self._floor_contact_force: list[torch.Tensor] = []
        self._ref_indices: list[torch.Tensor] = []
        self._executed_controls: list[torch.Tensor] = []
        self._last_scores = torch.empty(0, dtype=torch.float32, device=self.device)

    def initial_controls(self, config: Config) -> torch.Tensor:
        return torch.zeros(
            int(config.horizon_steps),
            int(config.nu),
            dtype=torch.float32,
            device=self.device,
        )

    def tail_controls(self, _sim_step: int, steps: int) -> torch.Tensor:
        return torch.zeros(
            int(steps),
            QPOS_DIM - 1,
            dtype=torch.float32,
            device=self.device,
        )

    def ref_slice(self, start: int, horizon: int) -> tuple[torch.Tensor, ...]:
        base_qpos = _slice_qpos_padded(self.motion.qpos(), start, horizon)
        return (
            torch.tensor([int(start)], dtype=torch.long, device=self.device),
            base_qpos,
        )

    def rollout(
        self,
        config: Config,
        _env,
        controls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        _env_param: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        start = int(ref_slice[0].detach().cpu().item())
        base_qpos = ref_slice[1].to(self.device)
        horizon = int(controls.shape[1])
        qpos = self.controls_to_qpos(controls.permute(1, 0, 2), base_qpos[:horizon])
        command = command_batch_from_qpos_trajectory(
            self._window_motion(start, horizon),
            qpos,
            self.rollout_config,
            preserve_template_first=True,
        )
        rollout = run_command_rollout(
            command,
            self.actor,
            _replace_rollout_config(self.rollout_config, config.num_samples, horizon),
            initial_qpos=self.current_qpos,
            initial_qvel=self.current_qvel,
            initial_last_action=self.current_last_action,
            initial_history_state=self.current_history_state,
            ref_start=start,
        )
        _, terms = compute_rollout_scores(self.motion, rollout)
        scores = _score_from_terms(
            terms,
            self.mode,
            self.reward_weights,
        )
        self._last_scores = scores.detach().clone()
        terminate = torch.zeros(
            int(config.num_samples),
            dtype=torch.bool,
            device=self.device,
        )
        info = {key: value.detach() for key, value in terms.items()}
        info["task_score"] = scores.detach()
        return controls, scores, terminate, info

    def execute(self, controls: torch.Tensor, sim_step: int) -> dict[str, Any]:
        execute_steps = int(controls.shape[0])
        base_qpos = _slice_qpos_padded(self.motion.qpos(), sim_step, execute_steps)
        qpos = self.controls_to_qpos(controls[:, None, :], base_qpos)
        info = self.execute_qpos_command(qpos, sim_step)
        self._executed_controls.extend(t.detach().clone() for t in controls)
        return {
            **info,
            "executed_controls_rms": torch.stack(
                [t.square().mean().sqrt() for t in controls]
            ).mean().detach().cpu().numpy(),
        }

    def execute_qpos_command(self, qpos: torch.Tensor, sim_step: int) -> dict[str, Any]:
        """Execute a precomputed qpos command chunk through the MPC rollout path."""

        qpos = qpos.to(self.device, dtype=torch.float32)
        if qpos.ndim == 3:
            if qpos.shape[1] != 1:
                raise ValueError(f"Expected single-env command qpos, got {qpos.shape}.")
            qpos = qpos[:, 0]
        if qpos.ndim != 2 or qpos.shape[-1] != QPOS_DIM:
            raise ValueError(f"Expected command qpos shape (T, {QPOS_DIM}), got {qpos.shape}.")

        execute_steps = int(qpos.shape[0])
        command = command_batch_from_qpos_trajectory(
            self._window_motion(sim_step, execute_steps),
            qpos[:, None, :],
            self.rollout_config,
            preserve_template_first=False,
            kinematics_batch_size=1,
        )
        rollout = run_command_rollout(
            command,
            self.actor,
            _replace_rollout_config(self.rollout_config, 1, execute_steps),
            initial_qpos=self.current_qpos,
            initial_qvel=self.current_qvel,
            initial_last_action=self.current_last_action,
            initial_history_state=self.current_history_state,
            ref_start=sim_step,
        )
        self._append_rollout(rollout)
        self.current_qpos = rollout.qpos[-1, 0].detach().clone()
        self.current_qvel = rollout.qvel[-1, 0].detach().clone()
        self.current_last_action = (
            None
            if rollout.final_last_action is None
            else rollout.final_last_action[0].detach().clone()
        )
        self.current_history_state = rollout.final_history_state
        return {}

    def controls_to_qpos(
        self,
        controls_time_major: torch.Tensor,
        base_qpos: torch.Tensor,
    ) -> torch.Tensor:
        base = base_qpos[:, None, :].expand(
            -1,
            int(controls_time_major.shape[1]),
            -1,
        ).clone()
        base[..., :3] = base[..., :3] + controls_time_major[..., :3]
        delta_quat = quat_from_axis_angle(controls_time_major[..., 3:6])
        base[..., 3:7] = normalize(quat_mul(delta_quat, base[..., 3:7]))
        base[..., 7:] = torch.clamp(
            base[..., 7:] + controls_time_major[..., 6:],
            self.joint_low,
            self.joint_high,
        )
        return base.contiguous()

    def build_result(
        self,
        controls: torch.Tensor,
        infos: list[dict[str, Any]],
        *,
        total_steps: int,
    ) -> G1WbcSpiderResult:
        rollout = self._stack_rollout()
        refined_qpos = self.motion.qpos()[: total_steps + 1].detach().clone()
        if self._executed_controls:
            executed = torch.stack(self._executed_controls, dim=0)
            base = self.motion.qpos()[: executed.shape[0]]
            refined_qpos[: executed.shape[0]] = self.controls_to_qpos(
                executed[:, None, :],
                base,
            )[:, 0]
        command = command_batch_from_qpos_trajectory(
            self.motion,
            refined_qpos[:, None, :],
            _replace_rollout_config(self.rollout_config, 1, total_steps),
            preserve_template_first=False,
        )
        return G1WbcSpiderResult(
            command=command,
            rollout=rollout,
            refined_qpos=refined_qpos,
            controls=controls.detach().clone(),
            infos=infos,
            scores=self._last_scores.detach().clone(),
            num_windows=len(infos),
        )

    def _window_motion(self, start: int, length: int) -> G1Motion:
        return _slice_motion_padded(self.motion, start, length)

    def _append_rollout(self, rollout: RolloutResult) -> None:
        if not self._qpos_trace:
            self._qpos_trace.append(rollout.qpos[0, 0].detach().clone())
            self._qvel_trace.append(rollout.qvel[0, 0].detach().clone())
            self._body_pos_trace.append(rollout.body_pos_w[0, 0].detach().clone())
            self._body_quat_trace.append(rollout.body_quat_w[0, 0].detach().clone())
            self._body_lin_vel_trace.append(rollout.body_lin_vel_w[0, 0].detach().clone())
            self._body_ang_vel_trace.append(rollout.body_ang_vel_w[0, 0].detach().clone())
            self._contact_indicator.append(rollout.contact_indicator[0, 0].detach().clone())
            self._contact_force.append(rollout.contact_force[0, 0].detach().clone())
            self._floor_contact_indicator.append(_floor_contact_indicator(rollout)[0, 0].detach().clone())
            self._floor_contact_force.append(_floor_contact_force(rollout)[0, 0].detach().clone())
            self._ref_indices.append(rollout.ref_indices[0, 0].detach().clone())

        self._qpos_trace.extend(t.detach().clone() for t in rollout.qpos[1:, 0])
        self._qvel_trace.extend(t.detach().clone() for t in rollout.qvel[1:, 0])
        self._body_pos_trace.extend(t.detach().clone() for t in rollout.body_pos_w[1:, 0])
        self._body_quat_trace.extend(t.detach().clone() for t in rollout.body_quat_w[1:, 0])
        self._body_lin_vel_trace.extend(t.detach().clone() for t in rollout.body_lin_vel_w[1:, 0])
        self._body_ang_vel_trace.extend(t.detach().clone() for t in rollout.body_ang_vel_w[1:, 0])
        self._contact_indicator.extend(t.detach().clone() for t in rollout.contact_indicator[1:, 0])
        self._contact_force.extend(t.detach().clone() for t in rollout.contact_force[1:, 0])
        self._floor_contact_indicator.extend(t.detach().clone() for t in _floor_contact_indicator(rollout)[1:, 0])
        self._floor_contact_force.extend(t.detach().clone() for t in _floor_contact_force(rollout)[1:, 0])
        self._ref_indices.extend(t.detach().clone() for t in rollout.ref_indices[1:, 0])
        self._actions.extend(t.detach().clone() for t in rollout.actions[:, 0])
        self._controls.extend(t.detach().clone() for t in rollout.controls[:, 0])

    def _stack_rollout(self) -> RolloutResult:
        return RolloutResult(
            qpos=torch.stack(self._qpos_trace, dim=0)[:, None, :],
            qvel=torch.stack(self._qvel_trace, dim=0)[:, None, :],
            body_pos_w=torch.stack(self._body_pos_trace, dim=0)[:, None, :, :],
            body_quat_w=torch.stack(self._body_quat_trace, dim=0)[:, None, :, :],
            body_lin_vel_w=torch.stack(self._body_lin_vel_trace, dim=0)[:, None, :, :],
            body_ang_vel_w=torch.stack(self._body_ang_vel_trace, dim=0)[:, None, :, :],
            actions=torch.stack(self._actions, dim=0)[:, None, :],
            controls=torch.stack(self._controls, dim=0)[:, None, :],
            contact_indicator=torch.stack(self._contact_indicator, dim=0)[:, None, :],
            contact_force=torch.stack(self._contact_force, dim=0)[:, None, :],
            floor_contact_indicator=torch.stack(self._floor_contact_indicator, dim=0)[:, None, :],
            floor_contact_force=torch.stack(self._floor_contact_force, dim=0)[:, None, :],
            ref_indices=torch.stack(self._ref_indices, dim=0)[:, None],
        )


def _replace_rollout_config(
    config: WbcRolloutConfig,
    num_envs: int,
    max_steps: int | None,
) -> WbcRolloutConfig:
    from dataclasses import replace

    return replace(config, num_envs=int(num_envs), max_steps=max_steps)


def _score_from_terms(
    terms: dict[str, torch.Tensor],
    mode: G1WbcObjective,
    reward_weights: dict[str, float] | None,
) -> torch.Tensor:
    weights = reward_weights_for(mode, reward_weights)
    missing = sorted(key for key in weights if key not in terms)
    if missing:
        raise KeyError(f"Reward weights reference missing terms: {missing}")
    loss = None
    for key, weight in weights.items():
        term = terms[key] * float(weight)
        loss = term if loss is None else loss + term
    if loss is None:
        first = next(iter(terms.values()))
        loss = torch.zeros_like(first)
    return -loss


def _slice_qpos_padded(qpos: torch.Tensor, start: int, length: int) -> torch.Tensor:
    start = int(start)
    length = int(length)
    out = qpos[start : start + length]
    if out.shape[0] < length:
        out = torch.cat([out, out[-1:].repeat(length - out.shape[0], 1)], dim=0)
    return out.contiguous()


def _slice_motion_padded(motion: G1Motion, start: int, length: int) -> G1Motion:
    start = int(start)
    length = int(length)

    def sl(value: torch.Tensor) -> torch.Tensor:
        out = value[start : start + length]
        if out.shape[0] < length:
            repeats = [length - out.shape[0]] + [1] * (out.ndim - 1)
            out = torch.cat([out, out[-1:].repeat(*repeats)], dim=0)
        return out.contiguous()

    return G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=motion.fps,
        joint_pos=sl(motion.joint_pos),
        joint_vel=sl(motion.joint_vel),
        body_pos_w=sl(motion.body_pos_w),
        body_quat_w=sl(motion.body_quat_w),
        body_lin_vel_w=sl(motion.body_lin_vel_w),
        body_ang_vel_w=sl(motion.body_ang_vel_w),
        contact=sl(motion.contact),
    )


def _floor_contact_indicator(rollout: RolloutResult) -> torch.Tensor:
    if rollout.floor_contact_indicator is not None:
        return rollout.floor_contact_indicator
    other = torch.zeros(
        *rollout.contact_indicator.shape[:-1],
        1,
        dtype=rollout.contact_indicator.dtype,
        device=rollout.contact_indicator.device,
    )
    return torch.cat([rollout.contact_indicator, other], dim=-1)


def _floor_contact_force(rollout: RolloutResult) -> torch.Tensor:
    if rollout.floor_contact_force is not None:
        return rollout.floor_contact_force
    other = torch.zeros(
        *rollout.contact_force.shape[:-1],
        1,
        dtype=rollout.contact_force.dtype,
        device=rollout.contact_force.device,
    )
    return torch.cat([rollout.contact_force, other], dim=-1)


def _joint_limits(
    rollout_config: WbcRolloutConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = load_wbc_model(rollout_config.model_path)
    low = []
    high = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}"
            )
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {joint_name}")
        if int(model.jnt_limited[joint_id]):
            low.append(float(model.jnt_range[joint_id, 0]))
            high.append(float(model.jnt_range[joint_id, 1]))
        else:
            low.append(-float("inf"))
            high.append(float("inf"))
    return (
        torch.tensor(low, dtype=torch.float32, device=device),
        torch.tensor(high, dtype=torch.float32, device=device),
    )


__all__ = [
    "G1CommandBatch",
    "G1Motion",
    "G1WbcSamplingTask",
    "G1WbcSpiderResult",
    "G1WbcObjective",
    "REWARD_WEIGHT_PRESETS",
    "RolloutResult",
    "WbcActor",
    "WbcRolloutConfig",
    "compute_rollout_metrics",
    "load_reward_weights",
    "reward_weights_for",
]
