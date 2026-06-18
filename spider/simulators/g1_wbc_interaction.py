"""G1 WBC interaction task adapter for SPIDER's sampling optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import torch

from spider.config import Config
from spider.tasks.g1_wbc.constants import MUJOCO_JOINT_NAMES, QPOS_DIM
from spider.tasks.g1_wbc.math_utils import normalize, quat_from_axis_angle, quat_mul
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.rollout import RolloutResult
from spider.tasks.g1_wbc_interaction.layout import InteractionModelLayout
from spider.tasks.g1_wbc_interaction.metrics import (
    InteractionScoreWeights,
    RetargetScoreWeights,
    compute_interaction_rollout_metrics,
    compute_interaction_rollout_scores,
    compute_retarget_rollout_scores,
)
from spider.tasks.g1_wbc_interaction.motion import (
    InteractionMotion,
    slice_interaction_motion,
)
from spider.tasks.g1_wbc_interaction.rollout import (
    InteractionRolloutConfig,
    command_from_full_qpos_trajectory,
    load_interaction_model_and_layout,
    run_interaction_command_rollout,
)


@dataclass
class G1WbcInteractionResult:
    command_qpos: torch.Tensor
    rollout: RolloutResult
    refined_qpos: torch.Tensor
    refined_qvel: torch.Tensor
    controls: torch.Tensor
    executed_controls: torch.Tensor
    infos: list[dict[str, Any]]
    scores: torch.Tensor
    num_windows: int = 0


class G1WbcInteractionSamplingTask:
    """Robot-only MPC command task with full-state object interaction rollout."""

    def __init__(
        self,
        motion: InteractionMotion,
        actor: WbcActor,
        rollout_config: InteractionRolloutConfig,
        *,
        reward_mode: str = "interaction",
        score_weights: InteractionScoreWeights | None = None,
        retarget_score_weights: RetargetScoreWeights | None = None,
    ) -> None:
        self.device = torch.device(rollout_config.device)
        self.motion = motion.to(self.device)
        if self.motion.layout is None:
            raise ValueError("InteractionMotion is missing layout.")
        self.layout = self.motion.layout
        self.actor = actor.to(self.device).eval()
        self.rollout_config = rollout_config
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in {"interaction", "retarget"}:
            raise ValueError(f"Unsupported reward_mode {self.reward_mode!r}.")
        self.score_weights = score_weights or InteractionScoreWeights()
        self.retarget_score_weights = retarget_score_weights or RetargetScoreWeights()
        self.joint_low, self.joint_high = _joint_limits(
            rollout_config.model_path,
            self.device,
        )
        self.current_qpos = self.motion.full_state_qpos()[0].detach().clone()
        self.current_qvel = self.motion.full_state_qvel()[0].detach().clone()
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

    def ref_slice(self, start: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_qpos = _slice_qpos_padded(self.motion.qpos(), start, horizon)
        return torch.tensor([int(start)], dtype=torch.long, device=self.device), base_qpos

    def rollout(
        self,
        config: Config,
        _env,
        controls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, torch.Tensor],
        _env_param: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        start = int(ref_slice[0].detach().cpu().item())
        base_robot_qpos = ref_slice[1].to(self.device)
        horizon = int(controls.shape[1])
        robot_qpos = self.controls_to_robot_qpos(
            controls.permute(1, 0, 2),
            base_robot_qpos[:horizon],
        )
        full_qpos = self._compose_full_command_qpos(robot_qpos, start, horizon)
        command = command_from_full_qpos_trajectory(
            self._window_motion(start, horizon),
            full_qpos,
            _replace_rollout_config(self.rollout_config, int(config.num_samples), horizon),
            preserve_template_first=True,
        )
        rollout = run_interaction_command_rollout(
            command,
            self.actor,
            _replace_rollout_config(self.rollout_config, int(config.num_samples), horizon),
            initial_qpos=self.current_qpos,
            initial_qvel=self.current_qvel,
            initial_last_action=self.current_last_action,
            initial_history_state=self.current_history_state,
            ref_start=start,
        )
        if self.reward_mode == "retarget":
            scores, terms = compute_retarget_rollout_scores(
                self.motion,
                rollout,
                layout=self.layout,
                weights=self.retarget_score_weights,
            )
        else:
            scores, terms = compute_interaction_rollout_scores(
                self.motion,
                rollout,
                layout=self.layout,
                weights=self.score_weights,
            )
        self._last_scores = scores.detach().clone()
        terminate = torch.zeros(int(config.num_samples), dtype=torch.bool, device=self.device)
        info = {key: value.detach() for key, value in terms.items()}
        info["task_score"] = scores.detach()
        return controls, scores, terminate, info

    def execute(self, controls: torch.Tensor, sim_step: int) -> dict[str, Any]:
        execute_steps = int(controls.shape[0])
        base_robot_qpos = _slice_qpos_padded(self.motion.qpos(), sim_step, execute_steps)
        robot_qpos = self.controls_to_robot_qpos(controls[:, None, :], base_robot_qpos)
        full_qpos = self._compose_full_command_qpos(robot_qpos, sim_step, execute_steps)
        command = command_from_full_qpos_trajectory(
            self._window_motion(sim_step, execute_steps),
            full_qpos,
            _replace_rollout_config(self.rollout_config, 1, execute_steps),
            preserve_template_first=False,
        )
        rollout = run_interaction_command_rollout(
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
        self._executed_controls.extend(t.detach().clone() for t in controls)
        self.current_qpos = rollout.qpos[-1, 0].detach().clone()
        self.current_qvel = rollout.qvel[-1, 0].detach().clone()
        self.current_last_action = (
            None
            if rollout.final_last_action is None
            else rollout.final_last_action[0].detach().clone()
        )
        self.current_history_state = rollout.final_history_state
        return {
            "executed_controls_rms": torch.stack(
                [t.square().mean().sqrt() for t in controls]
            ).mean().detach().cpu().numpy(),
        }

    def controls_to_robot_qpos(
        self,
        controls_time_major: torch.Tensor,
        base_qpos: torch.Tensor,
    ) -> torch.Tensor:
        base = base_qpos[:, None, :].expand(-1, int(controls_time_major.shape[1]), -1).clone()
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
    ) -> G1WbcInteractionResult:
        rollout = self._stack_rollout()
        refined_qpos = self.motion.full_state_qpos()[: total_steps + 1].detach().clone()
        refined_qvel = self.motion.full_state_qvel()[: total_steps + 1].detach().clone()
        if rollout.qpos.numel():
            n = min(refined_qpos.shape[0], rollout.qpos.shape[0])
            refined_qpos[:n] = rollout.qpos[:n, 0]
            refined_qvel[:n] = rollout.qvel[:n, 0]
        command_qpos = self.layout.robot_qpos(refined_qpos[:, None, :])
        executed_controls = (
            torch.stack(self._executed_controls, dim=0)
            if self._executed_controls
            else torch.empty(0, QPOS_DIM - 1, dtype=torch.float32, device=self.device)
        )
        return G1WbcInteractionResult(
            command_qpos=command_qpos,
            rollout=rollout,
            refined_qpos=refined_qpos,
            refined_qvel=refined_qvel,
            controls=controls.detach().clone(),
            executed_controls=executed_controls.detach().clone(),
            infos=infos,
            scores=self._last_scores.detach().clone(),
            num_windows=len(infos),
        )

    def metrics(self, rollout: RolloutResult) -> dict[str, float | bool]:
        return compute_interaction_rollout_metrics(self.motion, rollout, layout=self.layout)

    def _compose_full_command_qpos(
        self,
        robot_qpos: torch.Tensor,
        start: int,
        horizon: int,
    ) -> torch.Tensor:
        full = _slice_qpos_padded(self.motion.full_state_qpos(), start, horizon)
        full = full[:, None, :].expand(-1, int(robot_qpos.shape[1]), -1).clone()
        return self.layout.assign_robot_qpos(full, robot_qpos)

    def _window_motion(self, start: int, length: int) -> InteractionMotion:
        return slice_interaction_motion(self.motion, start, length)

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
            self._floor_contact_indicator.append(rollout.floor_contact_indicator[0, 0].detach().clone())
            self._floor_contact_force.append(rollout.floor_contact_force[0, 0].detach().clone())
            self._ref_indices.append(rollout.ref_indices[0, 0].detach().clone())
        self._qpos_trace.extend(t.detach().clone() for t in rollout.qpos[1:, 0])
        self._qvel_trace.extend(t.detach().clone() for t in rollout.qvel[1:, 0])
        self._body_pos_trace.extend(t.detach().clone() for t in rollout.body_pos_w[1:, 0])
        self._body_quat_trace.extend(t.detach().clone() for t in rollout.body_quat_w[1:, 0])
        self._body_lin_vel_trace.extend(t.detach().clone() for t in rollout.body_lin_vel_w[1:, 0])
        self._body_ang_vel_trace.extend(t.detach().clone() for t in rollout.body_ang_vel_w[1:, 0])
        self._contact_indicator.extend(t.detach().clone() for t in rollout.contact_indicator[1:, 0])
        self._contact_force.extend(t.detach().clone() for t in rollout.contact_force[1:, 0])
        self._floor_contact_indicator.extend(
            t.detach().clone() for t in rollout.floor_contact_indicator[1:, 0]
        )
        self._floor_contact_force.extend(
            t.detach().clone() for t in rollout.floor_contact_force[1:, 0]
        )
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
    config: InteractionRolloutConfig,
    num_envs: int,
    max_steps: int | None,
) -> InteractionRolloutConfig:
    from dataclasses import replace

    return replace(config, num_envs=int(num_envs), max_steps=max_steps)


def _slice_qpos_padded(qpos: torch.Tensor, start: int, length: int) -> torch.Tensor:
    start = int(start)
    length = int(length)
    out = qpos[start : start + length]
    if out.shape[0] < length:
        out = torch.cat([out, out[-1:].repeat(length - out.shape[0], 1)], dim=0)
    return out.contiguous()


def _joint_limits(model_path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model, _layout = load_interaction_model_and_layout(model_path)
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
    "G1WbcInteractionResult",
    "G1WbcInteractionSamplingTask",
    "InteractionRolloutConfig",
    "InteractionScoreWeights",
    "RetargetScoreWeights",
]
