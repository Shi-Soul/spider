"""G1 WBC simulator backend for SPIDER's generic sampling optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import torch

from spider.config import Config
from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
)
from spider.tasks.g1_wbc.metrics import compute_rollout_scores
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.obs import G1WbcObservationBuilder, RobotState
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.rollout import (
    G1WbcMujocoWarpEnv,
    RolloutResult,
    WbcRolloutConfig,
)


@dataclass
class G1WbcBackendState:
    last_action: torch.Tensor
    history_state: dict | None
    step_index: int
    command: G1CommandBatch | G1Motion | None
    ref_start: int
    task_context: dict[str, Any] | None
    trace: dict[str, list[torch.Tensor]]
    last_robot_state: RobotState | None
    last_floor_contact: torch.Tensor | None
    last_floor_force: torch.Tensor | None


class G1WbcBackend:
    """Simulation backend implementing SPIDER's rollout lifecycle for G1 WBC."""

    def __init__(
        self,
        actor: WbcActor,
        rollout_config: WbcRolloutConfig,
        *,
        initial_qpos: torch.Tensor,
        initial_qvel: torch.Tensor,
        command: G1CommandBatch | G1Motion | None = None,
    ) -> None:
        self.device = torch.device(rollout_config.device)
        self.rollout_config = rollout_config
        self.sim = G1WbcMujocoWarpEnv(rollout_config)
        self.actor = actor.to(self.device).eval()
        self.command = command.to(self.device) if command is not None else None
        self.obs_builder: G1WbcObservationBuilder | None = None
        self.task_context: dict[str, Any] | None = None
        self.last_action = torch.zeros(
            int(rollout_config.num_envs),
            ACTION_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.step_index = 0
        self.ref_start = 0
        self.trace = _empty_trace()
        self.last_robot_state: RobotState | None = None
        self.last_floor_contact: torch.Tensor | None = None
        self.last_floor_force: torch.Tensor | None = None

        self.sim.reset(initial_qpos, initial_qvel)
        self._refresh_initial_trace(ref_index=0)
        if self.command is not None:
            self.set_command(self.command, ref_start=0)

    @property
    def num_envs(self) -> int:
        return int(self.rollout_config.num_envs)

    def reset_physical_state(
        self,
        initial_qpos: torch.Tensor,
        initial_qvel: torch.Tensor,
        *,
        ref_index: int = 0,
    ) -> None:
        """Reset backend-owned physical state before a new independent run."""

        self.sim.reset(initial_qpos, initial_qvel)
        self.command = None
        self.obs_builder = None
        self.task_context = None
        self.last_action = torch.zeros(
            self.num_envs,
            ACTION_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.step_index = 0
        self.ref_start = int(ref_index)
        self._refresh_initial_trace(ref_index=int(ref_index))

    def set_command(
        self,
        command: G1CommandBatch | G1Motion,
        *,
        ref_start: int,
        initial_last_action: torch.Tensor | None = None,
        initial_history_state: dict | None = None,
    ) -> None:
        command = command.to(self.device)
        if isinstance(command, G1CommandBatch) and command.num_envs != self.num_envs:
            raise ValueError(
                f"Command batch has {command.num_envs} envs, backend has {self.num_envs}."
            )
        self.command = command
        self.obs_builder = G1WbcObservationBuilder(
            motion=command,
            num_envs=self.num_envs,
            default_joint_pos=self.sim.default_joint_pos,
            device=self.device,
        )
        self.obs_builder.load_history_state_dict(initial_history_state)
        if initial_last_action is None:
            self.last_action = torch.zeros(
                self.num_envs,
                ACTION_DIM,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.last_action = _batch_last_action(
                initial_last_action,
                self.num_envs,
                self.device,
            )
        self.step_index = 0
        self.ref_start = int(ref_start)
        self.trace = _empty_trace()
        self._refresh_initial_trace(ref_index=self.ref_start)

    def step(self) -> None:
        if self.command is None or self.obs_builder is None:
            raise RuntimeError("G1 WBC backend command is not configured.")
        state = self.sim.robot_state()
        ref_idx_scalar = min(
            max(self.step_index + int(self.rollout_config.ref_offset), 0),
            self.command.num_frames - 1,
        )
        ref_idx = torch.full(
            (self.num_envs,),
            ref_idx_scalar,
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            obs = self.obs_builder.compute(state, ref_idx, self.last_action)
            action = self.actor(obs)
            ctrl = self.sim.action_to_control(action)
            self.sim.step_control(ctrl)

        self.step_index += 1
        self.last_action = action.detach().clone()
        self._append_current_state(action, ctrl, ref_idx + self.ref_start)

    def save_state(self) -> G1WbcBackendState:
        if self.last_robot_state is None:
            self.last_robot_state = self.sim.robot_state()
            self.last_floor_contact, self.last_floor_force = self.sim.floor_contact()
        return G1WbcBackendState(
            last_action=self.last_action.detach().clone(),
            history_state=(
                None
                if self.obs_builder is None
                else self.obs_builder.history_state_dict()
            ),
            step_index=int(self.step_index),
            command=self.command,
            ref_start=int(self.ref_start),
            task_context=_clone_task_context(self.task_context),
            trace=_clone_trace(self.trace),
            last_robot_state=_clone_robot_state(self.last_robot_state),
            last_floor_contact=_clone_optional_tensor(self.last_floor_contact),
            last_floor_force=_clone_optional_tensor(self.last_floor_force),
        )

    def load_state(self, state: G1WbcBackendState) -> "G1WbcBackend":
        if state.last_robot_state is not None:
            self.sim.reset(
                state.last_robot_state.qpos,
                state.last_robot_state.qvel,
            )
        self.command = state.command
        self.task_context = _clone_task_context(state.task_context)
        if self.command is not None:
            self.obs_builder = G1WbcObservationBuilder(
                motion=self.command,
                num_envs=self.num_envs,
                default_joint_pos=self.sim.default_joint_pos,
                device=self.device,
            )
            self.obs_builder.load_history_state_dict(state.history_state)
        else:
            self.obs_builder = None
        self.last_action = state.last_action.detach().clone()
        self.step_index = int(state.step_index)
        self.ref_start = int(state.ref_start)
        self.trace = _clone_trace(state.trace)
        self.last_robot_state = _clone_robot_state(state.last_robot_state)
        self.last_floor_contact = _clone_optional_tensor(state.last_floor_contact)
        self.last_floor_force = _clone_optional_tensor(state.last_floor_force)
        return self

    def rollout_result(self) -> RolloutResult:
        if not self.trace["qpos"]:
            raise RuntimeError("No WBC rollout trace has been recorded.")
        actions = self.trace["actions"]
        controls = self.trace["controls"]
        if not actions:
            actions_tensor = torch.empty(
                0,
                self.num_envs,
                ACTION_DIM,
                dtype=torch.float32,
                device=self.device,
            )
            controls_tensor = torch.empty_like(actions_tensor)
        else:
            actions_tensor = torch.stack(actions, dim=0)
            controls_tensor = torch.stack(controls, dim=0)
        return RolloutResult(
            qpos=torch.stack(self.trace["qpos"], dim=0),
            qvel=torch.stack(self.trace["qvel"], dim=0),
            body_pos_w=torch.stack(self.trace["body_pos_w"], dim=0),
            body_quat_w=torch.stack(self.trace["body_quat_w"], dim=0),
            body_lin_vel_w=torch.stack(self.trace["body_lin_vel_w"], dim=0),
            body_ang_vel_w=torch.stack(self.trace["body_ang_vel_w"], dim=0),
            actions=actions_tensor,
            controls=controls_tensor,
            contact_indicator=torch.stack(self.trace["contact_indicator"], dim=0),
            contact_force=torch.stack(self.trace["contact_force"], dim=0),
            floor_contact_indicator=torch.stack(
                self.trace["floor_contact_indicator"], dim=0
            ),
            floor_contact_force=torch.stack(self.trace["floor_contact_force"], dim=0),
            ref_indices=torch.stack(self.trace["ref_indices"], dim=0),
            final_last_action=self.last_action.detach().clone(),
            final_history_state=(
                None
                if self.obs_builder is None
                else self.obs_builder.history_state_dict()
            ),
        )

    def _refresh_initial_trace(self, *, ref_index: int) -> None:
        state = self.sim.robot_state()
        floor_contact, floor_force = self.sim.floor_contact()
        self.last_robot_state = state
        self.last_floor_contact = floor_contact
        self.last_floor_force = floor_force
        self.trace = _empty_trace()
        _append_backend_state(
            self.trace,
            state,
            floor_contact,
            floor_force,
            torch.full(
                (self.num_envs,),
                int(ref_index),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _append_current_state(
        self,
        action: torch.Tensor,
        ctrl: torch.Tensor,
        ref_indices: torch.Tensor,
    ) -> None:
        state = self.sim.robot_state()
        floor_contact, floor_force = self.sim.floor_contact()
        self.last_robot_state = state
        self.last_floor_contact = floor_contact
        self.last_floor_force = floor_force
        _append_backend_state(
            self.trace,
            state,
            floor_contact,
            floor_force,
            ref_indices,
        )
        self.trace["actions"].append(action.detach().clone())
        self.trace["controls"].append(ctrl.detach().clone())


def setup_env(
    config: Config,
    ref_data: dict[str, Any],
) -> G1WbcBackend:
    motion = ref_data["motion"]
    actor = ref_data["actor"]
    rollout_config = _replace_rollout_config(
        ref_data["rollout_config"],
        int(config.num_samples),
        None,
    )
    return G1WbcBackend(
        actor,
        rollout_config,
        initial_qpos=motion.qpos()[0],
        initial_qvel=motion.qvel()[0],
    )


def _replace_rollout_config(
    config: WbcRolloutConfig,
    num_envs: int,
    max_steps: int | None,
) -> WbcRolloutConfig:
    return replace(config, num_envs=int(num_envs), max_steps=max_steps)


def save_state(env: G1WbcBackend) -> G1WbcBackendState:
    return env.save_state()


def load_state(env: G1WbcBackend, state: G1WbcBackendState) -> G1WbcBackend:
    return env.load_state(state)


def step_env(_config: Config, env: G1WbcBackend, _ctrl: torch.Tensor) -> None:
    env.step()


def get_reward(
    config: Config,
    env: G1WbcBackend,
    _ref,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zeros = torch.zeros(int(config.num_samples), dtype=torch.float32, device=config.device)
    return zeros, _zero_score_info(zeros)


def get_terminal_reward(
    config: Config,
    env: G1WbcBackend,
    _ref,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if env.task_context is None:
        raise RuntimeError("G1 WBC backend is missing task_context.")
    rollout = env.rollout_result()
    _, terms = compute_rollout_scores(env.task_context["motion"], rollout)
    scores = _score_from_terms(
        terms,
        env.task_context["reward_weights"],
    )
    env.task_context["last_scores"] = scores.detach().clone()
    horizon = max(int(env.step_index), 1)
    reward = scores * horizon
    info = {key: value.detach() * horizon for key, value in terms.items()}
    info["task_score"] = scores.detach() * horizon
    return reward, info


def get_terminate(config: Config, _env: G1WbcBackend, _ref) -> torch.Tensor:
    return torch.zeros(int(config.num_samples), dtype=torch.bool, device=config.device)


def get_trace(config: Config, _env: G1WbcBackend) -> torch.Tensor:
    return torch.empty(
        int(config.num_samples),
        0,
        3,
        dtype=torch.float32,
        device=config.device,
    )


def save_env_params(_config: Config, env: G1WbcBackend) -> dict[str, Any]:
    return {}


def load_env_params(
    config: Config,
    env: G1WbcBackend,
    env_param: dict[str, Any],
) -> G1WbcBackend:
    if not env_param:
        return env
    command = env_param.get("command")
    if command is None:
        return env
    task_context = env_param.get("task_context")
    if task_context is not None:
        env.task_context = task_context
    env.set_command(
        command,
        ref_start=int(env_param.get("ref_start", 0)),
        initial_last_action=env_param.get("initial_last_action"),
        initial_history_state=env_param.get("initial_history_state"),
    )
    return env


def copy_sample_state(
    _config: Config,
    _env: G1WbcBackend,
    _src_indices: torch.Tensor,
    _dst_indices: torch.Tensor,
) -> None:
    raise NotImplementedError("G1 WBC does not enable terminate_resample.")


def _score_from_terms(
    terms: dict[str, torch.Tensor],
    reward_weights: dict[str, float],
) -> torch.Tensor:
    missing = sorted(key for key in reward_weights if key not in terms)
    if missing:
        raise KeyError(f"Reward weights reference missing terms: {missing}")
    loss = None
    for key, weight in reward_weights.items():
        term = terms[key] * float(weight)
        loss = term if loss is None else loss + term
    if loss is None:
        first = next(iter(terms.values()))
        loss = torch.zeros_like(first)
    return -loss


def _zero_score_info(zeros: torch.Tensor) -> dict[str, torch.Tensor]:
    keys = {
        "root_pos_error",
        "root_rot_error",
        "joint_pos_error",
        "body_global_pos_error",
        "body_global_rot_error",
        "ee_global_pos_error",
        "ee_global_rot_error",
        "ee_local_pos_error",
        "ee_local_rot_error",
        "hand_global_pos_error",
        "hand_global_rot_error",
        "hand_local_pos_error",
        "hand_local_rot_error",
        "body_local_pos_error",
        "body_local_rot_error",
        "contact_mismatch",
        "contact_false_positive",
        "contact_false_negative",
        "contact_switch",
        "contact_force_excess",
        "contact_force_delta",
        "bad_floor_contact",
        "bad_floor_force_excess",
        "action_delta",
        "control_delta",
        "joint_acc",
        "joint_jerk",
        "task_score",
    }
    return {key: zeros.clone() for key in keys}


def _empty_trace() -> dict[str, list[torch.Tensor]]:
    return {
        "qpos": [],
        "qvel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "actions": [],
        "controls": [],
        "contact_indicator": [],
        "contact_force": [],
        "floor_contact_indicator": [],
        "floor_contact_force": [],
        "ref_indices": [],
    }


def _append_backend_state(
    trace: dict[str, list[torch.Tensor]],
    state: RobotState,
    floor_contact: torch.Tensor,
    floor_force: torch.Tensor,
    ref_indices: torch.Tensor,
) -> None:
    trace["qpos"].append(state.qpos.detach().clone())
    trace["qvel"].append(state.qvel.detach().clone())
    trace["body_pos_w"].append(state.body_pos_w.detach().clone())
    trace["body_quat_w"].append(state.body_quat_w.detach().clone())
    trace["body_lin_vel_w"].append(state.body_lin_vel_w.detach().clone())
    trace["body_ang_vel_w"].append(state.body_ang_vel_w.detach().clone())
    trace["contact_indicator"].append(floor_contact[:, :2].detach().clone())
    trace["contact_force"].append(floor_force[:, :2].detach().clone())
    trace["floor_contact_indicator"].append(floor_contact.detach().clone())
    trace["floor_contact_force"].append(floor_force.detach().clone())
    trace["ref_indices"].append(ref_indices.detach().clone())


def _clone_trace(trace: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
    return {
        key: [value.detach().clone() for value in values]
        for key, values in trace.items()
    }


def _clone_task_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if context is None else dict(context)


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _clone_robot_state(state: RobotState | None) -> RobotState | None:
    if state is None:
        return None
    return RobotState(
        qpos=state.qpos.detach().clone(),
        qvel=state.qvel.detach().clone(),
        body_pos_w=state.body_pos_w.detach().clone(),
        body_quat_w=state.body_quat_w.detach().clone(),
        body_lin_vel_w=state.body_lin_vel_w.detach().clone(),
        body_ang_vel_w=state.body_ang_vel_w.detach().clone(),
        base_ang_vel_b=_clone_optional_tensor(state.base_ang_vel_b),
    )


def _batch_last_action(
    value: torch.Tensor,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    value = value.to(device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.view(1, ACTION_DIM).expand(num_envs, ACTION_DIM)
    if value.shape != (num_envs, ACTION_DIM):
        if value.shape == (1, ACTION_DIM):
            value = value.expand(num_envs, ACTION_DIM)
        else:
            raise ValueError(
                f"Expected last action {(num_envs, ACTION_DIM)}, got {value.shape}."
            )
    return value.contiguous()


__all__ = [
    "G1CommandBatch",
    "G1Motion",
    "G1WbcBackend",
    "G1WbcBackendState",
    "RolloutResult",
    "WbcActor",
    "WbcRolloutConfig",
    "copy_sample_state",
    "get_reward",
    "get_terminal_reward",
    "get_terminate",
    "get_trace",
    "load_env_params",
    "load_state",
    "save_env_params",
    "save_state",
    "setup_env",
    "step_env",
]
