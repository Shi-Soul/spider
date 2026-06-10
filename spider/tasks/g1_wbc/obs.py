"""Observation construction for the G1 WBC policy."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ANCHOR_BODY_NAME,
    COMMAND_BODY_NAMES,
    LIMB_EE_BODY_NAMES,
    OBS_DIM,
    OBS_HISTORY_LENGTH,
    TRACKING_ANCHOR_BODY_NAME,
)
from spider.tasks.g1_wbc.math_utils import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion


class HistoryBuffer:
    """Fixed-length batched history with tracking_bfm-style first-frame backfill."""

    def __init__(self, num_envs: int, history_length: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.history_length = int(history_length)
        self.device = torch.device(device)
        self._buffer: torch.Tensor | None = None
        self._pointer = -1
        self._num_pushes = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    def append(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != self.num_envs:
            raise ValueError(f"Expected batch {self.num_envs}, got {value.shape[0]}")
        value = value.to(self.device)
        if self._buffer is None:
            self._buffer = torch.empty(
                (self.history_length, *value.shape),
                dtype=value.dtype,
                device=self.device,
            )
        self._pointer = (self._pointer + 1) % self.history_length
        self._buffer[self._pointer] = value
        first = self._num_pushes == 0
        if torch.any(first):
            self._buffer[:, first] = value[first]
        self._num_pushes += 1
        return self.flat()

    def flat(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("History buffer is not initialized.")
        idx = (
            torch.arange(self.history_length, device=self.device)
            + self._pointer
            + 1
        ) % self.history_length
        ordered = self._buffer.index_select(0, idx).transpose(0, 1)
        return ordered.reshape(self.num_envs, -1)


@dataclass
class RobotState:
    qpos: torch.Tensor
    qvel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor


@dataclass
class G1WbcObservationBuilder:
    """Build the actor observation used by the WXY G1 WBC checkpoints."""

    motion: G1Motion | G1CommandBatch
    num_envs: int
    default_joint_pos: torch.Tensor
    device: torch.device | str
    histories: dict[str, HistoryBuffer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        for name in (
            "ref_limb_ee_pose_b",
            "robot_limb_ee_pose_b",
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ):
            self.histories[name] = HistoryBuffer(
                self.num_envs, OBS_HISTORY_LENGTH, device
            )
        self.default_joint_pos = self.default_joint_pos.to(device).view(1, ACTION_DIM)
        self._command_body_indices = torch.tensor(
            [self.motion.body_index[name] for name in COMMAND_BODY_NAMES],
            dtype=torch.long,
            device=device,
        )
        self._limb_indices = torch.tensor(
            [COMMAND_BODY_NAMES.index(name) for name in LIMB_EE_BODY_NAMES],
            dtype=torch.long,
            device=device,
        )
        self._anchor_index = COMMAND_BODY_NAMES.index(ANCHOR_BODY_NAME)
        self._tracking_anchor_index = COMMAND_BODY_NAMES.index(TRACKING_ANCHOR_BODY_NAME)

    def _limb_pose_in_anchor_frame(
        self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor
    ) -> torch.Tensor:
        limb_pos_w = body_pos_w[:, self._limb_indices]
        limb_quat_w = body_quat_w[:, self._limb_indices]
        anchor_pos_w = body_pos_w[:, self._anchor_index : self._anchor_index + 1]
        anchor_quat_w = body_quat_w[:, self._anchor_index : self._anchor_index + 1]
        anchor_pos_w = anchor_pos_w.expand(-1, len(self._limb_indices), -1)
        anchor_quat_w = anchor_quat_w.expand(-1, len(self._limb_indices), -1)
        pos_b, quat_b = subtract_frame_transforms(
            anchor_pos_w, anchor_quat_w, limb_pos_w, limb_quat_w
        )
        rot6d = matrix_from_quat(quat_b)[..., :2].reshape(
            body_pos_w.shape[0], len(self._limb_indices), 6
        )
        return torch.cat([pos_b, rot6d], dim=-1).reshape(body_pos_w.shape[0], -1)

    def _ref_fields(self, ref_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        ref_indices = ref_indices.clamp(0, self.motion.num_frames - 1)
        cmd_body_idx = self._command_body_indices
        if isinstance(self.motion, G1CommandBatch):
            env_ids = torch.arange(ref_indices.shape[0], device=ref_indices.device)
            return {
                "joint_pos": self.motion.joint_pos[ref_indices, env_ids],
                "joint_vel": self.motion.joint_vel[ref_indices, env_ids],
                "body_pos_w": self.motion.body_pos_w[ref_indices, env_ids][:, cmd_body_idx],
                "body_quat_w": self.motion.body_quat_w[ref_indices, env_ids][:, cmd_body_idx],
                "body_ang_vel_w": self.motion.body_ang_vel_w[ref_indices, env_ids][
                    :, cmd_body_idx
                ],
            }
        return {
            "joint_pos": self.motion.joint_pos[ref_indices],
            "joint_vel": self.motion.joint_vel[ref_indices],
            "body_pos_w": self.motion.body_pos_w[ref_indices][:, cmd_body_idx],
            "body_quat_w": self.motion.body_quat_w[ref_indices][:, cmd_body_idx],
            "body_ang_vel_w": self.motion.body_ang_vel_w[ref_indices][:, cmd_body_idx],
        }

    def compute(
        self,
        robot: RobotState,
        ref_indices: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        ref = self._ref_fields(ref_indices.to(robot.qpos.device))
        command = torch.cat([ref["joint_pos"], ref["joint_vel"]], dim=-1)
        ref_limb = self._limb_pose_in_anchor_frame(
            ref["body_pos_w"], ref["body_quat_w"]
        )

        robot_body_pos = robot.body_pos_w[:, self._command_body_indices]
        robot_body_quat = robot.body_quat_w[:, self._command_body_indices]
        robot_limb = self._limb_pose_in_anchor_frame(robot_body_pos, robot_body_quat)

        root_quat = robot.qpos[:, 3:7]
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device).expand(
            self.num_envs, -1
        )
        projected_gravity = quat_apply_inverse(root_quat, gravity_w)
        base_ang_vel_b = quat_apply_inverse(root_quat, robot.body_ang_vel_w[:, 0])
        joint_pos_rel = robot.qpos[:, 7:] - self.default_joint_pos
        joint_vel_rel = robot.qvel[:, 6:]
        motion_ref_ang_vel = ref["body_ang_vel_w"][:, self._tracking_anchor_index]

        obs = torch.cat(
            [
                command,
                self.histories["ref_limb_ee_pose_b"].append(ref_limb),
                motion_ref_ang_vel,
                self.histories["robot_limb_ee_pose_b"].append(robot_limb),
                self.histories["projected_gravity"].append(projected_gravity),
                self.histories["base_ang_vel"].append(base_ang_vel_b),
                self.histories["joint_pos"].append(joint_pos_rel),
                self.histories["joint_vel"].append(joint_vel_rel),
                self.histories["actions"].append(last_action),
            ],
            dim=-1,
        )
        if obs.shape[-1] != OBS_DIM:
            raise RuntimeError(f"Expected obs dim {OBS_DIM}, got {obs.shape[-1]}")
        return obs
