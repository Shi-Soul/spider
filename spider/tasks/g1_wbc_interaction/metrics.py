"""Metrics and MPC scores for G1 WBC interaction rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from spider.math import quat_sub
from spider.tasks.g1_wbc.math_utils import quat_error_magnitude
from spider.tasks.g1_wbc.metrics import compute_rollout_metrics, compute_rollout_scores
from spider.tasks.g1_wbc.rollout import RolloutResult
from spider.tasks.g1_wbc_interaction.layout import InteractionModelLayout
from spider.tasks.g1_wbc_interaction.motion import InteractionMotion


@dataclass(frozen=True)
class InteractionScoreWeights:
    """Default MPC loss weights for object interaction tracking."""

    robot_contact_mismatch: float = 4.0
    robot_contact_switch: float = 2.0
    robot_contact_false_positive: float = 0.0
    robot_contact_false_negative: float = 0.0
    robot_contact_force_excess: float = 0.0
    robot_contact_force_delta: float = 0.0
    robot_bad_floor_contact: float = 0.0
    robot_bad_floor_force_excess: float = 0.0
    robot_ee_global_pos: float = 2.5
    robot_ee_global_rot: float = 0.0
    robot_ee_local_pos: float = 1.5
    robot_ee_local_rot: float = 0.0
    robot_hand_global_pos: float = 0.0
    robot_hand_global_rot: float = 0.0
    robot_hand_local_pos: float = 0.0
    robot_hand_local_rot: float = 0.0
    robot_body_global_pos: float = 0.0
    robot_body_global_rot: float = 0.0
    robot_body_local_pos: float = 0.0
    robot_body_local_rot: float = 0.0
    robot_root_pos: float = 1.0
    robot_root_rot: float = 0.4
    robot_joint_pos: float = 0.15
    robot_action_delta: float = 0.0
    robot_joint_acc: float = 0.0
    robot_joint_jerk: float = 0.0
    control_delta: float = 0.04
    object_pos: float = 35.0
    object_rot: float = 4.0
    object_final_pos: float = 80.0
    object_final_rot: float = 8.0
    object_vel: float = 0.5


@dataclass(frozen=True)
class RetargetScoreWeights:
    """MJWP-style full-state retarget tracking weights."""

    base_pos: float = 1.0
    base_rot: float = 1.0
    joint: float = 1.0
    object_pos: float = 1.0
    object_rot: float = 0.3
    vel: float = 0.0


def compute_interaction_rollout_metrics(
    motion: InteractionMotion,
    rollout: RolloutResult,
    *,
    layout: InteractionModelLayout,
) -> dict[str, float | bool]:
    robot_rollout = _robot_rollout_view(
        rollout,
        layout,
        max_ref_index=motion.num_frames - 1,
    )
    robot_metrics = compute_rollout_metrics(motion, robot_rollout)
    object_metrics = compute_object_metrics(
        rollout.qpos,
        motion.full_state_qpos(),
        rollout.qvel,
        motion.full_state_qvel(),
        rollout.ref_indices,
        layout,
    )
    metrics = {f"robot_{key}": value for key, value in robot_metrics.items()}
    metrics.update(object_metrics)
    success = (
        bool(robot_metrics.get("success", False))
        and float(object_metrics.get("obj_pos_err", 0.0)) <= 0.1
        and float(object_metrics.get("obj_quat_err", 0.0)) <= 0.5
    )
    metrics["success"] = success
    return metrics


def compute_retarget_rollout_scores(
    motion: InteractionMotion,
    rollout: RolloutResult,
    *,
    layout: InteractionModelLayout,
    weights: RetargetScoreWeights = RetargetScoreWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score rollouts with the MJWP humanoid_object qpos/qvel tracking reward."""

    device = rollout.qpos.device
    ref_indices = rollout.ref_indices.to(device).clamp(0, motion.num_frames - 1)
    ref_indices = _post_step_ref_indices(
        ref_indices,
        rollout.qpos.shape[0],
        motion.num_frames,
    )
    ref_qpos = motion.full_state_qpos().to(device)[ref_indices]
    ref_qvel = motion.full_state_qvel().to(device)[ref_indices]
    columns: list[torch.Tensor] = []
    weights_cols: list[torch.Tensor] = []
    root_pos_slice = slice(layout.root_qpos_adr, layout.root_qpos_adr + 3)
    root_quat_slice = slice(layout.root_qpos_adr + 3, layout.root_qpos_adr + 7)
    columns.append(rollout.qpos[..., root_pos_slice] - ref_qpos[..., root_pos_slice])
    weights_cols.append(
        torch.full(
            columns[-1].shape,
            float(weights.base_pos),
            dtype=rollout.qpos.dtype,
            device=device,
        )
    )
    columns.append(
        quat_error_vec(
            rollout.qpos[..., root_quat_slice],
            ref_qpos[..., root_quat_slice],
        )
    )
    weights_cols.append(
        torch.full(
            columns[-1].shape,
            float(weights.base_rot),
            dtype=rollout.qpos.dtype,
            device=device,
        )
    )
    robot_joint_idx = torch.as_tensor(
        layout.robot_joint_qpos_indices,
        device=device,
        dtype=torch.long,
    )
    columns.append(
        rollout.qpos.index_select(-1, robot_joint_idx)
        - ref_qpos.index_select(-1, robot_joint_idx)
    )
    weights_cols.append(
        torch.full(
            columns[-1].shape,
            float(weights.joint),
            dtype=rollout.qpos.dtype,
            device=device,
        )
    )
    for obj in layout.objects:
        if obj.has_freejoint_pose:
            columns.append(
                rollout.qpos[..., obj.qpos_slice][..., :3]
                - ref_qpos[..., obj.qpos_slice][..., :3]
            )
            weights_cols.append(
                torch.full(
                    columns[-1].shape,
                    float(weights.object_pos),
                    dtype=rollout.qpos.dtype,
                    device=device,
                )
            )
            columns.append(
                quat_error_vec(
                    rollout.qpos[..., obj.qpos_slice][..., 3:7],
                    ref_qpos[..., obj.qpos_slice][..., 3:7],
                )
            )
            weights_cols.append(
                torch.full(
                    columns[-1].shape,
                    float(weights.object_rot),
                    dtype=rollout.qpos.dtype,
                    device=device,
                )
            )
        elif obj.qpos_indices:
            obj_idx = torch.as_tensor(obj.qpos_indices, device=device, dtype=torch.long)
            columns.append(
                rollout.qpos.index_select(-1, obj_idx)
                - ref_qpos.index_select(-1, obj_idx)
            )
            weights_cols.append(
                torch.full(
                    columns[-1].shape,
                    float(weights.object_pos),
                    dtype=rollout.qpos.dtype,
                    device=device,
                )
            )
    qpos_diff = torch.cat(columns, dim=-1)
    qpos_weight = torch.cat(weights_cols, dim=-1)
    qvel_indices = list(layout.robot_qvel_indices)
    for obj in layout.objects:
        qvel_indices.extend(obj.qvel_indices)
    qvel_idx = torch.as_tensor(
        qvel_indices,
        device=device,
        dtype=torch.long,
    )
    sim_qvel = rollout.qvel.index_select(-1, qvel_idx)
    target_qvel = ref_qvel.index_select(-1, qvel_idx)
    qpos_dist = torch.linalg.norm(qpos_diff * qpos_weight, dim=-1)
    qvel_dist = torch.linalg.norm(sim_qvel - target_qvel, dim=-1)
    qpos_dist_score = _per_env_mean(qpos_dist)
    qvel_dist_score = _per_env_mean(qvel_dist)
    reward = -(qpos_dist_score + float(weights.vel) * qvel_dist_score)
    terms = {
        "retarget_qpos_dist": qpos_dist_score,
        "retarget_qvel_dist": qvel_dist_score,
        "retarget_reward": reward,
    }
    return reward, terms


def compute_interaction_rollout_scores(
    motion: InteractionMotion,
    rollout: RolloutResult,
    *,
    layout: InteractionModelLayout,
    weights: InteractionScoreWeights = InteractionScoreWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    robot_rollout = _robot_rollout_view(
        rollout,
        layout,
        max_ref_index=motion.num_frames - 1,
    )
    _, robot_terms = compute_rollout_scores(motion, robot_rollout)
    object_terms = compute_object_score_terms(
        rollout.qpos,
        motion.full_state_qpos(),
        rollout.qvel,
        motion.full_state_qvel(),
        rollout.ref_indices,
        layout,
    )
    loss = (
        weights.robot_contact_mismatch * robot_terms["contact_mismatch"]
        + weights.robot_contact_switch * robot_terms["contact_switch"]
        + weights.robot_contact_false_positive * robot_terms["contact_false_positive"]
        + weights.robot_contact_false_negative * robot_terms["contact_false_negative"]
        + weights.robot_contact_force_excess * robot_terms["contact_force_excess"]
        + weights.robot_contact_force_delta * robot_terms["contact_force_delta"]
        + weights.robot_bad_floor_contact * robot_terms["bad_floor_contact"]
        + weights.robot_bad_floor_force_excess * robot_terms["bad_floor_force_excess"]
        + weights.robot_ee_global_pos * robot_terms["ee_global_pos_error"]
        + weights.robot_ee_global_rot * robot_terms["ee_global_rot_error"]
        + weights.robot_ee_local_pos * robot_terms["ee_local_pos_error"]
        + weights.robot_ee_local_rot * robot_terms["ee_local_rot_error"]
        + weights.robot_hand_global_pos * robot_terms["hand_global_pos_error"]
        + weights.robot_hand_global_rot * robot_terms["hand_global_rot_error"]
        + weights.robot_hand_local_pos * robot_terms["hand_local_pos_error"]
        + weights.robot_hand_local_rot * robot_terms["hand_local_rot_error"]
        + weights.robot_body_global_pos * robot_terms["body_global_pos_error"]
        + weights.robot_body_global_rot * robot_terms["body_global_rot_error"]
        + weights.robot_body_local_pos * robot_terms["body_local_pos_error"]
        + weights.robot_body_local_rot * robot_terms["body_local_rot_error"]
        + weights.robot_root_pos * robot_terms["root_pos_error"]
        + weights.robot_root_rot * robot_terms["root_rot_error"]
        + weights.robot_joint_pos * robot_terms["joint_pos_error"]
        + weights.robot_action_delta * robot_terms["action_delta"]
        + weights.robot_joint_acc * robot_terms["joint_acc"]
        + weights.robot_joint_jerk * robot_terms["joint_jerk"]
        + weights.control_delta * robot_terms["control_delta"]
        + weights.object_pos * object_terms["object_pos_error"]
        + weights.object_rot * object_terms["object_rot_error"]
        + weights.object_final_pos * object_terms["object_final_pos_error"]
        + weights.object_final_rot * object_terms["object_final_rot_error"]
        + weights.object_vel * object_terms["object_vel_error"]
    )
    terms = {f"robot_{key}": value for key, value in robot_terms.items()}
    terms.update(object_terms)
    return -loss, terms


def quat_error_vec(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape[:-1]
    return quat_sub(q1.reshape(-1, 4), q2.reshape(-1, 4)).reshape(shape + (3,))


def compute_object_score_terms(
    qpos: torch.Tensor,
    ref_qpos_all: torch.Tensor,
    qvel: torch.Tensor | None,
    ref_qvel_all: torch.Tensor | None,
    ref_indices: torch.Tensor,
    layout: InteractionModelLayout,
) -> dict[str, torch.Tensor]:
    device = qpos.device
    ref_indices = ref_indices.to(device).clamp(0, ref_qpos_all.shape[0] - 1)
    if not layout.objects:
        batch = qpos.shape[1] if qpos.ndim == 3 else 1
        zero = torch.zeros(batch, dtype=qpos.dtype, device=device)
        return {
            "object_pos_error": zero,
            "object_rot_error": zero,
            "object_final_pos_error": zero,
            "object_final_rot_error": zero,
            "object_vel_error": zero,
            "object_qpos_error": zero,
        }
    if qpos.ndim != 3:
        raise ValueError(f"Expected rollout qpos shape (T, N, nq), got {qpos.shape}.")
    ref_indices = _post_step_ref_indices(ref_indices, qpos.shape[0], ref_qpos_all.shape[0])
    ref = ref_qpos_all.to(device)[ref_indices]
    pos_terms = []
    rot_terms = []
    final_pos_terms = []
    final_rot_terms = []
    vel_terms = []
    qpos_terms = []
    for obj in layout.objects:
        if obj.qpos_indices:
            qpos_idx = torch.as_tensor(
                obj.qpos_indices, device=device, dtype=torch.long
            )
            sim_qpos = qpos.index_select(-1, qpos_idx)
            ref_qpos = ref.index_select(-1, qpos_idx)
            qpos_terms.append(_per_env_mean(torch.linalg.norm(sim_qpos - ref_qpos, dim=-1)))
        if obj.has_freejoint_pose:
            sim_obj = qpos[..., obj.qpos_slice]
            ref_obj = ref[..., obj.qpos_slice]
            pos_err = torch.linalg.norm(sim_obj[..., :3] - ref_obj[..., :3], dim=-1)
            rot_err = quat_error_magnitude(sim_obj[..., 3:7], ref_obj[..., 3:7])
        elif obj.qpos_indices:
            pos_err = torch.linalg.norm(sim_qpos - ref_qpos, dim=-1)
            rot_err = torch.zeros_like(pos_err)
        else:
            pos_err = torch.zeros(qpos.shape[:2], dtype=qpos.dtype, device=device)
            rot_err = torch.zeros_like(pos_err)
        pos_terms.append(_per_env_mean(pos_err))
        rot_terms.append(_per_env_mean(rot_err))
        final_pos_terms.append(pos_err[-1])
        final_rot_terms.append(rot_err[-1])
        if qvel is not None and ref_qvel_all is not None and obj.qvel_indices:
            qvel_idx = torch.as_tensor(
                obj.qvel_indices, device=device, dtype=torch.long
            )
            ref_vel = ref_qvel_all.to(device)[ref_indices].index_select(-1, qvel_idx)
            sim_vel = qvel.index_select(-1, qvel_idx)
            vel_terms.append(_per_env_mean(torch.linalg.norm(sim_vel - ref_vel, dim=-1)))
    object_pos = torch.stack(pos_terms, dim=0).mean(dim=0)
    object_rot = torch.stack(rot_terms, dim=0).mean(dim=0)
    final_pos = torch.stack(final_pos_terms, dim=0).mean(dim=0)
    final_rot = torch.stack(final_rot_terms, dim=0).mean(dim=0)
    if vel_terms:
        object_vel = torch.stack(vel_terms, dim=0).mean(dim=0)
    else:
        object_vel = torch.zeros_like(object_pos)
    if qpos_terms:
        object_qpos = torch.stack(qpos_terms, dim=0).mean(dim=0)
    else:
        object_qpos = torch.zeros_like(object_pos)
    return {
        "object_pos_error": object_pos,
        "object_rot_error": object_rot,
        "object_final_pos_error": final_pos,
        "object_final_rot_error": final_rot,
        "object_vel_error": object_vel,
        "object_qpos_error": object_qpos,
    }


def _post_step_ref_indices(
    ref_indices: torch.Tensor,
    rollout_frames: int,
    ref_frames: int,
) -> torch.Tensor:
    out = ref_indices.clone()
    if int(rollout_frames) > 1 and out.shape[0] == int(rollout_frames):
        out[1:] = out[1:] + 1
    return out.clamp(0, int(ref_frames) - 1)


def compute_object_metrics(
    qpos: torch.Tensor,
    ref_qpos_all: torch.Tensor,
    qvel: torch.Tensor | None,
    ref_qvel_all: torch.Tensor | None,
    ref_indices: torch.Tensor,
    layout: InteractionModelLayout,
) -> dict[str, float]:
    terms = compute_object_score_terms(qpos, ref_qpos_all, qvel, ref_qvel_all, ref_indices, layout)
    out = {
        "obj_pos_err": _scalar_mean(terms["object_pos_error"]),
        "obj_quat_err": _scalar_mean(terms["object_rot_error"]),
        "object_final_pos_err": _scalar_mean(terms["object_final_pos_error"]),
        "object_final_quat_err": _scalar_mean(terms["object_final_rot_error"]),
        "object_vel_err": _scalar_mean(terms["object_vel_error"]),
        "object_qpos_err": _scalar_mean(terms["object_qpos_error"]),
        "num_objects": float(len(layout.objects)),
    }
    out["add_auc10_mean"] = _add_auc10(qpos, ref_qpos_all, ref_indices, layout)
    out["object_success"] = out["obj_pos_err"] <= 0.1 and out["obj_quat_err"] <= 0.5
    return out


def compute_retarget_object_metrics(
    qpos: np.ndarray,
    ref_qpos: np.ndarray,
    layout: InteractionModelLayout,
    *,
    pos_threshold: float = 0.1,
    quat_threshold: float = 0.5,
) -> dict[str, object]:
    """Object tracking metrics using the retarget benchmark field names.

    The computation is layout-driven: every non-robot freejoint discovered in
    the model contributes one object track, and aggregate metrics average over
    all object-frame errors.
    """

    qpos = _as_flat_qpos(qpos, layout.nq)
    ref_qpos = _as_flat_qpos(ref_qpos, layout.nq)
    n = min(qpos.shape[0], ref_qpos.shape[0])
    qpos = qpos[:n]
    ref_qpos = ref_qpos[:n]
    if n == 0:
        raise ValueError("Need at least one frame to compute object metrics.")

    if not layout.objects:
        return {
            "frames": int(n),
            "num_objects": 0,
            "obj_pos_err": float("nan"),
            "obj_quat_err": float("nan"),
            "obj_pos_err_mean": float("nan"),
            "obj_pos_err_max": float("nan"),
            "obj_quat_err_mean": float("nan"),
            "obj_quat_err_max": float("nan"),
            "add_auc10": float("nan"),
            "add_auc10_mean": float("nan"),
            "success_10cm_0p5rad": False,
            "per_object": {},
        }

    per_object: dict[str, dict[str, float]] = {}
    pos_columns: list[np.ndarray] = []
    quat_columns: list[np.ndarray] = []
    auc_values: list[float] = []
    for obj in layout.objects:
        if not obj.has_freejoint_pose:
            continue
        sim_obj = qpos[:, obj.qpos_slice]
        ref_obj = ref_qpos[:, obj.qpos_slice]
        pos_err = np.linalg.norm(sim_obj[:, :3] - ref_obj[:, :3], axis=-1)
        quat_err = _quat_error_np(sim_obj[:, 3:7], ref_obj[:, 3:7])
        auc10 = _add_auc10_np(sim_obj[:, :3], ref_obj[:, :3])
        pos_columns.append(pos_err)
        quat_columns.append(quat_err)
        auc_values.append(float(auc10))
        per_object[obj.name] = {
            "obj_pos_err_mean": float(pos_err.mean()),
            "obj_pos_err_max": float(pos_err.max()),
            "obj_quat_err_mean": float(quat_err.mean()),
            "obj_quat_err_max": float(quat_err.max()),
            "add_auc10": float(auc10),
        }

    if not pos_columns:
        return {
            "frames": int(n),
            "num_objects": int(len(layout.objects)),
            "num_pose_objects": 0,
            "obj_pos_err": float("nan"),
            "obj_quat_err": float("nan"),
            "obj_pos_err_mean": float("nan"),
            "obj_pos_err_max": float("nan"),
            "obj_quat_err_mean": float("nan"),
            "obj_quat_err_max": float("nan"),
            "add_auc10": float("nan"),
            "add_auc10_mean": float("nan"),
            "success_10cm_0p5rad": False,
            "per_object": per_object,
        }

    pos_all = np.stack(pos_columns, axis=1)
    quat_all = np.stack(quat_columns, axis=1)
    obj_pos_mean = float(pos_all.mean())
    obj_quat_mean = float(quat_all.mean())
    add_auc10 = float(np.mean(auc_values))
    success = obj_pos_mean <= float(pos_threshold) and obj_quat_mean <= float(
        quat_threshold
    )
    return {
        "frames": int(n),
        "num_objects": int(len(layout.objects)),
        "num_pose_objects": int(len(pos_columns)),
        "obj_pos_err": obj_pos_mean,
        "obj_quat_err": obj_quat_mean,
        "obj_pos_err_mean": obj_pos_mean,
        "obj_pos_err_max": float(pos_all.max()),
        "obj_quat_err_mean": obj_quat_mean,
        "obj_quat_err_max": float(quat_all.max()),
        "add_auc10": add_auc10,
        "add_auc10_mean": add_auc10,
        "success_10cm_0p5rad": bool(success),
        "per_object": per_object,
    }


def _robot_rollout_view(
    rollout: RolloutResult,
    layout: InteractionModelLayout,
    *,
    max_ref_index: int | None = None,
) -> RolloutResult:
    ref_indices = rollout.ref_indices
    if max_ref_index is not None:
        ref_indices = ref_indices.clamp(0, int(max_ref_index))
    return RolloutResult(
        qpos=layout.robot_qpos(rollout.qpos),
        qvel=layout.robot_qvel(rollout.qvel),
        body_pos_w=rollout.body_pos_w,
        body_quat_w=rollout.body_quat_w,
        body_lin_vel_w=rollout.body_lin_vel_w,
        body_ang_vel_w=rollout.body_ang_vel_w,
        actions=rollout.actions,
        controls=rollout.controls,
        contact_indicator=rollout.contact_indicator,
        contact_force=rollout.contact_force,
        ref_indices=ref_indices,
        floor_contact_indicator=rollout.floor_contact_indicator,
        floor_contact_force=rollout.floor_contact_force,
        dt=rollout.dt,
        final_last_action=rollout.final_last_action,
        final_history_state=rollout.final_history_state,
    )


def _per_env_mean(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.mean().view(1)
    return value.reshape(value.shape[0], value.shape[1], -1).mean(dim=(0, 2))


def _scalar_mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def _add_auc10(
    qpos: torch.Tensor,
    ref_qpos_all: torch.Tensor,
    ref_indices: torch.Tensor,
    layout: InteractionModelLayout,
) -> float:
    if not layout.objects:
        return float("nan")
    qpos0 = qpos[:, 0].detach().cpu()
    ref = ref_qpos_all[ref_indices[:, 0].detach().cpu()].detach().cpu()
    aucs = []
    for obj in layout.objects:
        if not obj.has_freejoint_pose:
            continue
        err = torch.linalg.norm(
            qpos0[:, obj.qpos_slice][:, :3] - ref[:, obj.qpos_slice][:, :3],
            dim=-1,
        ).numpy()
        thresholds = np.linspace(0.0, 0.10, 100, dtype=np.float64)
        scores = (err[:, None] <= thresholds[None, :]).astype(np.float64).mean(axis=0)
        trapz = getattr(np, "trapezoid", np.trapz)
        aucs.append(float(trapz(scores, thresholds / 0.10)))
    if not aucs:
        return float("nan")
    return float(sum(aucs) / len(aucs))


def _as_flat_qpos(qpos: np.ndarray, nq: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.ndim != 2 or qpos.shape[-1] != nq:
        raise ValueError(f"Expected qpos shape (T, {nq}) or (..., {nq}), got {qpos.shape}.")
    return qpos


def _quat_error_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1_t = torch.as_tensor(q1, dtype=torch.float64)
    q2_t = torch.as_tensor(q2, dtype=torch.float64)
    return quat_error_magnitude(q1_t, q2_t).detach().cpu().numpy()


def _add_auc10_np(achieved: np.ndarray, target: np.ndarray) -> float:
    err = np.linalg.norm(np.asarray(achieved) - np.asarray(target), axis=-1)
    thresholds = np.linspace(0.0, 0.10, 100, dtype=np.float64)
    success = (err[:, None] <= thresholds[None, :]).astype(np.float64).mean(axis=0)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(success, thresholds / 0.10))


__all__ = [
    "InteractionScoreWeights",
    "RetargetScoreWeights",
    "compute_interaction_rollout_metrics",
    "compute_interaction_rollout_scores",
    "compute_object_metrics",
    "compute_object_score_terms",
    "compute_retarget_rollout_scores",
    "compute_retarget_object_metrics",
]
