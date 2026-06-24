"""Standardized full-state HOI motion evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import ACTION_DIM
from spider.tasks.g1_wbc.rollout import RolloutResult
from spider.tasks.g1_wbc_interaction.layout import InteractionModelLayout
from spider.tasks.g1_wbc_interaction.metrics import (
    compute_interaction_rollout_metrics,
    compute_retarget_object_metrics,
)
from spider.tasks.g1_wbc_interaction.motion import (
    InteractionMotion,
    load_interaction_motion,
    qvel_from_full_qpos,
    slice_interaction_motion,
)
from spider.tasks.g1_wbc_interaction.rollout import (
    G1WbcInteractionMujocoWarpEnv,
    InteractionRolloutConfig,
)


@dataclass(frozen=True)
class FullStateTrajectory:
    qpos: np.ndarray
    qvel: np.ndarray | None
    time: np.ndarray


@dataclass(frozen=True)
class StandardEvaluation:
    metrics: dict[str, Any]
    qpos: np.ndarray
    qvel: np.ndarray
    time: np.ndarray
    reference_qpos: np.ndarray
    metadata: dict[str, Any]


def load_reference_motion_for_standard_eval(
    motion_path: str | Path,
    *,
    model,
    layout: InteractionModelLayout,
    device: str,
    source_dt: float,
) -> tuple[InteractionMotion, np.ndarray]:
    """Load the reference on its source time grid, not the WBC policy grid."""

    time = load_motion_time_grid(motion_path, layout=layout, fallback_dt=source_dt)
    motion = load_interaction_motion(
        motion_path,
        model=model,
        layout=layout,
        device=device,
        target_dt=_uniform_dt(time, fallback_dt=source_dt),
        source_dt=_uniform_dt(time, fallback_dt=source_dt),
    )
    if motion.full_state_qpos().shape[0] != time.shape[0]:
        raise ValueError(
            "Reference motion and reference time grid disagree: "
            f"{motion.full_state_qpos().shape[0]} frames vs {time.shape[0]} times."
        )
    return motion, time


def load_motion_time_grid(
    path: str | Path,
    *,
    layout: InteractionModelLayout,
    fallback_dt: float,
) -> np.ndarray:
    motion_path = Path(path).expanduser().resolve()
    with np.load(motion_path) as data:
        if "qpos" not in data.files:
            raise ValueError(f"{motion_path} is missing qpos.")
        qpos = _flatten_qpos(np.asarray(data["qpos"]), layout.nq)
        if "time" in data.files:
            time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
            if time.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"{motion_path} has {qpos.shape[0]} qpos frames but "
                    f"{time.shape[0]} timestamps."
                )
            return _normalize_time(time)
        if "dt" in data.files:
            dt = float(np.asarray(data["dt"]).item())
        elif "fps" in data.files:
            dt = 1.0 / float(np.asarray(data["fps"]).item())
        else:
            dt = float(fallback_dt)
        return np.arange(qpos.shape[0], dtype=np.float64) * dt


def load_full_state_trajectory(
    path: str | Path,
    *,
    layout: InteractionModelLayout,
    fallback_dt: float,
) -> FullStateTrajectory:
    traj_path = Path(path).expanduser().resolve()
    with np.load(traj_path) as data:
        qpos = _first_array(
            data,
            traj_path,
            ("qpos", "refined_qpos", "command_qpos_trajectory"),
        )
        qvel = _optional_array(data, ("qvel", "refined_qvel"))
        qpos = _flatten_qpos(qpos, layout.nq)
        if qvel is not None:
            qvel = _flatten_qvel(qvel, layout.nv)
            if qvel.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"{traj_path} has {qpos.shape[0]} qpos frames but "
                    f"{qvel.shape[0]} qvel frames."
                )
        if "time" in data.files:
            time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
            if time.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"{traj_path} has {qpos.shape[0]} qpos frames but "
                    f"{time.shape[0]} timestamps."
                )
        else:
            time = np.arange(qpos.shape[0], dtype=np.float64) * float(fallback_dt)
    return FullStateTrajectory(qpos=qpos, qvel=qvel, time=_normalize_time(time))


def trajectory_from_rollout(rollout: RolloutResult) -> FullStateTrajectory:
    qpos = rollout.qpos[:, 0].detach().cpu().numpy()
    qvel = rollout.qvel[:, 0].detach().cpu().numpy()
    time = np.arange(qpos.shape[0], dtype=np.float64) * float(rollout.dt)
    return FullStateTrajectory(qpos=qpos, qvel=qvel, time=time)


def restrict_reference_to_covered_time_grid(
    reference_motion: InteractionMotion,
    reference_time: np.ndarray,
    trajectories: list[FullStateTrajectory],
) -> tuple[InteractionMotion, np.ndarray]:
    """Keep one source-grid interval covered by every evaluated trajectory."""

    if not trajectories:
        raise ValueError("At least one trajectory is required for standard evaluation.")
    reference_time = np.asarray(reference_time, dtype=np.float64).reshape(-1)
    if reference_motion.full_state_qpos().shape[0] != reference_time.shape[0]:
        raise ValueError(
            "Reference motion/time mismatch before grid restriction: "
            f"{reference_motion.full_state_qpos().shape[0]} frames vs "
            f"{reference_time.shape[0]} times."
        )
    coverage_start = max(float(traj.time[0]) for traj in trajectories)
    coverage_end = min(float(traj.time[-1]) for traj in trajectories)
    tol = _time_tolerance(reference_time)
    mask = (reference_time >= coverage_start - tol) & (
        reference_time <= coverage_end + tol
    )
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError(
            "No reference time-grid frames are covered by all trajectories: "
            f"covered interval=[{coverage_start}, {coverage_end}], "
            f"reference interval=[{reference_time[0]}, {reference_time[-1]}]."
        )
    start = int(indices[0])
    length = int(indices[-1] - indices[0] + 1)
    if length != int(indices.size):
        raise ValueError("Reference time-grid coverage must be one contiguous interval.")
    return slice_interaction_motion(reference_motion, start, length), reference_time[indices]


def evaluate_trajectory_on_reference_grid(
    trajectory: FullStateTrajectory,
    *,
    reference_motion: InteractionMotion,
    reference_time: np.ndarray,
    layout: InteractionModelLayout,
    rollout_config: InteractionRolloutConfig,
    pos_threshold: float,
    quat_threshold: float,
) -> StandardEvaluation:
    """Evaluate a trajectory only on the original reference time grid."""

    reference_qpos = reference_motion.full_state_qpos().detach().cpu().numpy()
    reference_time = np.asarray(reference_time, dtype=np.float64).reshape(-1)
    if reference_qpos.shape[0] != reference_time.shape[0]:
        raise ValueError(
            "Reference qpos/time mismatch: "
            f"{reference_qpos.shape[0]} frames vs {reference_time.shape[0]} times."
        )

    qpos = _qpos_at_times_strict(trajectory.qpos, trajectory.time, layout, reference_time)
    if trajectory.qvel is None:
        qvel = _qvel_from_qpos_np(qpos, layout, reference_time, rollout_config.device)
    else:
        qvel = _array_at_times_strict(trajectory.qvel, trajectory.time, reference_time)
    rollout_dt = _uniform_dt(reference_time, fallback_dt=1.0)
    rollout = static_rollout_on_grid(qpos, qvel, rollout_config, dt=rollout_dt)
    metrics = compute_interaction_rollout_metrics(
        reference_motion,
        rollout,
        layout=layout,
    )
    metrics.update(
        compute_retarget_object_metrics(
            qpos,
            reference_qpos,
            layout,
            pos_threshold=pos_threshold,
            quat_threshold=quat_threshold,
        )
    )
    metadata = {
        "time_grid": "reference_motion",
        "frames": int(reference_time.shape[0]),
        "start": float(reference_time[0]) if reference_time.size else None,
        "end": float(reference_time[-1]) if reference_time.size else None,
        "dt_mean": _dt_stat(reference_time, "mean"),
        "dt_min": _dt_stat(reference_time, "min"),
        "dt_max": _dt_stat(reference_time, "max"),
    }
    return StandardEvaluation(
        metrics=metrics,
        qpos=qpos,
        qvel=qvel,
        time=reference_time,
        reference_qpos=reference_qpos,
        metadata=metadata,
    )


def static_rollout_on_grid(
    qpos: np.ndarray,
    qvel: np.ndarray,
    config: InteractionRolloutConfig,
    *,
    dt: float,
) -> RolloutResult:
    device = torch.device(config.device)
    qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    qvel_t = torch.as_tensor(qvel, dtype=torch.float32, device=device)
    if qpos_t.ndim == 2:
        qpos_t = qpos_t[:, None, :]
    if qvel_t.ndim == 2:
        qvel_t = qvel_t[:, None, :]
    total_steps = qpos_t.shape[0] - 1
    env = G1WbcInteractionMujocoWarpEnv(
        replace(
            config,
            num_envs=int(qpos_t.shape[1]),
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
            env.reset(qpos_t[frame_idx], qvel_t[frame_idx])
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
        qpos_t.shape[1],
        ACTION_DIM,
        dtype=torch.float32,
        device=device,
    )
    controls = env.layout.robot_qpos(qpos_t[:-1])[:, :, 7:].contiguous()
    ref_indices = torch.arange(total_steps + 1, dtype=torch.long, device=device)
    ref_indices = ref_indices[:, None].expand(total_steps + 1, qpos_t.shape[1])
    return RolloutResult(
        qpos=qpos_t,
        qvel=qvel_t,
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
        dt=float(dt),
    )


def _first_array(data, path: Path, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    raise ValueError(f"{path} is missing {'/'.join(keys)}.")


def _optional_array(data, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    return None


def _flatten_qpos(value: np.ndarray, nq: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 3:
        value = value.reshape(-1, value.shape[-1])
    if value.ndim != 2 or value.shape[-1] != nq:
        raise ValueError(f"Expected qpos shape (T,{nq}) or (...,{nq}), got {value.shape}.")
    return np.ascontiguousarray(value)


def _flatten_qvel(value: np.ndarray, nv: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 3:
        value = value.reshape(-1, value.shape[-1])
    if value.ndim != 2 or value.shape[-1] != nv:
        raise ValueError(f"Expected qvel shape (T,{nv}) or (...,{nv}), got {value.shape}.")
    return np.ascontiguousarray(value)


def _normalize_time(time: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if time.ndim != 1 or time.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 1D time grid, got {time.shape}.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Time grid must be strictly increasing.")
    return np.ascontiguousarray(time)


def _uniform_dt(time: np.ndarray, *, fallback_dt: float) -> float:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if time.shape[0] <= 1:
        return float(fallback_dt)
    dt = np.diff(time)
    mean_dt = float(dt.mean())
    if not np.allclose(dt, mean_dt, rtol=1.0e-4, atol=1.0e-7):
        raise ValueError("Standard HOI evaluation requires a uniform reference time grid.")
    return mean_dt


def _dt_stat(time: np.ndarray, stat: str) -> float | None:
    if time.shape[0] <= 1:
        return None
    dt = np.diff(time)
    if stat == "mean":
        return float(dt.mean())
    if stat == "min":
        return float(dt.min())
    if stat == "max":
        return float(dt.max())
    raise ValueError(stat)


def _qpos_at_times_strict(
    qpos: np.ndarray,
    qpos_times: np.ndarray,
    layout: InteractionModelLayout,
    times: np.ndarray,
) -> np.ndarray:
    _assert_time_coverage(qpos_times, times)
    qpos = _flatten_qpos(qpos, layout.nq)
    qpos_times = np.asarray(qpos_times, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if qpos_times.shape[0] != qpos.shape[0]:
        raise ValueError(
            f"Expected {qpos.shape[0]} qpos timestamps, got {qpos_times.shape[0]}."
        )
    if qpos.shape[0] == 1:
        return np.repeat(qpos[:1], times.shape[0], axis=0)
    i1 = np.searchsorted(qpos_times, times, side="left")
    i1 = np.clip(i1, 1, qpos.shape[0] - 1)
    i0 = i1 - 1
    denom = np.maximum(qpos_times[i1] - qpos_times[i0], 1.0e-12)
    alpha = ((times - qpos_times[i0]) / denom)[:, None]
    out = qpos[i0] * (1.0 - alpha) + qpos[i1] * alpha
    for qpos_adr in _freejoint_qpos_addrs(layout):
        out[:, qpos_adr + 3 : qpos_adr + 7] = _slerp_np(
            qpos[i0, qpos_adr + 3 : qpos_adr + 7],
            qpos[i1, qpos_adr + 3 : qpos_adr + 7],
            alpha,
        )
    return np.ascontiguousarray(out)


def _array_at_times_strict(
    values: np.ndarray,
    value_times: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    _assert_time_coverage(value_times, times)
    values = np.asarray(values, dtype=np.float64)
    value_times = np.asarray(value_times, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if values.shape[0] != value_times.shape[0]:
        raise ValueError(
            f"Expected {values.shape[0]} value timestamps, got {value_times.shape[0]}."
        )
    if values.shape[0] == 1:
        return np.repeat(values[:1], times.shape[0], axis=0)
    i1 = np.searchsorted(value_times, times, side="left")
    i1 = np.clip(i1, 1, values.shape[0] - 1)
    i0 = i1 - 1
    denom = np.maximum(value_times[i1] - value_times[i0], 1.0e-12)
    alpha = ((times - value_times[i0]) / denom).reshape((-1,) + (1,) * (values.ndim - 1))
    return np.ascontiguousarray(values[i0] * (1.0 - alpha) + values[i1] * alpha)


def _assert_time_coverage(source_time: np.ndarray, target_time: np.ndarray) -> None:
    source_time = np.asarray(source_time, dtype=np.float64).reshape(-1)
    target_time = np.asarray(target_time, dtype=np.float64).reshape(-1)
    if source_time.shape[0] == 0 or target_time.shape[0] == 0:
        raise ValueError("Cannot evaluate empty time grids.")
    tol = _time_tolerance(target_time)
    if source_time[0] > target_time[0] + tol or source_time[-1] < target_time[-1] - tol:
        raise ValueError(
            "Trajectory does not cover the reference time grid: "
            f"trajectory=[{source_time[0]}, {source_time[-1]}], "
            f"reference=[{target_time[0]}, {target_time[-1]}]."
        )


def _time_tolerance(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    span = float(time[-1] - time[0]) if time.shape[0] > 1 else 1.0
    return max(1.0e-7, 1.0e-5 * max(1.0, span))


def _qvel_from_qpos_np(
    qpos: np.ndarray,
    layout: InteractionModelLayout,
    time: np.ndarray,
    device: str,
) -> np.ndarray:
    dt = _uniform_dt(time, fallback_dt=1.0)
    qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    return qvel_from_full_qpos(qpos_t, layout, dt=dt).detach().cpu().numpy()


def _freejoint_qpos_addrs(layout: InteractionModelLayout) -> list[int]:
    addrs = [int(layout.root_qpos_adr)]
    addrs.extend(
        int(obj.pose_qpos_adr)
        for obj in layout.objects
        if obj.has_freejoint_pose and obj.pose_qpos_adr is not None
    )
    return addrs


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
