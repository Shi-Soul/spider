"""Full-state motion loading for G1 WBC interaction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from spider.tasks.g1_wbc.constants import MUJOCO_BODY_NAMES, POLICY_DT
from spider.tasks.g1_wbc.math_utils import (
    axis_angle_from_quat,
    quat_inv,
    quat_mul,
    world_velocity_to_qvel,
)
from spider.tasks.g1_wbc.motion import G1Motion, estimate_foot_contacts
from spider.tasks.g1_wbc_interaction.layout import (
    InteractionModelLayout,
    validate_full_state_shapes,
)


@dataclass
class InteractionMotion(G1Motion):
    """A WBC command motion plus full robot/object reference state."""

    full_qpos: torch.Tensor | None = None
    full_qvel: torch.Tensor | None = None
    layout: InteractionModelLayout | None = None

    def qpos(self) -> torch.Tensor:
        if self.full_qpos is not None and self.layout is not None:
            return self.layout.robot_qpos(self.full_qpos)
        return super().qpos()

    def qvel(self) -> torch.Tensor:
        if self.full_qvel is not None and self.layout is not None:
            return self.layout.robot_qvel(self.full_qvel)
        return super().qvel()

    def full_state_qpos(self) -> torch.Tensor:
        if self.full_qpos is None:
            return super().qpos()
        return self.full_qpos

    def full_state_qvel(self) -> torch.Tensor:
        if self.full_qvel is None:
            return super().qvel()
        return self.full_qvel

    def to(self, device: str | torch.device) -> "InteractionMotion":
        return InteractionMotion(
            path=self.path,
            motion_type=self.motion_type,
            fps=self.fps,
            joint_pos=self.joint_pos.to(device),
            joint_vel=self.joint_vel.to(device),
            body_pos_w=self.body_pos_w.to(device),
            body_quat_w=self.body_quat_w.to(device),
            body_lin_vel_w=self.body_lin_vel_w.to(device),
            body_ang_vel_w=self.body_ang_vel_w.to(device),
            contact=self.contact.to(device),
            full_qpos=None if self.full_qpos is None else self.full_qpos.to(device),
            full_qvel=None if self.full_qvel is None else self.full_qvel.to(device),
            layout=self.layout,
        )


def load_interaction_motion(
    motion_path: str | Path,
    *,
    model: mujoco.MjModel,
    layout: InteractionModelLayout,
    device: str | torch.device = "cpu",
    target_dt: float = POLICY_DT,
    source_dt: float | None = None,
) -> InteractionMotion:
    """Load a retarget-style full qpos/qvel reference for one complete model."""

    path = Path(motion_path).expanduser().resolve()
    raw = np.load(path)
    if "qpos" not in raw.files:
        raise ValueError(f"Motion file {path} is missing qpos.")
    qpos_np = _flatten_traj_array(np.asarray(raw["qpos"], dtype=np.float32), layout.nq)
    qvel_np = (
        _flatten_traj_array(np.asarray(raw["qvel"], dtype=np.float32), layout.nv)
        if "qvel" in raw.files
        else None
    )
    validate_full_state_shapes(layout, qpos_np, qvel_np)
    if qvel_np is None:
        qvel_np = np.zeros((qpos_np.shape[0], layout.nv), dtype=np.float32)

    fps = _resolve_fps(raw, source_dt, target_dt)
    full_qpos = torch.as_tensor(qpos_np, dtype=torch.float32, device=device)
    full_qvel = torch.as_tensor(qvel_np, dtype=torch.float32, device=device)
    src_dt = 1.0 / fps
    if abs(src_dt - target_dt) >= 1.0e-7:
        full_qpos = _resample_full_qpos(full_qpos, layout, src_dt, target_dt)
        full_qvel = _resample_linear(full_qvel, src_dt, target_dt)
    if full_qvel.shape[0] != full_qpos.shape[0] or torch.allclose(
        full_qvel, torch.zeros_like(full_qvel)
    ):
        full_qvel = qvel_from_full_qpos(full_qpos, layout, dt=target_dt)

    body_pos, body_quat, body_lin_vel, body_ang_vel = _robot_body_kinematics(
        model,
        layout,
        full_qpos,
        full_qvel,
        device=device,
    )
    robot_qpos = layout.robot_qpos(full_qpos)
    robot_qvel = layout.robot_qvel(full_qvel)
    motion = InteractionMotion(
        path=path,
        motion_type="mujoco",
        fps=1.0 / target_dt,
        joint_pos=robot_qpos[:, 7:],
        joint_vel=robot_qvel[:, 6:],
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        contact=torch.empty((full_qpos.shape[0], 2), device=device),
        full_qpos=full_qpos,
        full_qvel=full_qvel,
        layout=layout,
    )
    return InteractionMotion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=motion.fps,
        joint_pos=motion.joint_pos,
        joint_vel=motion.joint_vel,
        body_pos_w=motion.body_pos_w,
        body_quat_w=motion.body_quat_w,
        body_lin_vel_w=motion.body_lin_vel_w,
        body_ang_vel_w=motion.body_ang_vel_w,
        contact=estimate_foot_contacts(motion),
        full_qpos=motion.full_qpos,
        full_qvel=motion.full_qvel,
        layout=layout,
    )


def qvel_from_full_qpos(
    qpos: torch.Tensor,
    layout: InteractionModelLayout,
    *,
    dt: float = POLICY_DT,
) -> torch.Tensor:
    """Finite-difference full qpos using MuJoCo freejoint qvel convention."""

    if qpos.ndim == 2:
        qpos_batched = qpos[:, None, :]
        squeeze = True
    elif qpos.ndim == 3:
        qpos_batched = qpos
        squeeze = False
    else:
        raise ValueError(f"Expected qpos shape (T, nq) or (T, N, nq), got {qpos.shape}.")
    if qpos_batched.shape[-1] != layout.nq:
        raise ValueError(f"Expected qpos dim {layout.nq}, got {qpos_batched.shape}.")

    qvel = torch.zeros(
        qpos_batched.shape[:-1] + (layout.nv,),
        dtype=qpos_batched.dtype,
        device=qpos_batched.device,
    )
    if qpos_batched.shape[0] <= 1:
        return qvel[:, 0] if squeeze else qvel

    freejoints = [
        (layout.root_qpos_adr, layout.root_qvel_adr),
        *[
            (obj.pose_qpos_adr, obj.pose_qvel_adr)
            for obj in layout.objects
            if obj.has_freejoint_pose
        ],
    ]
    for qpos_adr, qvel_adr in freejoints:
        assert qpos_adr is not None and qvel_adr is not None
        lin_vel = torch.zeros_like(qvel[..., qvel_adr : qvel_adr + 3])
        ang_vel_w = torch.zeros_like(lin_vel)
        lin_vel[:-1] = (
            qpos_batched[1:, :, qpos_adr : qpos_adr + 3]
            - qpos_batched[:-1, :, qpos_adr : qpos_adr + 3]
        ) / dt
        lin_vel[-1] = lin_vel[-2]
        delta_quat = quat_mul(
            qpos_batched[1:, :, qpos_adr + 3 : qpos_adr + 7],
            quat_inv(qpos_batched[:-1, :, qpos_adr + 3 : qpos_adr + 7]),
        )
        ang_vel_w[:-1] = axis_angle_from_quat(delta_quat) / dt
        ang_vel_w[-1] = ang_vel_w[-2]
        qvel[..., qvel_adr : qvel_adr + 6] = world_velocity_to_qvel(
            qpos_batched[..., qpos_adr : qpos_adr + 7],
            torch.cat([lin_vel, ang_vel_w], dim=-1),
        )

    qpos_idx = torch.as_tensor(
        layout.robot_joint_qpos_indices, device=qpos.device, dtype=torch.long
    )
    qvel_idx = torch.as_tensor(
        layout.robot_joint_qvel_indices, device=qpos.device, dtype=torch.long
    )
    joints = qpos_batched.index_select(-1, qpos_idx)
    joint_vel = torch.zeros_like(joints)
    joint_vel[:-1] = (joints[1:] - joints[:-1]) / dt
    joint_vel[-1] = joint_vel[-2]
    qvel.index_copy_(-1, qvel_idx, joint_vel)
    assigned_qvel = set(layout.robot_qvel_indices)
    assigned_qvel.update(range(layout.root_qvel_adr, layout.root_qvel_adr + 6))
    for obj in layout.objects:
        if obj.has_freejoint_pose:
            assert obj.pose_qvel_adr is not None
            assigned_qvel.update(range(obj.pose_qvel_adr, obj.pose_qvel_adr + 6))
        for joint in obj.joints:
            if joint.joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            if any(
                qvel_idx in assigned_qvel
                for qvel_idx in _joint_qvel_indices(joint.joint_type, joint.qvel_adr)
            ):
                continue
            if joint.joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                quat = qpos_batched[..., joint.qpos_adr : joint.qpos_adr + 4]
                ang_vel_w = torch.zeros(
                    qpos_batched.shape[:-1] + (3,),
                    dtype=qpos.dtype,
                    device=qpos.device,
                )
                delta_quat = quat_mul(quat[1:], quat_inv(quat[:-1]))
                ang_vel_w[:-1] = axis_angle_from_quat(delta_quat) / dt
                ang_vel_w[-1] = ang_vel_w[-2]
                qvel[..., joint.qvel_adr : joint.qvel_adr + 3] = ang_vel_w
                assigned_qvel.update(range(joint.qvel_adr, joint.qvel_adr + 3))
                continue
            pos = qpos_batched[..., joint.qpos_adr : joint.qpos_adr + 1]
            vel = torch.zeros_like(pos)
            vel[:-1] = (pos[1:] - pos[:-1]) / dt
            vel[-1] = vel[-2]
            qvel[..., joint.qvel_adr : joint.qvel_adr + 1] = vel
            assigned_qvel.add(joint.qvel_adr)
    return qvel[:, 0] if squeeze else qvel


def _joint_qvel_indices(joint_type: int, qvel_adr: int) -> tuple[int, ...]:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return tuple(range(qvel_adr, qvel_adr + 6))
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return tuple(range(qvel_adr, qvel_adr + 3))
    return (qvel_adr,)


def _robot_body_kinematics(
    model: mujoco.MjModel,
    layout: InteractionModelLayout,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = mujoco.MjData(model)
    body_pos = []
    body_quat = []
    body_lin_vel = []
    body_ang_vel = []
    qpos_np = qpos.detach().cpu().numpy()
    qvel_np = qvel.detach().cpu().numpy()
    for frame_idx in range(qpos_np.shape[0]):
        data.qpos[:] = qpos_np[frame_idx]
        data.qvel[:] = qvel_np[frame_idx]
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[list(layout.robot_body_ids)].copy())
        body_quat.append(data.xquat[list(layout.robot_body_ids)].copy())
        cvel = data.cvel[list(layout.robot_body_ids)].copy()
        root_subtree_com = data.subtree_com[layout.root_body_id].copy()
        xpos = body_pos[-1]
        ang_vel_w = cvel[:, 0:3]
        lin_vel_c = cvel[:, 3:6]
        lin_vel_w = lin_vel_c - np.cross(ang_vel_w, root_subtree_com[None, :] - xpos)
        body_lin_vel.append(lin_vel_w)
        body_ang_vel.append(ang_vel_w)
    return (
        torch.as_tensor(np.stack(body_pos), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(body_quat), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(body_lin_vel), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(body_ang_vel), dtype=torch.float32, device=device),
    )


def _resolve_fps(
    raw: np.lib.npyio.NpzFile,
    source_dt: float | None,
    target_dt: float,
) -> float:
    if source_dt is not None:
        return 1.0 / float(source_dt)
    if "fps" in raw.files:
        return float(raw["fps"].item())
    if "dt" in raw.files:
        return 1.0 / float(raw["dt"].item())
    return 1.0 / target_dt


def _flatten_traj_array(value: np.ndarray, dim: int) -> np.ndarray:
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 3:
        value = value.reshape(-1, value.shape[-1])
    if value.ndim != 2 or value.shape[-1] != dim:
        raise ValueError(f"Expected trajectory shape (T, {dim}) or (..., {dim}), got {value.shape}.")
    return np.ascontiguousarray(value)


def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    small = dot > 0.9995
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1e-8)
    theta = theta_0 * alpha
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    out = s0 * q0 + s1 * q1
    lerp = q0 + alpha * (q1 - q0)
    out = torch.where(small, lerp, out)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _resample_linear(x: torch.Tensor, src_dt: float, target_dt: float) -> torch.Tensor:
    if x.shape[0] <= 1 or abs(src_dt - target_dt) < 1e-7:
        return x
    duration = (x.shape[0] - 1) * src_dt
    out_len = int(np.floor(duration / target_dt + 1e-6)) + 1
    t = torch.arange(out_len, device=x.device, dtype=x.dtype) * target_dt
    u = (t / src_dt).clamp(max=x.shape[0] - 1)
    i0 = torch.floor(u).long()
    i1 = torch.clamp(i0 + 1, max=x.shape[0] - 1)
    a = (u - i0.to(u.dtype)).view(-1, *([1] * (x.ndim - 1)))
    return x[i0] * (1.0 - a) + x[i1] * a


def _resample_quat(x: torch.Tensor, src_dt: float, target_dt: float) -> torch.Tensor:
    if x.shape[0] <= 1 or abs(src_dt - target_dt) < 1e-7:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    duration = (x.shape[0] - 1) * src_dt
    out_len = int(np.floor(duration / target_dt + 1e-6)) + 1
    t = torch.arange(out_len, device=x.device, dtype=x.dtype) * target_dt
    u = (t / src_dt).clamp(max=x.shape[0] - 1)
    i0 = torch.floor(u).long()
    i1 = torch.clamp(i0 + 1, max=x.shape[0] - 1)
    a = (u - i0.to(u.dtype)).view(-1, *([1] * (x.ndim - 1)))
    return _slerp(x[i0], x[i1], a)


def _resample_full_qpos(
    qpos: torch.Tensor,
    layout: InteractionModelLayout,
    src_dt: float,
    target_dt: float,
) -> torch.Tensor:
    out = _resample_linear(qpos, src_dt, target_dt)
    for quat_slice in _quat_qpos_slices(layout):
        out[:, quat_slice] = _resample_quat(
            qpos[:, quat_slice],
            src_dt,
            target_dt,
        )
    return out


def _quat_qpos_slices(layout: InteractionModelLayout) -> tuple[slice, ...]:
    slices = [slice(layout.root_qpos_adr + 3, layout.root_qpos_adr + 7)]
    for obj in layout.objects:
        if obj.has_freejoint_pose:
            assert obj.pose_qpos_adr is not None
            slices.append(slice(obj.pose_qpos_adr + 3, obj.pose_qpos_adr + 7))
        for joint in obj.joints:
            if joint.joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                slices.append(slice(joint.qpos_adr, joint.qpos_adr + 4))
    return tuple(slices)


def slice_interaction_motion(
    motion: InteractionMotion,
    start: int,
    length: int,
) -> InteractionMotion:
    start = int(start)
    length = int(length)

    def sl(value: torch.Tensor) -> torch.Tensor:
        out = value[start : start + length]
        if out.shape[0] < length:
            repeats = [length - out.shape[0]] + [1] * (out.ndim - 1)
            out = torch.cat([out, out[-1:].repeat(*repeats)], dim=0)
        return out.contiguous()

    return InteractionMotion(
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
        full_qpos=sl(motion.full_state_qpos()),
        full_qvel=sl(motion.full_state_qvel()),
        layout=motion.layout,
    )


__all__ = [
    "InteractionMotion",
    "load_interaction_motion",
    "qvel_from_full_qpos",
    "slice_interaction_motion",
]
