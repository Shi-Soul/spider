"""Torch math helpers used by the standalone G1 WBC task."""

from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        q1, q2 = torch.broadcast_tensors(q1, q2)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def _broadcast_quat_vec(
    quat: torch.Tensor, vec: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    while quat.ndim < vec.ndim:
        quat = quat.unsqueeze(-2)
    while vec.ndim < quat.ndim:
        vec = vec.unsqueeze(-2)
    prefix = torch.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    quat = quat.expand(prefix + (4,))
    vec = vec.expand(prefix + (3,))
    return quat, vec, vec.shape


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat, vec, shape = _broadcast_quat_vec(quat, vec)
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2.0
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat, vec, shape = _broadcast_quat_vec(quat, vec)
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2.0
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    shape = quat.shape
    q = quat.reshape(-1, 4)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    out = torch.zeros_like(q)
    out[:, 0] = torch.cos(yaw / 2.0)
    out[:, 3] = torch.sin(yaw / 2.0)
    return normalize(out).view(shape)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    out = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return out.reshape(quaternions.shape[:-1] + (3, 3))


def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    t12 = quat_apply(q10, t02 - t01) if t02 is not None else quat_apply(q10, -t01)
    return t12, q12


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0).to(quat.dtype))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    denom = torch.where(
        angle.abs() > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48.0,
    )
    return quat[..., 1:4] / denom.unsqueeze(-1)


def quat_from_axis_angle(axis_angle: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Convert an axis-angle vector to a quaternion in wxyz convention."""

    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=eps)
    half_angle = 0.5 * angle
    sin_half = torch.sin(half_angle)
    quat = torch.cat([torch.cos(half_angle), axis * sin_half], dim=-1)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(angle <= eps, identity, normalize(quat))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.norm(axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2))), dim=-1)


def qvel_to_world_velocity(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    """Convert MuJoCo free-joint qvel angular part from local to world frame."""
    return torch.cat([qvel[..., :3], quat_apply(qpos[..., 3:7], qvel[..., 3:6])], dim=-1)


def world_velocity_to_qvel(qpos: torch.Tensor, world_vel: torch.Tensor) -> torch.Tensor:
    """Convert root velocity from world angular frame to MuJoCo qvel convention."""
    return torch.cat(
        [world_vel[..., :3], quat_apply_inverse(qpos[..., 3:7], world_vel[..., 3:6])],
        dim=-1,
    )


def finite_difference_root_velocity(qpos: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute world-frame root velocity from qpos trajectory."""
    lin = torch.zeros(qpos.shape[:-1] + (3,), dtype=qpos.dtype, device=qpos.device)
    ang = torch.zeros_like(lin)
    if qpos.shape[0] <= 1:
        return torch.cat([lin, ang], dim=-1)
    lin[:-1] = (qpos[1:, :3] - qpos[:-1, :3]) / dt
    lin[-1] = lin[-2]
    dq = quat_mul(qpos[1:, 3:7], quat_inv(qpos[:-1, 3:7]))
    ang[:-1] = axis_angle_from_quat(dq) / dt
    ang[-1] = ang[-2]
    return torch.cat([lin, ang], dim=-1)
