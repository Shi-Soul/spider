"""Replay video rendering for G1 WBC interaction trajectories."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from spider.tasks.g1_wbc_interaction.layout import load_interaction_model


CYAN_RGBA = np.array([0.0, 0.85, 1.0, 0.33], dtype=np.float32)


def render_interaction_comparison_video(
    *,
    model_path: str | Path,
    subject_qpos: np.ndarray,
    reference_qpos: np.ndarray,
    out_path: str | Path,
    label: str,
    fps: int = 30,
    width: int = 960,
    height: int = 720,
    camera: str = "auto",
) -> None:
    """Render one solid trajectory with a cyan reference ghost overlay."""

    model = load_interaction_model(model_path)
    subject_qpos = _flat_qpos(subject_qpos, model.nq)
    reference_qpos = _flat_qpos(reference_qpos, model.nq)
    n = min(subject_qpos.shape[0], reference_qpos.shape[0])
    if n <= 0:
        raise ValueError("Need at least one frame to render a video.")
    subject_qpos = subject_qpos[:n]
    reference_qpos = reference_qpos[:n]

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    data = mujoco.MjData(model)
    ref_data = mujoco.MjData(model)
    original_rgba = model.geom_rgba.copy()
    ghost_mask = _ghost_geom_mask(model)
    cameras = _make_follow_cameras(subject_qpos, reference_qpos)
    frames: list[np.ndarray] = []
    try:
        for frame_idx in range(n):
            data.qpos[:] = subject_qpos[frame_idx]
            ref_data.qpos[:] = reference_qpos[frame_idx]
            mujoco.mj_forward(model, data)
            mujoco.mj_forward(model, ref_data)
            cam = cameras[frame_idx] if camera == "auto" else camera
            renderer.update_scene(data, camera=cam)
            frame = renderer.render().copy()

            model.geom_rgba[:] = original_rgba
            model.geom_rgba[ghost_mask, :] = CYAN_RGBA
            renderer.update_scene(ref_data, camera=cam)
            ref_frame = renderer.render().copy()
            frame = cv2.addWeighted(frame, 0.68, ref_frame, 0.32, 0.0)

            model.geom_rgba[:] = original_rgba
            frames.append(_label(frame, f"cyan: reference  solid: {label}"))
    finally:
        model.geom_rgba[:] = original_rgba
        renderer.close()

    imageio.mimsave(out_path, frames, fps=int(fps))


def _flat_qpos(qpos: np.ndarray, nq: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.ndim != 2 or qpos.shape[-1] != nq:
        raise ValueError(f"Expected qpos shape (T, {nq}) or (..., {nq}), got {qpos.shape}.")
    return qpos


def _ghost_geom_mask(model: mujoco.MjModel) -> np.ndarray:
    mask = np.ones(model.ngeom, dtype=bool)
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(model.geom_type[geom_id])
        if name in ("floor", "terrain") or geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            mask[geom_id] = False
    return mask


def _make_follow_cameras(
    subject_qpos: np.ndarray,
    reference_qpos: np.ndarray,
) -> list[mujoco.MjvCamera]:
    columns = [subject_qpos[:, :3], reference_qpos[:, :3]]
    roots = np.stack(columns, axis=0)
    cameras = []
    for frame_idx in range(roots.shape[1]):
        frame_roots = roots[:, frame_idx, :]
        xyz_min = frame_roots.min(axis=0)
        xyz_max = frame_roots.max(axis=0)
        center = 0.5 * (xyz_min + xyz_max)
        span_xy = float(np.linalg.norm(xyz_max[:2] - xyz_min[:2]))
        span_z = float(max(xyz_max[2] - xyz_min[2], 0.5))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.lookat[2] = max(float(center[2]), 0.75)
        camera.distance = max(2.3, 1.2 * span_xy + 1.7, 2.2 * span_z)
        camera.azimuth = 135.0
        camera.elevation = -18.0
        cameras.append(camera)
    return cameras


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    bar_h = 34
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], bar_h), (8, 8, 8), thickness=-1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0.0, out)
    cv2.putText(
        out,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return out


__all__ = ["render_interaction_comparison_video"]
