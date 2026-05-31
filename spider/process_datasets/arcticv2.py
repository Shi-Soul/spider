# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process Arctic raw_seqs into SPIDER's mjwp_fast schema.

This pipeline reads Arctic-v2 ``raw_seqs/<subject>/<sequence>.{mano,object}.npy``
plus ``meta/object_vtemplates/<obj>/{top,bottom}.obj`` and produces a single
clip per (subject, sequence). Compared to ``oakinkv2.py`` this pipeline:

1. Filters to the 6 dexmachina-pinned objects
   (box, ketchup, laptop, mixer, notebook, waffleiron). See
   :mod:`spider.postprocess.evaluate_dexmachina` lines 230-234.
2. Treats the object as RIGID: welds top+bottom at frame-0 articulation angle
   and writes a single ``visual.obj``. Articulated-object support is a
   future feature; we log a warning at run time.
3. Clips to the grasp stage only: starts ``pre_grasp_seconds`` before the
   first frame where any hand vert is within 5cm of the object, and ends
   ``post_lift_seconds`` after the object lifts above
   ``frame0_z + lift_z_thresh``.
4. Bimanual: both hand poses come from MANO FK; ``active_hand`` records
   which hand(s) actually contact the object.

Author: Chaoyi Pan
Date: 2026-05-31
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import loguru
import mujoco
import numpy as np
import pymeshlab
import torch
import trimesh
import tyro
from scipy.spatial.transform import Rotation as R

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Arctic mocap rate
FPS_MOCAP = 30
# manotorch MANO 21-joint indices for the 5 fingertips (SNAP layout)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
# bone connectivity for the 21-joint skeleton (parent -> child)
SKELETON_BONES = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# 6 dexmachina-pinned objects.
# See spider/postprocess/evaluate_dexmachina.py:230-234 for the canonical
# enumeration; not exported there, so duplicated here.
ARCTIC_OBJECTS = ["box", "ketchup", "laptop", "mixer", "notebook", "waffleiron"]


def _ensure_mano_assets_layout(mano_assets_root: str) -> str:
    """Return a directory ``D`` such that ``D/models/MANO_RIGHT.pkl`` exists.

    manotorch.ManoLayer expects ``mano_assets_root/models/MANO_<side>.pkl``.
    Arctic ships ``body_models/mano/MANO_<side>.pkl`` and a separate
    ``body_models/models/smplx/`` folder. If the user passed
    ``~/arctic/unpack/body_models``, symlink ``mano/MANO_*.pkl`` into
    ``models/`` so manotorch can find them; if the user passed
    ``~/arctic/unpack/body_models/mano`` directly, create a sibling
    ``models/`` symlink target.
    """
    mano_assets_root = os.path.abspath(os.path.expanduser(mano_assets_root))
    target = os.path.join(mano_assets_root, "models")
    os.makedirs(target, exist_ok=True)
    # Candidates where the .pkl might actually live
    candidates_dirs = [
        mano_assets_root,
        os.path.join(mano_assets_root, "mano"),
        os.path.dirname(mano_assets_root),  # parent
        os.path.join(os.path.dirname(mano_assets_root), "mano"),
    ]
    for side in ("RIGHT", "LEFT"):
        dst = os.path.join(target, f"MANO_{side}.pkl")
        if os.path.exists(dst):
            continue
        src = None
        for d in candidates_dirs:
            cand = os.path.join(d, f"MANO_{side}.pkl")
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            raise FileNotFoundError(
                f"Could not find MANO_{side}.pkl under any of: {candidates_dirs}. "
                "Pass --mano-assets-root pointing at a directory whose "
                "siblings/children contain MANO_RIGHT.pkl and MANO_LEFT.pkl."
            )
        os.symlink(src, dst)
    return mano_assets_root


def _patch_chumpy_compat() -> None:
    """Make chumpy 0.70 importable on Python 3.12 + NumPy 2.x.

    chumpy (manotorch transitive dep) does ``from numpy import bool, int, ...``
    and ``inspect.getargspec``, both removed in modern stacks. Inject the
    legacy attributes so chumpy can import; we never use those classes
    ourselves.
    """
    import inspect as _inspect

    import numpy as _np

    if not hasattr(_inspect, "getargspec"):
        _inspect.getargspec = _inspect.getfullargspec  # type: ignore[attr-defined]
    for alias, impl in [
        ("bool", bool), ("int", int), ("float", float), ("complex", complex),
        ("object", object), ("unicode", str), ("str", str),
    ]:
        if not hasattr(_np, alias):
            setattr(_np, alias, impl)


def _load_mano_layers(mano_assets_root: str) -> tuple:
    """Return (right_layer, left_layer) ManoLayer instances."""
    _patch_chumpy_compat()
    from manotorch.manolayer import ManoLayer

    root = _ensure_mano_assets_layout(mano_assets_root)
    right = ManoLayer(
        mano_assets_root=root,
        rot_mode="axisang",
        side="right",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    )
    left = ManoLayer(
        mano_assets_root=root,
        rot_mode="axisang",
        side="left",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    )
    return right, left


def _mano_fk(
    mano_layer,
    rot: np.ndarray,    # (T, 3) global orient axis-angle
    pose: np.ndarray,   # (T, 45) finger axis-angle
    trans: np.ndarray,  # (T, 3) wrist translation, meters
    shape: np.ndarray,  # (10,) betas
) -> tuple[np.ndarray, np.ndarray]:
    """Run MANO FK with arctic params.

    Returns (joints_world (T,21,3), verts_world (T, 778, 3)).
    """
    T = rot.shape[0]
    pose_coeffs = torch.from_numpy(
        np.concatenate([rot, pose], axis=-1)
    ).float()  # (T, 48)
    betas = torch.from_numpy(np.tile(shape[None, :], (T, 1))).float()
    tsl = torch.from_numpy(trans).float()
    with torch.no_grad():
        out = mano_layer(pose_coeffs=pose_coeffs, betas=betas)
        joints = out.joints + tsl[:, None, :]
        verts = out.verts + tsl[:, None, :]
    return joints.cpu().numpy(), verts.cpu().numpy()


def _build_rigid_object_mesh(
    obj_template_dir: str,
    arti_angle_rad: float,
    out_path: str,
) -> None:
    """Concatenate top.obj + bottom.obj after applying a frame-0 articulation.

    Arctic articulates the top part about the z-axis through the origin
    (the canonical convention). We rotate top.obj vertices by ``arti_angle_rad``
    about z, then merge with bottom.obj into a single welded mesh.

    Vertices are kept in millimetres (the Arctic template native unit) and
    converted to metres on output.
    """
    top = trimesh.load(os.path.join(obj_template_dir, "top.obj"), process=False)
    bot = trimesh.load(os.path.join(obj_template_dir, "bottom.obj"), process=False)
    top_verts = np.asarray(top.vertices, dtype=np.float64)
    bot_verts = np.asarray(bot.vertices, dtype=np.float64)
    top_faces = np.asarray(top.faces, dtype=np.int64)
    bot_faces = np.asarray(bot.faces, dtype=np.int64)

    # Apply articulation: rotate top about z-axis through origin.
    # Arctic stores arti as a positive scalar; sign matches the canonical
    # opening direction in arctic.utils.articulate.
    rot = R.from_rotvec(np.array([0.0, 0.0, arti_angle_rad]))
    top_verts = rot.apply(top_verts)

    # Merge.
    n_top = top_verts.shape[0]
    verts = np.concatenate([top_verts, bot_verts], axis=0)
    faces = np.concatenate([top_faces, bot_faces + n_top], axis=0)
    # mm -> m
    verts = verts / 1000.0

    merged = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    merged.export(out_path)


def _detect_grasp_window(
    obj_verts_world: np.ndarray,   # (T, V_obj, 3)
    rh_verts_world: np.ndarray,    # (T, 778, 3)
    lh_verts_world: np.ndarray,    # (T, 778, 3)
    fps: float,
    pre_grasp_seconds: float,
    post_lift_seconds: float,
    lift_z_thresh: float,
    contact_thresh: float = 0.05,
) -> tuple[int, int, str]:
    """Detect the grasp window and active hand.

    Returns (start_frame, end_frame_exclusive, active_hand) where active_hand
    is 'right', 'left', or 'both'.
    """
    T = obj_verts_world.shape[0]
    # Sub-sample object verts for speed (every ~50th vertex is fine for a
    # nearest-distance lower bound).
    stride = max(1, obj_verts_world.shape[1] // 200)
    obj_sub = obj_verts_world[:, ::stride, :]
    rh_sub = rh_verts_world[:, ::8, :]
    lh_sub = lh_verts_world[:, ::8, :]

    rh_dist = np.empty(T)
    lh_dist = np.empty(T)
    for t in range(T):
        d_rh = np.linalg.norm(
            rh_sub[t][:, None, :] - obj_sub[t][None, :, :], axis=-1
        ).min()
        d_lh = np.linalg.norm(
            lh_sub[t][:, None, :] - obj_sub[t][None, :, :], axis=-1
        ).min()
        rh_dist[t] = d_rh
        lh_dist[t] = d_lh

    rh_contacts = np.where(rh_dist < contact_thresh)[0]
    lh_contacts = np.where(lh_dist < contact_thresh)[0]
    if rh_contacts.size == 0 and lh_contacts.size == 0:
        # Fall back to the closest-approach frame; treat both hands as active.
        contact_start = int(np.argmin(np.minimum(rh_dist, lh_dist)))
        active = "both"
    else:
        if rh_contacts.size and lh_contacts.size:
            contact_start = int(min(rh_contacts[0], lh_contacts[0]))
            active = "both"
        elif rh_contacts.size:
            contact_start = int(rh_contacts[0])
            active = "right"
        else:
            contact_start = int(lh_contacts[0])
            active = "left"

    # Object's lowest world z (per-frame), used to detect lift.
    obj_min_z = obj_verts_world[:, :, 2].min(axis=1)
    z0 = obj_min_z[0]
    lift_frames = np.where(obj_min_z > z0 + lift_z_thresh)[0]
    # The "stable lift" frame is the first frame after contact_start where
    # the object has lifted; if no lift happens, use the end of the clip.
    lift_after_contact = lift_frames[lift_frames >= contact_start]
    lift_frame = (
        int(lift_after_contact[0]) if lift_after_contact.size > 0 else T - 1
    )

    pre_n = int(round(pre_grasp_seconds * fps))
    post_n = int(round(post_lift_seconds * fps))
    start = max(0, contact_start - pre_n)
    end = min(T, lift_frame + post_n + 1)
    if end <= start:
        end = min(T, start + 1)
    return start, end, active


def main(
    arctic_root: Path = Path("~/arctic"),  # noqa: B008  expanded inside main
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    subject: str = "s01",
    sequence: str = "box_use_01",
    data_id: int = 0,
    embodiment_type: str = "bimanual",
    robot_type: str = "xhand",
    pre_grasp_seconds: float = 1.0,
    post_lift_seconds: float = 2.0,
    lift_z_thresh: float = 0.05,
    target_fps: float = 30.0,
    mano_assets_root: str | None = None,
    no_preview: bool = False,
    show_viewer: bool = True,
):
    """Process a single (subject, sequence) Arctic clip.

    Args:
        arctic_root: Root of the unpacked Arctic dataset (containing
            ``unpack/arctic_data/data/raw_seqs`` and ``unpack/body_models``).
        dataset_dir: Where to write processed outputs.
        subject: ``s01`` ... ``s10``.
        sequence: e.g. ``box_use_01``. Object name is the prefix before the
            first underscore.
        data_id: Output sub-directory.
        embodiment_type: Always ``bimanual`` for Arctic.
        robot_type: e.g. ``xhand``, ``sharpa``. Used only to name the output
            dir; the schema itself is robot-agnostic.
        pre_grasp_seconds: Pre-contact window before clipping.
        post_lift_seconds: Post-lift settle window.
        lift_z_thresh: Object min-z must rise this much above frame-0 to
            count as "lifted".
        target_fps: Output FPS. Arctic raw is 30 Hz; default keeps it at 30.
        mano_assets_root: Directory containing MANO pickles. Defaults to
            ``<arctic_root>/unpack/body_models``.
        no_preview: Alias for ``--no-show-viewer``.
        show_viewer: Open the viser preview after writing the NPZ.
    """
    if no_preview:
        show_viewer = False
    if embodiment_type != "bimanual":
        loguru.logger.warning(
            f"arcticv2 is bimanual; got embodiment_type={embodiment_type}. "
            "Proceeding but downstream consumers expect 'bimanual'."
        )

    arctic_root = Path(os.path.abspath(os.path.expanduser(str(arctic_root))))
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    if mano_assets_root is None:
        for c in (
            arctic_root / "unpack" / "body_models",
            arctic_root / "body_models",
            arctic_root / "unpack" / "body_models" / "mano",
            arctic_root / "body_models" / "mano",
        ):
            if c.exists():
                mano_assets_root = str(c)
                break
        else:
            mano_assets_root = str(arctic_root / "unpack" / "body_models")

    # Object-class filter (the 6 dexmachina objects).
    obj_name = sequence.split("_")[0]
    if obj_name not in ARCTIC_OBJECTS:
        raise ValueError(
            f"Sequence '{sequence}' object='{obj_name}' is not in the "
            f"dexmachina-pinned set {ARCTIC_OBJECTS}."
        )

    # Articulation discarded warning.
    loguru.logger.warning(
        "arcticv2: articulation discarded; treating object as rigid (welded "
        "at frame-0). Articulated-object support is a future feature."
    )

    # 1. Load raw arrays.
    seq_dir = arctic_root / "unpack/arctic_data/data/raw_seqs" / subject
    mano_path = seq_dir / f"{sequence}.mano.npy"
    object_path = seq_dir / f"{sequence}.object.npy"
    if not mano_path.exists():
        raise FileNotFoundError(f"Missing {mano_path}")
    if not object_path.exists():
        raise FileNotFoundError(f"Missing {object_path}")
    mano = np.load(mano_path, allow_pickle=True).item()
    obj = np.load(object_path, allow_pickle=True)  # (T, 7) float32
    if obj.ndim != 2 or obj.shape[-1] != 7:
        raise ValueError(f"Unexpected object array shape: {obj.shape}")

    # Object: [arti_rad, axang_x, axang_y, axang_z, tx_mm, ty_mm, tz_mm]
    obj_arti = obj[:, 0].astype(np.float32)
    obj_axang = obj[:, 1:4].astype(np.float32)
    obj_trans_m = (obj[:, 4:7].astype(np.float32)) / 1000.0  # mm -> m

    T = obj.shape[0]
    if mano["right"]["rot"].shape[0] != T:
        raise ValueError(
            f"frame-count mismatch: object T={T}, mano T={mano['right']['rot'].shape[0]}"
        )
    loguru.logger.info(
        f"Loaded {sequence} ({subject}): T={T} frames @ {FPS_MOCAP}Hz, "
        f"object={obj_name}"
    )

    obj_template_dir = (
        arctic_root / "unpack/arctic_data/data/meta/object_vtemplates" / obj_name
    )

    # 2. MANO FK on the FULL trajectory (so the grasp detector sees real
    # vertex distances). Both hands.
    rh_layer, lh_layer = _load_mano_layers(mano_assets_root)
    rh_joints, rh_verts = _mano_fk(
        rh_layer, mano["right"]["rot"], mano["right"]["pose"],
        mano["right"]["trans"], mano["right"]["shape"],
    )
    lh_joints, lh_verts = _mano_fk(
        lh_layer, mano["left"]["rot"], mano["left"]["pose"],
        mano["left"]["trans"], mano["left"]["shape"],
    )

    # 3. Build per-frame object world verts using the un-articulated combined
    # mesh.obj (good enough for distance-based grasp detection; the welded
    # mesh used for export is rebuilt after we know ``start``).
    full_mesh = trimesh.load(
        str(obj_template_dir / "mesh.obj"), process=False,
    )
    full_verts_local = np.asarray(full_mesh.vertices, dtype=np.float64) / 1000.0
    sub = full_verts_local[:: max(1, full_verts_local.shape[0] // 500)]
    obj_world_verts = np.empty((T, sub.shape[0], 3), dtype=np.float64)
    for t in range(T):
        Rt = R.from_rotvec(obj_axang[t])
        obj_world_verts[t] = Rt.apply(sub) + obj_trans_m[t]

    # 4. Detect grasp window + active hand on the full clip.
    start, end, active_hand = _detect_grasp_window(
        obj_world_verts, rh_verts, lh_verts,
        fps=FPS_MOCAP,
        pre_grasp_seconds=pre_grasp_seconds,
        post_lift_seconds=post_lift_seconds,
        lift_z_thresh=lift_z_thresh,
    )
    loguru.logger.info(
        f"clipped frame range = [{start}, {end}) ({end - start} frames @ "
        f"{FPS_MOCAP}Hz); active_hand={active_hand}"
    )

    # 5. Build the rigid welded mesh at the FIRST CLIP FRAME's articulation.
    weld_arti = float(obj_arti[start])
    safe_name = f"{obj_name}_{subject}_{sequence}_arti{weld_arti:.3f}".replace(
        ".", "p"
    )
    mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir, dataset_name="arcticv2", object_name=safe_name,
    )
    os.makedirs(mesh_dir, exist_ok=True)
    out_mesh = os.path.join(mesh_dir, "visual.obj")
    _build_rigid_object_mesh(str(obj_template_dir), weld_arti, out_mesh)
    loguru.logger.info(
        f"Wrote welded rigid mesh (clip-frame-0 arti={weld_arti:.3f} rad) to "
        f"{out_mesh}"
    )

    # 6. Slice all per-frame arrays to [start, end).
    rh_joints = rh_joints[start:end]
    rh_verts = rh_verts[start:end]
    lh_joints = lh_joints[start:end]
    lh_verts = lh_verts[start:end]
    obj_axang_clip = obj_axang[start:end]
    obj_trans_clip = obj_trans_m[start:end]
    rh_rot_clip = mano["right"]["rot"][start:end]
    lh_rot_clip = mano["left"]["rot"][start:end]

    # 7. Optional resample. Arctic is 30 fps; default target_fps=30 -> no-op.
    n_in = end - start
    if abs(target_fps - FPS_MOCAP) > 1e-3:
        n_out = int(round(n_in * target_fps / FPS_MOCAP))
        idx = np.linspace(0, n_in - 1, num=max(2, n_out)).round().astype(int)
        rh_joints = rh_joints[idx]
        rh_verts = rh_verts[idx]
        lh_joints = lh_joints[idx]
        lh_verts = lh_verts[idx]
        obj_axang_clip = obj_axang_clip[idx]
        obj_trans_clip = obj_trans_clip[idx]
        rh_rot_clip = rh_rot_clip[idx]
        lh_rot_clip = lh_rot_clip[idx]
        n = len(idx)
    else:
        n = n_in
    loguru.logger.info(f"output {n} frames @ {target_fps}Hz")

    # 8. Output dirs.
    task = f"{subject}-{sequence}"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="arcticv2",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    # 9. Coordinate frame: Arctic captures are already z-up in metres
    # (top.obj + obj_trans/1000 + MANO trans share the same world). No
    # global rotation is needed.
    unit_quat = np.array([1.0, 0.0, 0.0, 0.0])

    qpos_wrist_right = np.zeros((n, 7))
    qpos_finger_right = np.zeros((n, 5, 7))
    qpos_obj_right = np.zeros((n, 7))
    qpos_wrist_left = np.zeros((n, 7))
    qpos_finger_left = np.zeros((n, 5, 7))
    qpos_obj_left = np.zeros((n, 7))

    # Per-hand wrist + fingertips from the FK output (joint 0 = wrist).
    for i in range(n):
        qpos_wrist_right[i, :3] = rh_joints[i, 0]
        r = R.from_rotvec(rh_rot_clip[i])
        xyzw = r.as_quat()
        qpos_wrist_right[i, 3:] = np.concatenate([xyzw[3:], xyzw[:3]])  # wxyz

        qpos_wrist_left[i, :3] = lh_joints[i, 0]
        r = R.from_rotvec(lh_rot_clip[i])
        xyzw = r.as_quat()
        qpos_wrist_left[i, 3:] = np.concatenate([xyzw[3:], xyzw[:3]])

        for j in range(5):
            qpos_finger_right[i, j, :3] = rh_joints[i, FINGERTIP_INDICES[j]]
            qpos_finger_right[i, j, 3:] = unit_quat
            qpos_finger_left[i, j, :3] = lh_joints[i, FINGERTIP_INDICES[j]]
            qpos_finger_left[i, j, 3:] = unit_quat

    # Object pose is the bottom link's world pose; both hands track the same
    # rigid object.
    for i in range(n):
        r = R.from_rotvec(obj_axang_clip[i])
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_obj_right[i, :3] = obj_trans_clip[i]
        qpos_obj_right[i, 3:] = wxyz
        qpos_obj_left[i, :3] = obj_trans_clip[i]
        qpos_obj_left[i, 3:] = wxyz

    # 10. Lowest-z statistics for floor/plate placement.
    visual_mesh = pymeshlab.MeshSet()
    visual_mesh.load_new_mesh(out_mesh)
    obj_verts_local_full = np.asarray(visual_mesh.current_mesh().vertex_matrix())
    R_obj0 = R.from_rotvec(obj_axang_clip[0])
    obj_verts_world_z_frame0 = float(
        (R_obj0.apply(obj_verts_local_full) + qpos_obj_right[0, :3]).min(axis=0)[2]
    )
    obj_min_z_per_frame = []
    for k in range(n):
        Rk = R.from_rotvec(obj_axang_clip[k])
        zk = (Rk.apply(obj_verts_local_full) + qpos_obj_right[k, :3]).min(axis=0)[2]
        obj_min_z_per_frame.append(zk)
    obj_min_z_traj = float(np.min(obj_min_z_per_frame))
    scene_min_z = float(min(
        obj_min_z_traj,
        qpos_wrist_right[:, 2].min(),
        qpos_finger_right[:, :, 2].min(),
        qpos_wrist_left[:, 2].min(),
        qpos_finger_left[:, :, 2].min(),
    ))
    object_descends = obj_min_z_traj < obj_verts_world_z_frame0 - 0.02

    # 11. task_info.
    rel_mesh_dir = str(Path(mesh_dir).relative_to(dataset_dir))
    task_info = {
        "task": task,
        "dataset_name": "arcticv2",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": rel_mesh_dir,
        "left_object_mesh_dir": rel_mesh_dir,
        "ref_dt": 1.0 / target_fps,
        "n_frames": n,
        "obj_first_frame_lowest_world_z": obj_verts_world_z_frame0,
        "scene_lowest_world_z": scene_min_z,
        "object_descends_from_frame0": bool(object_descends),
        # arctic-specific provenance
        "arctic_subject": subject,
        "arctic_object": obj_name,
        "arctic_clip_use": sequence,
        "arctic_frame_range": [int(start), int(end)],
        "active_hand": active_hand,
        "pre_grasp_seconds": pre_grasp_seconds,
        "post_lift_seconds": post_lift_seconds,
        "lift_z_thresh": lift_z_thresh,
        "articulation_discarded": True,
        "frame0_arti_rad": float(obj_arti[start]),
    }
    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")
    loguru.logger.info(
        f"Object frame-0 lowest world z = {obj_verts_world_z_frame0:+.3f} m; "
        f"scene lowest z = {scene_min_z:+.3f} m"
        + (" (object descends from frame 0)" if object_descends else " (object stays above frame-0 z)")
        + ". Final needs_object_support is decided by decompose_fast/decompose."
    )

    if not show_viewer:
        return

    # 12. Optional viser preview: build a minimal mocap scene with the rigid
    # object mesh and a per-hand 21-joint skeleton overlay.
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")
    object_right_handle = mj_spec.worldbody.add_body(name="right_object", mocap=True)
    object_right_handle.add_site(
        name="right_object", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.001, 0.001, 0.001], rgba=[1, 0, 0, 0], group=3,
    )
    mj_spec.add_mesh(name="right_object", file=out_mesh)
    object_right_handle.add_geom(
        name="right_object", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="right_object", pos=[0, 0, 0], quat=[1, 0, 0, 0],
        group=0, condim=1,
    )
    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)

    qpos_list = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
        ],
        axis=1,
    )

    from spider.viewers.viser_viewer import (
        _STATE,
        build_and_log_scene_from_spec,
        init_viser,
        log_frame,
    )

    init_viser(app_name="spider-arcticv2")
    body_entity_and_ids = build_and_log_scene_from_spec(
        mj_spec, mj_model, build_ref=False
    )
    server = _STATE.server

    rh_handles = [
        server.scene.add_icosphere(
            f"hand/right/joint_{i}", radius=0.006,
            color=(220, 80, 80) if i in FINGERTIP_INDICES else (80, 130, 220),
            position=tuple(rh_joints[0, i]),
        )
        for i in range(21)
    ]
    lh_handles = [
        server.scene.add_icosphere(
            f"hand/left/joint_{i}", radius=0.006,
            color=(220, 140, 80) if i in FINGERTIP_INDICES else (130, 200, 80),
            position=tuple(lh_joints[0, i]),
        )
        for i in range(21)
    ]
    bone_handles = []
    for parent, child in SKELETON_BONES:
        h_r = server.scene.add_spline_catmull_rom(
            f"hand/right/bone_{parent}_{child}",
            positions=np.stack([rh_joints[0, parent], rh_joints[0, child]]),
            color=(255, 200, 50), line_width=2.5,
        )
        h_l = server.scene.add_spline_catmull_rom(
            f"hand/left/bone_{parent}_{child}",
            positions=np.stack([lh_joints[0, parent], lh_joints[0, child]]),
            color=(180, 255, 80), line_width=2.5,
        )
        bone_handles.append((h_r, h_l, parent, child))

    loguru.logger.info(
        f"Viser scene built; pushing {n} frames. Open the printed URL "
        "in your browser."
    )
    for t in range(n):
        mj_data.mocap_pos[:] = qpos_list[t, :, :3]
        mj_data.mocap_quat[:] = qpos_list[t, :, 3:]
        mujoco.mj_kinematics(mj_model, mj_data)
        log_frame(
            mj_data, sim_time=t / target_fps,
            viewer_body_entity_and_ids=body_entity_and_ids,
            show_ui=True, playback_fps=target_fps,
        )

    def _update_hand(frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, n - 1))
        for i, h in enumerate(rh_handles):
            h.position = tuple(rh_joints[frame_idx, i])
        for i, h in enumerate(lh_handles):
            h.position = tuple(lh_joints[frame_idx, i])
        for h_r, h_l, parent, child in bone_handles:
            h_r.positions = np.stack([
                rh_joints[frame_idx, parent], rh_joints[frame_idx, child]
            ])
            h_l.positions = np.stack([
                lh_joints[frame_idx, parent], lh_joints[frame_idx, child]
            ])

    if _STATE.playback_slider is not None:
        @_STATE.playback_slider.on_update
        def _(_):
            _update_hand(int(_STATE.playback_slider.value))

    loguru.logger.info("Press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
