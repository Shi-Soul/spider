#!/usr/bin/env python3
"""G1 WBC visualization: executed robot(s) with a reference ghost overlay.

Modes:
  --method no_mpc         : no-MPC rollout with reference ghost
  --method g1_wbc_joint   : MPC joint rollout with reference ghost
  --method g1_wbc_joint_global : MPC global-only joint rollout with reference ghost
  --method compare-all    : no-MPC + MPC joint/global/EE panels
  --method saved          : render --saved-rollout/--saved-command trajectories
  --method compare        : no-MPC + MPC joint/EE panels

Usage (from tracking_bfm venv):
  python scripts/visualize_g1_wbc.py --motion MOTION.npz --method compare
  python scripts/visualize_g1_wbc.py --motion MOTION.npz --method saved \
    --saved-rollout no_mpc:/path/to/rollout.npz \
    --saved-command mpc_joint:/path/to/mpc_command.npz
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import torch

SPIDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIDER_ROOT))

from spider.tasks.g1_wbc.motion import load_motion
from spider.tasks.g1_wbc.policy import load_wbc_actor
from spider.tasks.g1_wbc.mpc import (
    G1WbcMpcConfig,
    mpc_config_from_preset,
    optimize_mpc_command,
)
from spider.tasks.g1_wbc.rollout import (
    WbcRolloutConfig,
    load_wbc_model,
    run_no_mpc_rollout,
)


GHOST_RGBA = np.array([0.0, 0.85, 1.0, 0.45], dtype=np.float32)


def render_panel(
    renderer,
    model,
    data_sim,
    data_ref,
    label,
    camera,
    robot_geom_ids,
):
    """Render one MuJoCo scene with executed robot and reference ghost geoms."""

    mujoco.mj_forward(model, data_sim)
    mujoco.mj_forward(model, data_ref)
    if camera is None:
        renderer.update_scene(data_sim)
    else:
        renderer.update_scene(data_sim, camera=camera)

    saved_rgba = model.geom_rgba[robot_geom_ids].copy()
    try:
        model.geom_rgba[robot_geom_ids] = GHOST_RGBA
        mujoco.mjv_addGeoms(
            model,
            data_ref,
            renderer._scene_option,
            mujoco.MjvPerturb(),
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            renderer.scene,
        )
    finally:
        model.geom_rgba[robot_geom_ids] = saved_rgba

    img = renderer.render()
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(
        img,
        "cyan ghost: reference",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 245, 245),
        2,
    )
    return img


def robot_geom_ids(model):
    ids = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name in ("terrain", "floor"):
            continue
        ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def make_follow_camera(ref_qpos, sim_qpos, *, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    lookat = 0.5 * (ref_qpos[:3] + sim_qpos[:3])
    lookat[2] = max(float(lookat[2] + 0.15), 0.75)
    camera.lookat[:] = lookat
    return camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--motion-type", default="isaaclab")
    p.add_argument("--checkpoint", default="bc")
    p.add_argument("--method", default="compare",
                   choices=("no_mpc", "g1_wbc_joint",
                            "g1_wbc_joint_global", "g1_wbc_ee",
                            "compare", "compare-all", "saved"))
    p.add_argument("--compare-with", default="g1_wbc_joint",
                   choices=("g1_wbc_joint", "g1_wbc_joint_global", "g1_wbc_ee"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--nconmax-per-env", type=int, default=WbcRolloutConfig.nconmax_per_env)
    p.add_argument("--njmax-per-env", type=int, default=WbcRolloutConfig.njmax_per_env)
    p.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction,
                   default=WbcRolloutConfig.use_cuda_graph)
    p.add_argument("--forward-after-step", action=argparse.BooleanOptionalAction,
                   default=WbcRolloutConfig.forward_after_step)
    p.add_argument("--output", default=None)
    p.add_argument("--fps", type=float, default=50)
    p.add_argument("--camera-mode", choices=("follow", "fixed"), default="follow")
    p.add_argument("--camera-distance", type=float, default=4.0)
    p.add_argument("--camera-azimuth", type=float, default=135.0)
    p.add_argument("--camera-elevation", type=float, default=-18.0)
    p.add_argument("--ghost-alpha", type=float, default=float(GHOST_RGBA[3]))
    p.add_argument(
        "--mpc-preset",
        default="aggressive",
        choices=("aggressive", "conservative", "explore", "rootrot"),
    )
    p.add_argument("--mpc-samples", type=int, default=None)
    p.add_argument("--mpc-iterations", type=int, default=None)
    p.add_argument("--mpc-elite-frac", type=float, default=None)
    p.add_argument("--mpc-temperature", type=float, default=None)
    p.add_argument(
        "--mpc-command-reg-weight",
        type=float,
        default=None,
    )
    p.add_argument(
        "--mpc-command-smooth-weight",
        type=float,
        default=None,
    )
    p.add_argument("--mpc-root-pos-sigma", type=float, default=None)
    p.add_argument("--mpc-root-rot-sigma", type=float, default=None)
    p.add_argument("--mpc-joint-sigma", type=float, default=None)
    p.add_argument("--seed", type=int, default=G1WbcMpcConfig.seed)
    p.add_argument(
        "--saved-rollout",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Append qpos from a saved evaluate.py --save-rollout rollout.npz.",
    )
    p.add_argument(
        "--saved-command",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Append qpos from a saved evaluate.py MPC mpc_command.npz.",
    )
    p.add_argument(
        "--saved-env-index",
        type=int,
        default=0,
        help="Environment index to render from saved batched arrays.",
    )
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()
    GHOST_RGBA[3] = float(args.ghost_alpha)

    device = torch.device(args.device)
    motion_path = Path(args.motion).expanduser().resolve()
    motion = load_motion(motion_path, motion_type=args.motion_type, device=device)
    actor = None
    if args.method != "saved" or not (args.saved_rollout or args.saved_command):
        actor = load_wbc_actor(args.checkpoint, device=device)
    cfg = WbcRolloutConfig(
        device=str(device),
        max_steps=args.max_steps,
        nconmax_per_env=args.nconmax_per_env,
        njmax_per_env=args.njmax_per_env,
        use_cuda_graph=args.use_cuda_graph,
        forward_after_step=args.forward_after_step,
    )

    def run_method(method):
        if method == "no_mpc":
            r = run_no_mpc_rollout(motion, actor, cfg)
            return r.qpos[:, 0].cpu().numpy(), method
        mpc = mpc_config_from_preset(method, args.mpc_preset)
        overrides = {
            "num_samples": args.mpc_samples,
            "num_iterations": args.mpc_iterations,
            "elite_frac": args.mpc_elite_frac,
            "temperature": args.mpc_temperature,
            "command_reg_weight": args.mpc_command_reg_weight,
            "command_smooth_weight": args.mpc_command_smooth_weight,
            "root_pos_sigma": args.mpc_root_pos_sigma,
            "root_rot_sigma": args.mpc_root_rot_sigma,
            "joint_sigma": args.mpc_joint_sigma,
            "seed": args.seed,
        }
        for name, value in overrides.items():
            if value is not None:
                setattr(mpc, name, value)
        r = optimize_mpc_command(motion, actor, cfg, mpc)
        return r.rollout.qpos[:, 0].cpu().numpy(), method

    saved_sims = load_saved_sims(
        args.saved_rollout,
        args.saved_command,
        env_index=args.saved_env_index,
        max_steps=args.max_steps,
    )

    if args.method == "saved":
        if not saved_sims:
            raise ValueError("--method saved requires --saved-rollout or --saved-command.")
        sims = saved_sims
    elif args.method == "compare":
        sims = [run_method("no_mpc"), run_method(args.compare_with)]
    elif args.method == "compare-all":
        sims = [
            run_method("no_mpc"),
            run_method("g1_wbc_joint"),
            run_method("g1_wbc_joint_global"),
            run_method("g1_wbc_ee"),
        ]
    else:
        sims = [run_method(args.method)]
    if args.method != "saved":
        sims.extend(saved_sims)

    model = load_wbc_model(cfg.model_path)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    fixed_camera = 0 if model.ncam > 0 else None
    geom_ids = robot_geom_ids(model)

    ref = motion.qpos().cpu().numpy()
    n = min(ref.shape[0], *(s.shape[0] for s, _ in sims))

    print(f"Rendering {n} frames, {len(sims)} overlay panels ...")
    frames = []
    d_ref = mujoco.MjData(model)
    d_sim = mujoco.MjData(model)
    for t in range(n):
        panels = []
        d_ref.qpos[:] = ref[t]
        for sim_qpos, label in sims:
            d_sim.qpos[:] = sim_qpos[t]
            if args.camera_mode == "follow":
                camera = make_follow_camera(
                    ref[t],
                    sim_qpos[t],
                    distance=args.camera_distance,
                    azimuth=args.camera_azimuth,
                    elevation=args.camera_elevation,
                )
            else:
                camera = fixed_camera
            panels.append(
                render_panel(
                    renderer,
                    model,
                    d_sim,
                    d_ref,
                    label,
                    camera,
                    geom_ids,
                )
            )
        frames.append(np.concatenate(panels, axis=1))
        if (t + 1) % 50 == 0:
            print(f"  {t + 1}/{n}")

    if args.output:
        out = Path(args.output).expanduser()
    else:
        stem = motion_path.stem if motion_path.stem != "motion" else motion_path.parent.name
        lbl = f"compare_{args.compare_with}" if args.method == "compare" else args.method
        out = SPIDER_ROOT / "videos" / f"{stem}_{args.checkpoint}_{lbl}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved -> {out}")


def load_saved_sims(
    saved_rollouts,
    saved_commands,
    *,
    env_index,
    max_steps,
):
    sims = []
    for item in saved_rollouts:
        label, path = parse_labeled_path(item)
        with np.load(path) as data:
            if "qpos" not in data.files:
                raise ValueError(f"Saved rollout {path} is missing qpos.")
            qpos = select_saved_qpos(data["qpos"], env_index=env_index)
        sims.append((trim_saved_qpos(qpos, max_steps), label))
    for item in saved_commands:
        label, path = parse_labeled_path(item)
        with np.load(path) as data:
            key = "refined_qpos" if "refined_qpos" in data.files else "command_qpos_trajectory"
            if key not in data.files:
                raise ValueError(
                    f"Saved command {path} is missing refined_qpos/command_qpos_trajectory."
                )
            qpos = select_saved_qpos(data[key], env_index=env_index)
        sims.append((trim_saved_qpos(qpos, max_steps), label))
    return sims


def parse_labeled_path(value):
    if ":" in value:
        label, raw_path = value.split(":", 1)
        label = label.strip()
    else:
        raw_path = value
        label = ""
    path = Path(raw_path).expanduser().resolve()
    if not label:
        label = path.parent.name if path.name in ("rollout.npz", "mpc_command.npz") else path.stem
    return label, path


def select_saved_qpos(qpos, *, env_index):
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 3:
        if not 0 <= env_index < qpos.shape[1]:
            raise ValueError(f"saved-env-index {env_index} outside qpos shape {qpos.shape}.")
        qpos = qpos[:, env_index]
    if qpos.ndim != 2 or qpos.shape[-1] != 36:
        raise ValueError(f"Expected saved qpos shape (T,36) or (T,N,36), got {qpos.shape}.")
    return qpos


def trim_saved_qpos(qpos, max_steps):
    if max_steps is None:
        return qpos
    return qpos[: int(max_steps) + 1]


if __name__ == "__main__":
    main()
