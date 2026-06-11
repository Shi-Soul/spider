#!/usr/bin/env python3
"""G1 WBC visualization: reference ghost vs executed robot(s).

Modes:
  --method no_mpc         : reference + no-MPC rollout (2 panels)
  --method g1_wbc_joint   : reference + MPC joint rollout (2 panels)
  --method compare        : reference + no-MPC + MPC joint (3 panels)

Usage (from tracking_bfm venv):
  python scripts/visualize_g1_wbc.py --motion MOTION.npz --method compare
"""

from __future__ import annotations

import argparse, sys, cv2
from pathlib import Path

import mujoco, numpy as np, torch

SPIDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIDER_ROOT))

from spider.tasks.g1_wbc.motion import load_motion
from spider.tasks.g1_wbc.policy import load_wbc_actor
from spider.tasks.g1_wbc.rollout import WbcRolloutConfig, load_wbc_model, run_no_mpc_rollout
from spider.tasks.g1_wbc.mpc import G1WbcMpcConfig, optimize_mpc_command


def render_panel(renderer, model, data, label):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=0)
    img = renderer.render()
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--motion-type", default="isaaclab")
    p.add_argument("--checkpoint", default="bc")
    p.add_argument("--method", default="compare",
                   choices=("no_mpc", "g1_wbc_joint", "g1_wbc_ee", "compare"))
    p.add_argument("--compare-with", default="g1_wbc_joint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--output", default=None)
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--mpc-samples", type=int, default=32)
    p.add_argument("--mpc-iterations", type=int, default=3)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()

    device = torch.device(args.device)
    motion_path = Path(args.motion).expanduser().resolve()
    motion = load_motion(motion_path, motion_type=args.motion_type, device=device)
    actor = load_wbc_actor(args.checkpoint, device=device)
    cfg = WbcRolloutConfig(device=str(device), max_steps=args.max_steps)

    def run_method(method):
        if method == "no_mpc":
            r = run_no_mpc_rollout(motion, actor, cfg)
            return r.qpos[:, 0].cpu().numpy(), method
        mpc = G1WbcMpcConfig(mode=method, num_samples=args.mpc_samples,
                              num_iterations=args.mpc_iterations)
        r = optimize_mpc_command(motion, actor, cfg, mpc)
        return r.rollout.qpos[:, 0].cpu().numpy(), method

    if args.method == "compare":
        sims = [run_method("no_mpc"), run_method(args.compare_with)]
    else:
        sims = [run_method(args.method)]

    model = load_wbc_model(cfg.model_path)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    ref = motion.qpos().cpu().numpy()
    n = min(ref.shape[0], *(s.shape[0] for s, _ in sims))

    print(f"Rendering {n} frames, {1+len(sims)} panels ...")
    frames = []
    d_ref = mujoco.MjData(model)
    d_sim = mujoco.MjData(model)
    for t in range(n):
        panels = []
        d_ref.qpos[:] = ref[t]
        panels.append(render_panel(renderer, model, d_ref, "reference"))
        for sim_qpos, label in sims:
            d_sim.qpos[:] = sim_qpos[t]
            panels.append(render_panel(renderer, model, d_sim, label))
        frames.append(np.concatenate(panels, axis=1))
        if (t + 1) % 50 == 0:
            print(f"  {t + 1}/{n}")

    if args.output:
        out = Path(args.output).expanduser()
    else:
        stem = motion_path.stem if motion_path.stem != "motion" else motion_path.parent.name
        lbl = args.method if args.method != "compare" else "compare"
        out = SPIDER_ROOT / "videos" / f"{stem}_{args.checkpoint}_{lbl}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
