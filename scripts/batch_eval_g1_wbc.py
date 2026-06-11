#!/usr/bin/env python3
"""Batch evaluation of G1 WBC methods across motions, checkpoints, and MPC modes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SPIDER_ROOT = Path(__file__).resolve().parents[1]
TRACKING_BFM_PYTHON = str(SPIDER_ROOT.parent / "tracking_bfm" / ".venv" / "bin" / "python")

DEFAULT_DATASETS = [
    "/home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz",
    "/home/bai/ARC/Dataset/TeleAI-MoCap-Hangzhou/G1-29dof-BYDnpz-50fps-segmented_2k/mocap2_interp10",
]

DEFAULT_METHODS = ["no_mpc", "g1_wbc_joint", "g1_wbc_ee"]
DEFAULT_CKPTS = ["bc", "bcrl"]


def find_motions(dataset_paths: list[str], limit: int | None = None) -> list[Path]:
    motions: list[Path] = []
    for ds in dataset_paths:
        ds_path = Path(ds).expanduser()
        if not ds_path.exists():
            print(f"WARNING: dataset not found: {ds}", file=sys.stderr)
            continue
        for npz in sorted(ds_path.rglob("motion.npz")):
            motions.append(npz)
            if limit is not None and len(motions) >= limit:
                return motions
    return motions


def run_eval(
    motion: Path,
    method: str,
    checkpoint: str,
    *,
    device: str = "cuda:0",
    max_steps: int = 250,
    mpc_samples: int = 32,
    mpc_iterations: int = 3,
) -> dict | None:
    cmd = [
        TRACKING_BFM_PYTHON,
        "-m", "spider.tasks.g1_wbc.evaluate",
        "--motion", str(motion),
        "--motion-type", "isaaclab",
        "--checkpoint", checkpoint,
        "--method", method,
        "--max-steps", str(max_steps),
        "--device", device,
    ]
    if method != "no_mpc":
        cmd += [
            "--mpc-samples", str(mpc_samples),
            "--mpc-iterations", str(mpc_iterations),
        ]

    proc = subprocess.run(
        cmd,
        cwd=str(SPIDER_ROOT),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        print(f"ERROR [{method}/{checkpoint}] {motion.name}: {proc.stderr[:200]}", file=sys.stderr)
        return None

    # Extract JSON from stdout (it is the last JSON block)
    lines = proc.stdout.splitlines()
    json_lines = []
    in_json = False
    brace_count = 0
    for line in lines:
        stripped = line.strip()
        if not in_json:
            if stripped.startswith("{"):
                in_json = True
                brace_count = stripped.count("{") - stripped.count("}")
                json_lines.append(stripped)
        else:
            json_lines.append(stripped)
            brace_count += stripped.count("{") - stripped.count("}")
            if brace_count <= 0:
                break
    if json_lines:
        try:
            return json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKPTS)
    parser.add_argument("--limit", type=int, default=None, help="Max motions per dataset.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--mpc-samples", type=int, default=32)
    parser.add_argument("--mpc-iterations", type=int, default=3)
    parser.add_argument("--output", default=None, help="JSON output path.")
    args = parser.parse_args()

    motions = find_motions(args.datasets, limit=args.limit)
    print(f"Found {len(motions)} motions across {len(args.datasets)} datasets")

    results: list[dict] = []
    for motion in motions:
        motion_name = str(motion.relative_to(motion.parents[2]) if len(motion.parents) > 2 else motion)
        for ckpt in args.checkpoints:
            for method in args.methods:
                print(f"[{method}/{ckpt}] {motion_name} ...", end=" ", flush=True)
                payload = run_eval(
                    motion,
                    method=method,
                    checkpoint=ckpt,
                    device=args.device,
                    max_steps=args.max_steps,
                    mpc_samples=args.mpc_samples,
                    mpc_iterations=args.mpc_iterations,
                )
                if payload is None:
                    print("FAILED")
                    continue
                metrics = payload.get("metrics", {})
                score = metrics.get("score", float("nan"))
                success = metrics.get("success", False)
                print(f"score={score:.3f} success={success}")
                results.append({
                    "motion": str(motion),
                    "method": method,
                    "checkpoint": ckpt,
                    "score": score,
                    "success": success,
                    "metrics": {k: v for k, v in metrics.items()
                                if isinstance(v, (int, float, bool, str))},
                })

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {out_path}")

    # Summary
    print("\n=== Summary ===")
    for ckpt in args.checkpoints:
        for method in args.methods:
            items = [r for r in results if r["checkpoint"] == ckpt and r["method"] == method]
            if not items:
                continue
            avg_score = sum(r["score"] for r in items) / len(items)
            n_success = sum(1 for r in items if r["success"])
            print(f"  {method}/{ckpt}: avg_score={avg_score:.3f} success={n_success}/{len(items)}")


if __name__ == "__main__":
    main()

