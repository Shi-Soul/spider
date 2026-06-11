#!/usr/bin/env python3
"""Batch evaluation of G1 WBC methods across motions, checkpoints, and MPC modes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SPIDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIDER_ROOT))

from spider.tasks.g1_wbc.mpc import G1WbcMpcConfig
from spider.tasks.g1_wbc.rollout import WbcRolloutConfig

TRACKING_BFM_PYTHON = str(SPIDER_ROOT.parent / "tracking_bfm" / ".venv" / "bin" / "python")

DEFAULT_DATASETS = [
    "/home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz",
    "/home/bai/ARC/Dataset/TeleAI-MoCap-Hangzhou/G1-29dof-BYDnpz-50fps-segmented_2k/mocap2_interp10",
]

DEFAULT_METHODS = [
    "no_mpc",
    "g1_wbc_joint",
    "g1_wbc_joint_global",
    "g1_wbc_ee",
]
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
    nconmax_per_env: int = WbcRolloutConfig.nconmax_per_env,
    njmax_per_env: int = WbcRolloutConfig.njmax_per_env,
    mpc_samples: int | None = None,
    mpc_iterations: int | None = None,
    mpc_planning_horizon_steps: int | None = None,
    mpc_control_steps: int | None = None,
    mpc_sampling_mode: str | None = None,
    mpc_knot_count: int | None = None,
    mpc_elite_frac: float | None = None,
    mpc_temperature: float | None = None,
    mpc_command_reg_weight: float | None = None,
    mpc_command_smooth_weight: float | None = None,
    mpc_acceptance_gate: bool | None = None,
    mpc_root_pos_sigma: float | None = None,
    mpc_root_rot_sigma: float | None = None,
    mpc_joint_sigma: float | None = None,
    mpc_smooth_passes: int | None = None,
    mpc_seed: int | None = None,
    mpc_preset: str = "aggressive",
    saved_qpos: Path | None = None,
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
        "--nconmax-per-env", str(nconmax_per_env),
        "--njmax-per-env", str(njmax_per_env),
    ]
    if method == "static_qpos":
        if saved_qpos is None:
            raise ValueError("static_qpos requires saved_qpos.")
        cmd += ["--saved-qpos", str(saved_qpos)]
    elif method != "no_mpc":
        cmd += ["--mpc-preset", mpc_preset]
        optional_args = {
            "--mpc-samples": mpc_samples,
            "--mpc-iterations": mpc_iterations,
            "--mpc-planning-horizon-steps": mpc_planning_horizon_steps,
            "--mpc-control-steps": mpc_control_steps,
            "--mpc-sampling-mode": mpc_sampling_mode,
            "--mpc-knot-count": mpc_knot_count,
            "--mpc-elite-frac": mpc_elite_frac,
            "--mpc-temperature": mpc_temperature,
            "--mpc-command-reg-weight": mpc_command_reg_weight,
            "--mpc-command-smooth-weight": mpc_command_smooth_weight,
            "--mpc-acceptance-gate": mpc_acceptance_gate,
            "--mpc-root-pos-sigma": mpc_root_pos_sigma,
            "--mpc-root-rot-sigma": mpc_root_rot_sigma,
            "--mpc-joint-sigma": mpc_joint_sigma,
            "--mpc-smooth-passes": mpc_smooth_passes,
            "--seed": mpc_seed,
        }
        for flag, value in optional_args.items():
            if value is not None:
                if isinstance(value, bool):
                    cmd.append(flag if value else f"--no-{flag[2:]}")
                else:
                    cmd += [flag, str(value)]

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

    # Extract the last complete JSON object from stdout.
    lines = proc.stdout.splitlines()
    json_blocks: list[list[str]] = []
    json_lines: list[str] = []
    in_json = False
    brace_count = 0
    for line in lines:
        stripped = line.strip()
        if not in_json:
            if stripped.startswith("{"):
                in_json = True
                brace_count = stripped.count("{") - stripped.count("}")
                json_lines.append(stripped)
                if brace_count <= 0:
                    json_blocks.append(json_lines)
                    json_lines = []
                    in_json = False
        else:
            json_lines.append(stripped)
            brace_count += stripped.count("{") - stripped.count("}")
            if brace_count <= 0:
                json_blocks.append(json_lines)
                json_lines = []
                in_json = False
    for block in reversed(json_blocks):
        try:
            return json.loads("\n".join(block))
        except json.JSONDecodeError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKPTS)
    parser.add_argument("--limit", type=int, default=None, help="Max motions per dataset.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--nconmax-per-env", type=int, default=WbcRolloutConfig.nconmax_per_env)
    parser.add_argument("--njmax-per-env", type=int, default=WbcRolloutConfig.njmax_per_env)
    parser.add_argument(
        "--mpc-preset",
        default="aggressive",
        choices=("aggressive", "conservative", "explore", "rootrot", "wide"),
    )
    parser.add_argument("--mpc-samples", type=int, default=None)
    parser.add_argument("--mpc-iterations", type=int, default=None)
    parser.add_argument("--mpc-planning-horizon-steps", type=int, default=None)
    parser.add_argument("--mpc-control-steps", type=int, default=None)
    parser.add_argument("--mpc-sampling-mode", choices=("full", "knot"), default=None)
    parser.add_argument("--mpc-knot-count", type=int, default=None)
    parser.add_argument("--mpc-elite-frac", type=float, default=None)
    parser.add_argument("--mpc-temperature", type=float, default=None)
    parser.add_argument(
        "--mpc-command-reg-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mpc-command-smooth-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mpc-acceptance-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--mpc-root-pos-sigma", type=float, default=None)
    parser.add_argument("--mpc-root-rot-sigma", type=float, default=None)
    parser.add_argument("--mpc-joint-sigma", type=float, default=None)
    parser.add_argument("--mpc-smooth-passes", type=int, default=None)
    parser.add_argument("--mpc-seed", type=int, default=None)
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
                    nconmax_per_env=args.nconmax_per_env,
                    njmax_per_env=args.njmax_per_env,
                    mpc_samples=args.mpc_samples,
                    mpc_iterations=args.mpc_iterations,
                    mpc_planning_horizon_steps=args.mpc_planning_horizon_steps,
                    mpc_control_steps=args.mpc_control_steps,
                    mpc_sampling_mode=args.mpc_sampling_mode,
                    mpc_knot_count=args.mpc_knot_count,
                    mpc_elite_frac=args.mpc_elite_frac,
                    mpc_temperature=args.mpc_temperature,
                    mpc_command_reg_weight=args.mpc_command_reg_weight,
                    mpc_command_smooth_weight=args.mpc_command_smooth_weight,
                    mpc_acceptance_gate=args.mpc_acceptance_gate,
                    mpc_root_pos_sigma=args.mpc_root_pos_sigma,
                    mpc_root_rot_sigma=args.mpc_root_rot_sigma,
                    mpc_joint_sigma=args.mpc_joint_sigma,
                    mpc_smooth_passes=args.mpc_smooth_passes,
                    mpc_seed=args.mpc_seed,
                    mpc_preset=args.mpc_preset,
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
                    "mpc": _compact_mpc_payload(payload.get("mpc")),
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


def _compact_mpc_payload(payload: dict | None) -> dict | None:
    if not payload:
        return None
    keys = (
        "preset",
        "accepted",
        "final_candidate_score",
        "final_baseline_score",
        "final_scores_max",
        "num_samples",
        "num_iterations",
        "planning_horizon_steps",
        "control_steps",
        "sampling_mode",
        "knot_count",
        "num_windows",
        "accepted_windows",
        "used_baseline_fallback",
        "acceptance_gate",
        "root_pos_sigma",
        "root_rot_sigma",
        "joint_sigma",
        "min_root_pos_sigma",
        "min_root_rot_sigma",
        "min_joint_sigma",
        "sigma_decay",
        "smooth_passes",
        "command_reg_weight",
        "command_smooth_weight",
    )
    return {
        key: payload[key]
        for key in keys
        if key in payload and isinstance(payload[key], (int, float, bool, str))
    }


if __name__ == "__main__":
    main()
