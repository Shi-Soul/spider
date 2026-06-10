"""CLI for evaluating G1 WBC policy rollouts on a single motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spider.tasks.g1_wbc.metrics import compute_rollout_metrics
from spider.tasks.g1_wbc.motion import load_motion, validate_motion_dims
from spider.tasks.g1_wbc.policy import load_wbc_actor, resolve_checkpoint_path
from spider.tasks.g1_wbc.rollout import RolloutResult, WbcRolloutConfig, run_no_mpc_rollout


def main() -> None:
    args = _parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")

    motion = load_motion(args.motion, motion_type=args.motion_type, device=device)
    validate_motion_dims(motion)
    actor = load_wbc_actor(args.checkpoint, device=device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if args.method != "no_mpc":
        raise NotImplementedError("Only no_mpc rollout is implemented in this batch.")

    config = WbcRolloutConfig(
        model_path=args.model_path,
        device=device,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        ref_offset=args.ref_offset,
        nconmax_per_env=args.nconmax_per_env,
        njmax_per_env=args.njmax_per_env,
        sync_after_step=args.sync_after_step,
    )
    rollout = run_no_mpc_rollout(motion, actor, config)
    metrics = compute_rollout_metrics(motion, rollout)

    payload = {
        "method": args.method,
        "motion": str(Path(args.motion).expanduser().resolve()),
        "motion_type": motion.motion_type,
        "checkpoint": str(checkpoint_path),
        "device": device,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "ref_offset": args.ref_offset,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.save_rollout:
            _save_rollout(output_dir / "rollout.npz", rollout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="Reference motion npz path.")
    parser.add_argument(
        "--motion-type",
        default="auto",
        choices=("auto", "mujoco", "isaaclab"),
        help="Input npz semantic ordering.",
    )
    parser.add_argument(
        "--checkpoint",
        default="bc",
        help="WXY checkpoint alias ('bc'/'bcrl'), checkpoint directory, or .pt file.",
    )
    parser.add_argument(
        "--method",
        default="no_mpc",
        choices=("no_mpc",),
        help="Evaluation method to run.",
    )
    parser.add_argument(
        "--model-path",
        default=str(WbcRolloutConfig.model_path),
        help="MuJoCo XML path for G1 WBC simulation.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch/Warp device.")
    parser.add_argument("--num-envs", type=int, default=1, help="Batched worlds.")
    parser.add_argument("--max-steps", type=int, default=None, help="Policy steps.")
    parser.add_argument(
        "--ref-offset",
        type=int,
        default=0,
        help="Reference frame offset used when constructing policy command.",
    )
    parser.add_argument("--nconmax-per-env", type=int, default=96)
    parser.add_argument("--njmax-per-env", type=int, default=320)
    parser.add_argument(
        "--sync-after-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize Warp before reading tensors.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional result directory.")
    parser.add_argument(
        "--save-rollout",
        action="store_true",
        help="Save rollout tensors to rollout.npz when output-dir is set.",
    )
    return parser.parse_args()


def _save_rollout(path: Path, rollout: RolloutResult) -> None:
    arrays = {
        "qpos": _cpu_np(rollout.qpos),
        "qvel": _cpu_np(rollout.qvel),
        "body_pos_w": _cpu_np(rollout.body_pos_w),
        "body_quat_w": _cpu_np(rollout.body_quat_w),
        "body_lin_vel_w": _cpu_np(rollout.body_lin_vel_w),
        "body_ang_vel_w": _cpu_np(rollout.body_ang_vel_w),
        "actions": _cpu_np(rollout.actions),
        "controls": _cpu_np(rollout.controls),
        "contact_indicator": _cpu_np(rollout.contact_indicator),
        "contact_force": _cpu_np(rollout.contact_force),
        "ref_indices": _cpu_np(rollout.ref_indices),
        "dt": np.array(rollout.dt, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)


def _cpu_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


if __name__ == "__main__":
    main()
