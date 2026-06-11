"""CLI for evaluating G1 WBC policy rollouts on a single motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spider.tasks.g1_wbc.metrics import compute_rollout_metrics
from spider.tasks.g1_wbc.mpc import G1WbcMpcConfig, mpc_config_from_preset, optimize_mpc_command
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

    config = WbcRolloutConfig(
        model_path=args.model_path,
        device=device,
        num_envs=args.num_envs if args.method == "no_mpc" else 1,
        max_steps=args.max_steps,
        ref_offset=args.ref_offset,
        nconmax_per_env=args.nconmax_per_env,
        njmax_per_env=args.njmax_per_env,
        sync_after_step=args.sync_after_step,
        forward_after_step=args.forward_after_step,
        use_cuda_graph=args.use_cuda_graph,
    )
    mpc_payload = None
    mpc_result = None
    if args.method == "no_mpc":
        assert actor is not None
        rollout = run_no_mpc_rollout(motion, actor, config)
    else:
        mpc_config = _build_mpc_config(args)
        mpc_result = optimize_mpc_command(motion, actor, config, mpc_config)
        rollout = mpc_result.rollout
        mpc_payload = {
            "preset": args.mpc_preset,
            "num_samples": mpc_config.num_samples,
            "num_iterations": mpc_config.num_iterations,
            "elite_frac": mpc_config.elite_frac,
            "temperature": mpc_config.temperature,
            "root_pos_sigma": mpc_config.root_pos_sigma,
            "root_rot_sigma": mpc_config.root_rot_sigma,
            "joint_sigma": mpc_config.joint_sigma,
            "sigma_decay": mpc_config.sigma_decay,
            "smooth_passes": mpc_config.smooth_passes,
            "command_reg_weight": mpc_config.command_reg_weight,
            "command_smooth_weight": mpc_config.command_smooth_weight,
            "guided_candidate": mpc_config.use_guided_candidate,
            "guided_root_pos_gain": mpc_config.guided_root_pos_gain,
            "guided_root_rot_gain": mpc_config.guided_root_rot_gain,
            "guided_joint_gain": mpc_config.guided_joint_gain,
            "history": [vars(item) for item in mpc_result.history],
            "final_scores_mean": _safe_tensor_stat(mpc_result.scores, "mean"),
            "final_scores_max": _safe_tensor_stat(mpc_result.scores, "max"),
            "accepted": mpc_result.accepted,
            "final_candidate_score": mpc_result.final_candidate_score,
            "final_baseline_score": mpc_result.final_baseline_score,
        }
    metrics = compute_rollout_metrics(motion, rollout)

    payload = {
        "method": args.method,
        "motion": str(Path(args.motion).expanduser().resolve()),
        "motion_type": motion.motion_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": device,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "ref_offset": args.ref_offset,
        "metrics": metrics,
    }
    if mpc_payload is not None:
        payload["mpc"] = mpc_payload
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.save_rollout:
            _save_rollout(output_dir / "rollout.npz", rollout)
            if mpc_result is not None:
                _save_mpc_result(output_dir / "mpc_command.npz", mpc_result)


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
        choices=(
            "no_mpc",
            "g1_wbc_ee",
            "g1_wbc_joint",
            "g1_wbc_joint_global",
        ),
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
    parser.add_argument(
        "--nconmax-per-env",
        type=int,
        default=WbcRolloutConfig.nconmax_per_env,
        help="Per-world MuJoCo contact buffer size.",
    )
    parser.add_argument(
        "--njmax-per-env",
        type=int,
        default=WbcRolloutConfig.njmax_per_env,
        help="Per-world MuJoCo Jacobian buffer size.",
    )
    parser.add_argument(
        "--sync-after-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize Warp before reading tensors.",
    )
    parser.add_argument(
        "--forward-after-step",
        action=argparse.BooleanOptionalAction,
        default=WbcRolloutConfig.forward_after_step,
        help="Run mj_forward after each policy step before reading derived tensors.",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=WbcRolloutConfig.use_cuda_graph,
        help="Capture MuJoCo Warp step/forward/reset in CUDA graphs when available.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional result directory.")
    parser.add_argument(
        "--save-rollout",
        action="store_true",
        help="Save rollout tensors to rollout.npz when output-dir is set.",
    )
    parser.add_argument(
        "--mpc-preset",
        default="aggressive",
        choices=("aggressive", "conservative"),
        help="Tuned MPC parameter preset. Explicit MPC flags override this.",
    )
    parser.add_argument("--mpc-samples", type=int, default=None)
    parser.add_argument("--mpc-iterations", type=int, default=None)
    parser.add_argument("--mpc-elite-frac", type=float, default=None)
    parser.add_argument("--mpc-temperature", type=float, default=None)
    parser.add_argument("--mpc-root-pos-sigma", type=float, default=None)
    parser.add_argument("--mpc-root-rot-sigma", type=float, default=None)
    parser.add_argument("--mpc-joint-sigma", type=float, default=None)
    parser.add_argument("--mpc-sigma-decay", type=float, default=None)
    parser.add_argument("--mpc-smooth-passes", type=int, default=None)
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
        "--mpc-guided-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include a no-MPC-error feedback candidate in the MPC sample batch.",
    )
    parser.add_argument(
        "--mpc-guided-root-pos-gain",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mpc-guided-root-rot-gain",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mpc-guided-joint-gain",
        type=float,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=G1WbcMpcConfig.seed)
    return parser.parse_args()


def _build_mpc_config(args: argparse.Namespace) -> G1WbcMpcConfig:
    config = mpc_config_from_preset(args.method, args.mpc_preset)
    overrides = {
        "num_samples": args.mpc_samples,
        "num_iterations": args.mpc_iterations,
        "elite_frac": args.mpc_elite_frac,
        "temperature": args.mpc_temperature,
        "root_pos_sigma": args.mpc_root_pos_sigma,
        "root_rot_sigma": args.mpc_root_rot_sigma,
        "joint_sigma": args.mpc_joint_sigma,
        "sigma_decay": args.mpc_sigma_decay,
        "smooth_passes": args.mpc_smooth_passes,
        "command_reg_weight": args.mpc_command_reg_weight,
        "command_smooth_weight": args.mpc_command_smooth_weight,
        "use_guided_candidate": args.mpc_guided_candidate,
        "guided_root_pos_gain": args.mpc_guided_root_pos_gain,
        "guided_root_rot_gain": args.mpc_guided_root_rot_gain,
        "guided_joint_gain": args.mpc_guided_joint_gain,
        "seed": args.seed,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    return config


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


def _save_mpc_result(path: Path, result) -> None:
    arrays = {
        "refined_qpos": _cpu_np(result.refined_qpos),
        "candidate_scores": _cpu_np(result.scores),
        "command_joint_pos": _cpu_np(result.command.joint_pos),
        "command_joint_vel": _cpu_np(result.command.joint_vel),
        "command_body_pos_w": _cpu_np(result.command.body_pos_w),
        "command_body_quat_w": _cpu_np(result.command.body_quat_w),
        "command_qpos_trajectory": _cpu_np(result.command.qpos_trajectory),
        "command_qvel_trajectory": _cpu_np(result.command.qvel_trajectory),
    }
    np.savez_compressed(path, **arrays)


def _cpu_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _safe_tensor_stat(value: torch.Tensor, stat: str) -> float:
    if value.numel() == 0:
        return 0.0
    value = torch.nan_to_num(value.float())
    if stat == "max":
        return float(value.max().detach().cpu().item())
    return float(value.mean().detach().cpu().item())


if __name__ == "__main__":
    main()
