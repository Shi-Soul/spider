#!/usr/bin/env python3
"""Run low-sample G1 WBC joint-global sweeps on testbed motions."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence

SPIDER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SPIDER_ROOT.parent
TESTBED_PACKAGE_ROOT = WORKSPACE_ROOT / "g1_wbc_testbed_motion_package_20260617"
DEFAULT_REWARD_WEIGHTS = (
    TESTBED_PACKAGE_ROOT
    / "metadata"
    / "g1_wbc_reward_weights_method_specific_v14_20260612.json"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/g1_wbc_low_sample_sweep")
DEFAULT_PYTHON_EXECUTABLE = (
    SPIDER_ROOT / ".venv" / "bin" / "python"
    if (SPIDER_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)

DEFAULT_SAMPLES = ("64", "128", "256")
DEFAULT_SEEDS = ("0", "1", "2")
DEFAULT_MOTIONS = ("walk", "jump", "qixing")
REFERENCE_METRICS = (
    "score",
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
    "ee_local_pos_error_mean",
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)

BASE_COLUMNS = [
    "motion_name",
    "samples",
    "iterations",
    "horizon",
    "control",
    "knot_count",
    "root_pos_sigma",
    "root_rot_sigma",
    "joint_sigma",
    "seed",
    "status",
    "exit_code",
    "duration_sec",
    "output_dir",
    "metrics_path",
    "reference_metrics_path",
    "motion",
    "method",
    "checkpoint",
    "device",
    "max_steps",
    "command",
    "error",
]

METRIC_COLUMNS = [
    "metric_num_steps",
    "metric_score",
    "metric_success",
    "metric_root_pos_error_mean",
    "metric_root_pos_error_max",
    "metric_root_rot_error_mean",
    "metric_joint_pos_error_mean",
    "metric_joint_vel_error_mean",
    "metric_body_global_pos_error_mean",
    "metric_body_global_rot_error_mean",
    "metric_body_local_pos_error_mean",
    "metric_body_local_rot_error_mean",
    "metric_ee_global_pos_error_mean",
    "metric_ee_global_rot_error_mean",
    "metric_ee_local_pos_error_mean",
    "metric_ee_local_rot_error_mean",
    "metric_hand_global_pos_error_mean",
    "metric_hand_global_rot_error_mean",
    "metric_hand_local_pos_error_mean",
    "metric_hand_local_rot_error_mean",
    "metric_contact_mismatch_rate",
    "metric_contact_false_positive_rate",
    "metric_contact_false_negative_rate",
    "metric_contact_switch_rate",
    "metric_reference_contact_switch_rate",
    "metric_contact_force_active_mean",
    "metric_contact_force_peak",
    "metric_contact_force_excess_mean",
    "metric_contact_force_delta_mean",
    "metric_bad_floor_contact_rate",
    "metric_bad_floor_force_mean",
    "metric_bad_floor_force_excess_mean",
    "metric_action_delta_mean",
    "metric_control_delta_mean",
    "metric_joint_acc_mean",
    "metric_joint_jerk_mean",
]

MPC_COLUMNS = [
    "mpc_preset",
    "mpc_num_samples",
    "mpc_num_iterations",
    "mpc_planning_horizon_steps",
    "mpc_control_steps",
    "mpc_sampling_mode",
    "mpc_knot_count",
    "mpc_use_warm_start",
    "mpc_warm_start_source",
    "mpc_warm_start_decay",
    "mpc_elite_frac",
    "mpc_temperature",
    "mpc_root_pos_sigma",
    "mpc_root_rot_sigma",
    "mpc_joint_sigma",
    "mpc_min_root_pos_sigma",
    "mpc_min_root_rot_sigma",
    "mpc_min_joint_sigma",
    "mpc_sigma_decay",
    "mpc_smooth_passes",
    "mpc_command_reg_weight",
    "mpc_command_smooth_weight",
    "mpc_reward_weight_source",
    "mpc_acceptance_gate",
    "mpc_guided_candidate",
    "mpc_final_scores_mean",
    "mpc_final_scores_max",
    "mpc_accepted",
    "mpc_used_baseline_fallback",
    "mpc_num_windows",
    "mpc_accepted_windows",
    "mpc_final_candidate_score",
    "mpc_final_baseline_score",
]

REFERENCE_COLUMNS = [
    value
    for metric in REFERENCE_METRICS
    for value in (f"ref_{metric}", f"delta_{metric}")
]


class MotionSpec(NamedTuple):
    name: str
    path: Path
    reference_metrics_path: Path


class SweepJob(NamedTuple):
    motion_name: str
    motion_path: Path
    reference_metrics_path: Path
    samples: int
    seed: int
    output_dir: Path
    iterations: int = 2
    horizon: int = 80
    control: int = 20
    knot_count: int = 8
    root_pos_sigma: float = 0.08
    root_rot_sigma: float = 0.18
    joint_sigma: float = 0.28


def parse_int_list(values: Sequence[str], flag_name: str) -> list[int]:
    """Parse argparse values that may be space-separated, comma-separated, or both."""

    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"{flag_name} expects integers, got {part!r}."
                ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{flag_name} needs at least one integer.")
    return parsed


def build_jobs(
    *,
    motions: Sequence[MotionSpec],
    samples: Sequence[int],
    seeds: Sequence[int],
    output_root: Path,
    iterations: Sequence[int] = (2,),
    horizons: Sequence[int] = (80,),
    controls: Sequence[int] = (20,),
    knot_counts: Sequence[int] = (8,),
    sigma_triplets: Sequence[tuple[float, float, float]] = ((0.08, 0.18, 0.28),),
) -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for motion in motions:
        for sample_count in samples:
            for iteration_count in iterations:
                for horizon in horizons:
                    for control in controls:
                        for knot_count in knot_counts:
                            for root_pos_sigma, root_rot_sigma, joint_sigma in sigma_triplets:
                                for seed in seeds:
                                    run_id = (
                                        f"{motion.name}_s{sample_count}_i{iteration_count}_"
                                        f"h{horizon}_c{control}_k{knot_count}_"
                                        f"sig{_compact_float(root_pos_sigma)}_"
                                        f"{_compact_float(root_rot_sigma)}_"
                                        f"{_compact_float(joint_sigma)}_seed{seed}"
                                    )
                                    jobs.append(
                                        SweepJob(
                                            motion_name=motion.name,
                                            motion_path=motion.path,
                                            reference_metrics_path=motion.reference_metrics_path,
                                            samples=sample_count,
                                            seed=seed,
                                            output_dir=output_root / motion.name / run_id,
                                            iterations=iteration_count,
                                            horizon=horizon,
                                            control=control,
                                            knot_count=knot_count,
                                            root_pos_sigma=root_pos_sigma,
                                            root_rot_sigma=root_rot_sigma,
                                            joint_sigma=joint_sigma,
                                        )
                                    )
    return jobs


def build_eval_command(
    job: SweepJob,
    *,
    python_executable: str,
    reward_weights: Path,
    checkpoint: str,
    max_steps: int,
    device: str,
    nconmax_per_env: int = 512,
    njmax_per_env: int = 2048,
    warm_start: bool = False,
    warm_start_source: str = "best",
    warm_start_decay: float = 1.0,
    command_reg_weight: float = 0.0,
    command_smooth_weight: float = 0.0,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "spider.tasks.g1_wbc.evaluate",
        "--motion",
        str(job.motion_path),
        "--motion-type",
        "isaaclab",
        "--checkpoint",
        checkpoint,
        "--method",
        "g1_wbc_joint_global",
        "--max-steps",
        str(max_steps),
        "--device",
        device,
        "--output-dir",
        str(job.output_dir),
        "--mpc-preset",
        "aggressive",
        "--mpc-sampling-mode",
        "knot",
        "--mpc-knot-count",
        str(job.knot_count),
        "--mpc-samples",
        str(job.samples),
        "--mpc-iterations",
        str(job.iterations),
        "--mpc-planning-horizon-steps",
        str(job.horizon),
        "--mpc-control-steps",
        str(job.control),
        "--mpc-temperature",
        "0.7",
        "--mpc-root-pos-sigma",
        str(job.root_pos_sigma),
        "--mpc-root-rot-sigma",
        str(job.root_rot_sigma),
        "--mpc-joint-sigma",
        str(job.joint_sigma),
        "--mpc-smooth-passes",
        "0",
        "--mpc-command-reg-weight",
        str(command_reg_weight),
        "--mpc-command-smooth-weight",
        str(command_smooth_weight),
        "--mpc-guided-root-pos-gain",
        "0.50",
        "--mpc-guided-root-rot-gain",
        "0.50",
        "--mpc-guided-joint-gain",
        "0.50",
        "--mpc-guided-root-pos-clip",
        "0.05",
        "--mpc-guided-root-rot-clip",
        "0.12",
        "--mpc-guided-joint-clip",
        "0.35",
        "--mpc-guided-candidate",
        "--mpc-acceptance-gate",
        "--mpc-reward-weights",
        str(reward_weights),
        "--nconmax-per-env",
        str(nconmax_per_env),
        "--njmax-per-env",
        str(njmax_per_env),
        "--seed",
        str(job.seed),
    ]
    if warm_start:
        command.extend(
            [
                "--mpc-warm-start",
                "--mpc-warm-start-source",
                warm_start_source,
                "--mpc-warm-start-decay",
                str(warm_start_decay),
            ]
        )
    return command


def row_from_metrics(
    job: SweepJob,
    *,
    metrics_path: Path,
    status: str,
    exit_code: int,
    duration_sec: float,
    command: Sequence[str],
    error: str,
) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text())
    row = _base_row(
        job,
        metrics_path=metrics_path,
        status=status,
        exit_code=exit_code,
        duration_sec=duration_sec,
        command=command,
        error=error,
    )
    for key in ("motion", "method", "checkpoint", "device", "max_steps"):
        row[key] = payload.get(key, "")
    for key, value in payload.get("metrics", {}).items():
        if _is_scalar(value):
            row[f"metric_{key}"] = value
    _add_reference_columns(row, payload.get("metrics", {}), job.reference_metrics_path)
    for key, value in payload.get("mpc", {}).items():
        if _is_scalar(value):
            row[f"mpc_{key}"] = value
    return row


def write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _summary_fieldnames(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_job(
    job: SweepJob,
    *,
    command: Sequence[str],
) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    proc = subprocess.run(
        list(command),
        cwd=str(SPIDER_ROOT),
        capture_output=True,
        text=True,
    )
    duration_sec = time.monotonic() - start
    (job.output_dir / "stdout.txt").write_text(proc.stdout)
    (job.output_dir / "stderr.txt").write_text(proc.stderr)

    metrics_path = job.output_dir / "metrics.json"
    error = _tail(proc.stderr) if proc.returncode else ""
    if proc.returncode != 0:
        return _base_row(
            job,
            metrics_path=metrics_path,
            status="failed",
            exit_code=proc.returncode,
            duration_sec=duration_sec,
            command=command,
            error=error,
        )
    if not metrics_path.exists():
        return _base_row(
            job,
            metrics_path=metrics_path,
            status="missing_metrics",
            exit_code=proc.returncode,
            duration_sec=duration_sec,
            command=command,
            error="evaluate.py finished but metrics.json was not written",
        )
    try:
        return row_from_metrics(
            job,
            metrics_path=metrics_path,
            status="ok",
            exit_code=proc.returncode,
            duration_sec=duration_sec,
            command=command,
            error="",
        )
    except (OSError, json.JSONDecodeError) as exc:
        return _base_row(
            job,
            metrics_path=metrics_path,
            status="invalid_metrics",
            exit_code=proc.returncode,
            duration_sec=duration_sec,
            command=command,
            error=str(exc),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Print planned evaluate.py commands without running them. This is the default.",
    )
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Run evaluate.py jobs and write the CSV summary.",
    )
    parser.set_defaults(execute=False)

    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON_EXECUTABLE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--iterations", nargs="+", default=["2"])
    parser.add_argument("--horizons", nargs="+", default=["80"])
    parser.add_argument("--controls", nargs="+", default=["20"])
    parser.add_argument("--knot-counts", nargs="+", default=["8"])
    parser.add_argument(
        "--sigma-triplets",
        nargs="+",
        default=["0.08,0.18,0.28"],
        help="Space-separated root_pos,root_rot,joint sigma triplets.",
    )
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--motions",
        nargs="+",
        default=list(DEFAULT_MOTIONS),
        help="Motion names walk/jump/qixing or explicit motion.npz paths.",
    )
    parser.add_argument(
        "--motion",
        default=None,
        help="Legacy single-motion alias. Overrides --motions when set.",
    )
    parser.add_argument("--checkpoint", default="bc")
    parser.add_argument("--mpc-reward-weights", default=str(DEFAULT_REWARD_WEIGHTS))
    parser.add_argument(
        "--mpc-warm-start",
        action="store_true",
        help="Enable shifted receding-plan warm start in evaluate.py.",
    )
    parser.add_argument(
        "--mpc-warm-start-source",
        choices=("best", "mean"),
        default="best",
    )
    parser.add_argument("--mpc-warm-start-decay", type=float, default=1.0)
    parser.add_argument("--mpc-command-reg-weight", type=float, default=0.0)
    parser.add_argument("--mpc-command-smooth-weight", type=float, default=0.0)
    parser.add_argument("--nconmax-per-env", type=int, default=512)
    parser.add_argument("--njmax-per-env", type=int, default=2048)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        samples = parse_int_list(args.samples, "--samples")
        iterations = parse_int_list(args.iterations, "--iterations")
        horizons = parse_int_list(args.horizons, "--horizons")
        controls = parse_int_list(args.controls, "--controls")
        knot_counts = parse_int_list(args.knot_counts, "--knot-counts")
        sigma_triplets = parse_sigma_triplets(args.sigma_triplets)
        seeds = parse_int_list(args.seeds, "--seeds")
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    _validate_positive(samples, "--samples")
    _validate_positive(iterations, "--iterations")
    _validate_positive(horizons, "--horizons")
    _validate_positive(controls, "--controls")
    _validate_positive(knot_counts, "--knot-counts")
    _validate_nonnegative(seeds, "--seeds")

    motion_values = [args.motion] if args.motion is not None else args.motions
    motions = resolve_motion_specs(motion_values)
    reward_weights = Path(args.mpc_reward_weights).expanduser()
    output_root = Path(args.output_root).expanduser()
    summary_path = output_root / "summary.csv"
    jobs = build_jobs(
        motions=motions,
        samples=samples,
        iterations=iterations,
        horizons=horizons,
        controls=controls,
        knot_counts=knot_counts,
        sigma_triplets=sigma_triplets,
        seeds=seeds,
        output_root=output_root,
    )
    commands = [
        build_eval_command(
            job,
            python_executable=args.python_executable,
            reward_weights=reward_weights,
            checkpoint=args.checkpoint,
            max_steps=args.max_steps,
            device=args.device,
            nconmax_per_env=args.nconmax_per_env,
            njmax_per_env=args.njmax_per_env,
            warm_start=args.mpc_warm_start,
            warm_start_source=args.mpc_warm_start_source,
            warm_start_decay=args.mpc_warm_start_decay,
            command_reg_weight=args.mpc_command_reg_weight,
            command_smooth_weight=args.mpc_command_smooth_weight,
        )
        for job in jobs
    ]

    if not args.execute:
        _print_dry_run(jobs, commands, summary_path)
        return 0

    for motion in motions:
        _require_file(motion.path, f"--motions {motion.name}")
    _require_file(reward_weights, "--mpc-reward-weights")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, (job, command) in enumerate(zip(jobs, commands), start=1):
        print(
            f"[{index}/{len(jobs)}] motion={job.motion_name} "
            f"samples={job.samples} seed={job.seed}"
        )
        row = run_job(job, command=command)
        rows.append(row)
        score = row.get("metric_score", "")
        success = row.get("metric_success", "")
        print(f"  status={row['status']} score={score} success={success}")

    write_summary_csv(rows, summary_path)
    print(f"Wrote {len(rows)} rows to {summary_path}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


def _base_row(
    job: SweepJob,
    *,
    metrics_path: Path,
    status: str,
    exit_code: int,
    duration_sec: float,
    command: Sequence[str],
    error: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "motion_name": job.motion_name,
        "samples": job.samples,
        "iterations": job.iterations,
        "horizon": job.horizon,
        "control": job.control,
        "knot_count": job.knot_count,
        "root_pos_sigma": job.root_pos_sigma,
        "root_rot_sigma": job.root_rot_sigma,
        "joint_sigma": job.joint_sigma,
        "seed": job.seed,
        "status": status,
        "exit_code": exit_code,
        "duration_sec": round(duration_sec, 3),
        "output_dir": str(job.output_dir),
        "metrics_path": str(metrics_path),
        "reference_metrics_path": str(job.reference_metrics_path),
        "motion": "",
        "method": "g1_wbc_joint_global",
        "checkpoint": "",
        "device": "",
        "max_steps": "",
        "command": shlex.join(list(command)),
        "error": error,
    }
    _add_reference_columns(row, {}, job.reference_metrics_path)
    return row


def _summary_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    known = BASE_COLUMNS + METRIC_COLUMNS + REFERENCE_COLUMNS + MPC_COLUMNS
    extras = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in known
            and (
                key.startswith("metric_")
                or key.startswith("mpc_")
                or key.startswith("ref_")
                or key.startswith("delta_")
            )
        }
    )
    return known + extras


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _tail(text: str, *, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _validate_positive(values: Sequence[int], flag_name: str) -> None:
    for value in values:
        if value <= 0:
            raise SystemExit(f"{flag_name} values must be positive, got {value}.")


def _validate_nonnegative(values: Sequence[int], flag_name: str) -> None:
    for value in values:
        if value < 0:
            raise SystemExit(f"{flag_name} values must be non-negative, got {value}.")


def parse_sigma_triplets(values: Sequence[str]) -> list[tuple[float, float, float]]:
    """Parse root_pos,root_rot,joint sigma triplets."""

    triplets: list[tuple[float, float, float]] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"--sigma-triplets expects comma triplets, got {value!r}."
            )
        try:
            triplets.append(tuple(float(part) for part in parts))  # type: ignore[arg-type]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--sigma-triplets expects floats, got {value!r}."
            ) from exc
    return triplets


def resolve_motion_specs(values: Sequence[str]) -> list[MotionSpec]:
    specs: list[MotionSpec] = []
    for value in values:
        path = Path(value).expanduser()
        if path.suffix == ".npz" or "/" in value:
            motion_path = path.resolve()
            name = (
                motion_path.parent.name
                if motion_path.name == "motion.npz"
                else motion_path.stem
            )
        else:
            name = value
            motion_path = TESTBED_PACKAGE_ROOT / "input_motions" / name / "motion.npz"
        reference_path = (
            TESTBED_PACKAGE_ROOT
            / "outputs"
            / "spider"
            / name
            / "g1_wbc_joint_global"
            / "metrics.json"
        )
        specs.append(
            MotionSpec(
                name=_safe_name(name),
                path=motion_path,
                reference_metrics_path=reference_path,
            )
        )
    return specs


def _add_reference_columns(
    row: dict[str, Any],
    metrics: Any,
    reference_metrics_path: Path,
) -> None:
    for metric in REFERENCE_METRICS:
        row[f"ref_{metric}"] = ""
        row[f"delta_{metric}"] = ""
    if not isinstance(metrics, dict) or not reference_metrics_path.exists():
        return
    try:
        reference_payload = json.loads(reference_metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    reference_metrics = reference_payload.get("metrics", {})
    if not isinstance(reference_metrics, dict):
        return
    for metric in REFERENCE_METRICS:
        ref_value = reference_metrics.get(metric)
        run_value = metrics.get(metric)
        row[f"ref_{metric}"] = ref_value if _is_scalar(ref_value) else ""
        if isinstance(run_value, (int, float)) and isinstance(ref_value, (int, float)):
            row[f"delta_{metric}"] = run_value - ref_value


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _compact_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _require_file(path: Path, flag_name: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{flag_name} does not exist or is not a file: {path}")


def _print_dry_run(
    jobs: Sequence[SweepJob],
    commands: Sequence[Sequence[str]],
    summary_path: Path,
) -> None:
    print(f"Dry run: {len(jobs)} evaluate.py jobs")
    for job, command in zip(jobs, commands):
        print(
            f"\n# motion={job.motion_name} samples={job.samples} iter={job.iterations} "
            f"horizon={job.horizon} control={job.control} seed={job.seed}"
        )
        print(shlex.join(list(command)))
    print(f"\nWould write summary CSV: {summary_path}")
    print("Pass --execute to run the jobs.")


if __name__ == "__main__":
    raise SystemExit(main())
