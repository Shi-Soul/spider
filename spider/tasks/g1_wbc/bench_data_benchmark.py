"""Phase-1 benchmark tooling for G1 WBC on the local bench_data set."""

# ruff: noqa: D101,D102,D103,D105

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_METHODS = (
    "no_mpc",
    "g1_wbc_joint_global",
    "g1_wbc_joint",
    "g1_wbc_ee",
)
MPC_METHODS = frozenset(DEFAULT_METHODS[1:])
DEFAULT_SMOKE_MOTION_IDS = [
    "locomotion_walk__sonic_filtered_220705__loop_backward_walk_001_a017",
    "jump_hop__sonic_filtered_220705__jump_002_a017",
    "crawl_climb__sonic_filtered_230119__"
    "change_idle_crawl_to_idle_crawl_right_001_a125",
]
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = DEFAULT_WORKSPACE_ROOT / "wxy" / "0608_ckpt_bc" / "model_8000.pt"
DEFAULT_REWARD_WEIGHTS = (
    DEFAULT_WORKSPACE_ROOT
    / "g1_wbc_testbed_motion_package_20260617"
    / "metadata"
    / "g1_wbc_reward_weights_method_specific_v14_20260612.json"
)

HARD_GATES = {
    "root_pos_error_mean": (0.25, "root_translation_drift"),
    "root_rot_error_mean": (0.60, "root_orientation_drift"),
    "ee_global_pos_error_mean": (0.25, "ee_global_tracking_failure"),
    "ee_local_pos_error_mean": (0.20, "ee_local_tracking_failure"),
    "contact_mismatch_rate": (0.35, "contact_schedule_failure"),
}
STABILITY_METRICS = (
    "contact_switch_rate",
    "control_delta_mean",
    "joint_acc_mean",
    "joint_jerk_mean",
)


@dataclass(frozen=True)
class BenchmarkSource:
    category: str
    path: Path
    line_no: int


@dataclass(frozen=True)
class BenchMotion:
    id: str
    category: str
    source_group: str
    path: Path
    repo_relative_path: str
    motion_type: str
    fps: float
    frames: int
    duration_sec: float
    max_steps: int


@dataclass(frozen=True)
class RunnerConfig:
    workspace_root: Path
    package_dir: Path
    python_executable: str = sys.executable
    device: str = "cuda:0"
    checkpoint: Path = DEFAULT_CHECKPOINT
    reward_weights: Path = DEFAULT_REWARD_WEIGHTS
    max_steps_cap: int = 800
    methods: tuple[str, ...] = DEFAULT_METHODS
    mpc_preset: str = "aggressive"
    mpc_samples: int = 8192
    mpc_iterations: int = 2
    mpc_planning_horizon_steps: int = 80
    mpc_control_steps: int = 20
    mpc_sampling_mode: str = "knot"
    mpc_knot_count: int = 8
    mpc_temperature: float = 0.7
    mpc_root_pos_sigma: float = 0.08
    mpc_root_rot_sigma: float = 0.18
    mpc_joint_sigma: float = 0.28
    mpc_smooth_passes: int = 0
    mpc_command_reg_weight: float = 0.0
    mpc_command_smooth_weight: float = 0.0
    mpc_guided_root_pos_gain: float = 0.50
    mpc_guided_root_rot_gain: float = 0.50
    mpc_guided_joint_gain: float = 0.50
    mpc_guided_root_pos_clip: float = 0.05
    mpc_guided_root_rot_clip: float = 0.12
    mpc_guided_joint_clip: float = 0.35
    mpc_guided_candidate: bool = True
    mpc_acceptance_gate: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_phase1_methods(self.methods)

    @property
    def spider_root(self) -> Path:
        return self.workspace_root / "spider"


@dataclass(frozen=True)
class EvalCommand:
    motion: BenchMotion
    method: str
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    output_dir: Path
    log_path: Path


@dataclass(frozen=True)
class RenderCommand:
    motion: BenchMotion
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    output_path: Path
    log_path: Path


@dataclass(frozen=True)
class RunStatus:
    track_status: str
    failure_labels: tuple[str, ...]
    missing_metrics: tuple[str, ...] = ()
    hard_failures: int = 0


@dataclass(frozen=True)
class MpcComparison:
    score_delta: float | None
    score_status: str
    failure_labels: tuple[str, ...]


def parse_benchmark_yaml(path: Path) -> list[BenchmarkSource]:
    """Parse the local benchmark list while preserving comment categories."""
    sources: list[BenchmarkSource] = []
    current_category: str | None = None
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current_category = stripped[1:].strip()
            if not current_category:
                raise ValueError(f"{path}:{line_no}: empty category comment")
            continue
        if stripped.startswith("-"):
            if current_category is None:
                raise ValueError(f"{path}:{line_no}: motion appears before any category")
            value = stripped[1:].strip()
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            value = _strip_quotes(value)
            sources.append(BenchmarkSource(current_category, Path(value), line_no))
            continue
        raise ValueError(f"{path}:{line_no}: unsupported benchmark.yaml line: {raw_line!r}")
    return sources


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "item"


def build_motion_id(category: str, source_group: str, motion_stem: str) -> str:
    return "__".join((slugify(category), slugify(source_group), slugify(motion_stem)))


def load_motion_metadata(
    source: BenchmarkSource,
    *,
    workspace_root: Path,
    max_steps_cap: int = 800,
    motion_type: str = "isaaclab",
    expected_fps: float = 50.0,
) -> BenchMotion:
    path = _resolve_source_path(source.path, workspace_root)
    repo_relative_path = _repo_relative(path, workspace_root)
    source_group = _source_group_from_repo_relative(repo_relative_path)
    with np.load(path) as data:
        if "fps" not in data.files:
            raise ValueError(f"{path} is missing required fps field")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        frames = _infer_frame_count(data, path)
    if frames < 2:
        raise ValueError(f"{path} has {frames} frames; expected at least 2")
    if abs(fps - expected_fps) > 1.0e-6:
        raise ValueError(f"{path} has fps={fps:g}; expected {expected_fps:g}")

    max_steps = min(max_steps_cap, frames - 1)
    duration_sec = (frames - 1) / fps
    motion_id = build_motion_id(source.category, source_group, path.stem)
    return BenchMotion(
        id=motion_id,
        category=source.category,
        source_group=source_group,
        path=path,
        repo_relative_path=repo_relative_path,
        motion_type=motion_type,
        fps=fps,
        frames=frames,
        duration_sec=duration_sec,
        max_steps=max_steps,
    )


def resolve_motion_ids(motions: list[BenchMotion]) -> list[BenchMotion]:
    counts: dict[str, int] = {}
    for motion in motions:
        counts[motion.id] = counts.get(motion.id, 0) + 1
    resolved: list[BenchMotion] = []
    for motion in motions:
        if counts[motion.id] == 1:
            resolved.append(motion)
            continue
        suffix = hashlib.sha1(str(motion.path).encode("utf-8")).hexdigest()[:8]
        resolved.append(replace(motion, id=f"{motion.id}__{suffix}"))
    return resolved


def load_bench_motions(
    benchmark_yaml: Path,
    *,
    workspace_root: Path,
    max_steps_cap: int = 800,
) -> list[BenchMotion]:
    sources = parse_benchmark_yaml(benchmark_yaml)
    motions = [
        load_motion_metadata(
            source,
            workspace_root=workspace_root,
            max_steps_cap=max_steps_cap,
        )
        for source in sources
    ]
    return resolve_motion_ids(motions)


def write_input_manifest(
    motions: Sequence[BenchMotion],
    output_path: Path,
    *,
    workspace_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "schema_version: 1",
        f"path_root: {workspace_root.resolve().as_posix()}",
        "path_mode: repo_relative",
        "motions:",
    ]
    for motion in motions:
        lines.extend(
            [
                f"  - motion_id: {motion.id}",
                f"    category: {motion.category}",
                f"    source_group: {motion.source_group}",
                f"    path: {motion.repo_relative_path}",
                f"    motion_type: {motion.motion_type}",
                f"    fps: {_format_number(motion.fps)}",
                f"    frames: {motion.frames}",
                f"    duration_sec: {_format_number(motion.duration_sec)}",
                f"    max_steps: {motion.max_steps}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def load_input_manifest(path: Path, *, workspace_root: Path | None = None) -> list[BenchMotion]:
    """Load manifests generated by write_input_manifest without requiring PyYAML."""
    root = workspace_root
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text().splitlines():
        if raw_line.startswith("path_root:") and root is None:
            root = Path(raw_line.split(":", 1)[1].strip())
        elif raw_line.startswith("  - motion_id:"):
            if current is not None:
                records.append(current)
            current = {"motion_id": raw_line.split(":", 1)[1].strip()}
        elif current is not None and raw_line.startswith("    "):
            key, value = raw_line.strip().split(":", 1)
            current[key] = value.strip()
    if current is not None:
        records.append(current)
    if root is None:
        raise ValueError(f"{path} is missing path_root")

    motions: list[BenchMotion] = []
    for record in records:
        rel_path = record["path"]
        motions.append(
            BenchMotion(
                id=record["motion_id"],
                category=record["category"],
                source_group=record["source_group"],
                path=(root / rel_path).resolve(),
                repo_relative_path=rel_path,
                motion_type=record["motion_type"],
                fps=float(record["fps"]),
                frames=int(record["frames"]),
                duration_sec=float(record["duration_sec"]),
                max_steps=int(record["max_steps"]),
            )
        )
    return motions


def build_eval_command(
    motion: BenchMotion,
    method: str,
    config: RunnerConfig,
) -> EvalCommand:
    if method not in config.methods:
        raise ValueError(f"Unsupported method {method!r}; expected one of {config.methods}")
    if method == "no_mpc":
        output_dir = config.package_dir / "outputs" / "no_mpc" / motion.id
    else:
        output_dir = config.package_dir / "outputs" / "spider" / motion.id / method
    log_path = config.package_dir / "logs" / "eval" / motion.id / f"{method}.log"
    argv = [
        config.python_executable,
        "-m",
        "spider.tasks.g1_wbc.evaluate",
        "--motion",
        str(motion.path),
        "--motion-type",
        motion.motion_type,
        "--checkpoint",
        str(config.checkpoint),
        "--method",
        method,
        "--max-steps",
        str(motion.max_steps),
        "--device",
        config.device,
        "--output-dir",
        str(output_dir),
        "--save-rollout",
    ]
    if method == "no_mpc":
        argv.extend(["--num-envs", "1"])
    else:
        argv.extend(_mpc_argv(config))
    return EvalCommand(
        motion=motion,
        method=method,
        argv=argv,
        env={"PYTHONPATH": str(config.spider_root)},
        cwd=config.spider_root,
        output_dir=output_dir,
        log_path=log_path,
    )


def build_eval_commands(
    motions: Sequence[BenchMotion],
    config: RunnerConfig,
    *,
    mode: str,
    smoke_motion_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[EvalCommand]:
    selected = select_run_motions(
        motions,
        mode=mode,
        smoke_motion_ids=smoke_motion_ids,
        limit=limit,
    )
    return [
        build_eval_command(motion, method, config)
        for motion in selected
        for method in config.methods
    ]


def select_run_motions(
    motions: Sequence[BenchMotion],
    *,
    mode: str,
    smoke_motion_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[BenchMotion]:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode {mode!r}; expected smoke or full")
    if mode == "smoke":
        ids = list(smoke_motion_ids or DEFAULT_SMOKE_MOTION_IDS)
        by_id = {motion.id: motion for motion in motions}
        missing = [motion_id for motion_id in ids if motion_id not in by_id]
        if missing:
            raise ValueError(f"Smoke motion IDs are missing from manifest: {missing}")
        selected = [by_id[motion_id] for motion_id in ids]
    else:
        selected = list(motions)
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_render_command(motion: BenchMotion, config: RunnerConfig) -> RenderCommand:
    video_name = (
        f"{motion.id}_g1_wbc_bc_phase1_4method_2x2_"
        f"rollout{motion.max_steps}_mp4v.mp4"
    )
    output_path = config.package_dir / "videos" / "four_panel" / video_name
    log_path = config.package_dir / "logs" / "render" / f"{motion.id}.log"
    argv = [
        config.python_executable,
        str(config.spider_root / "scripts" / "visualize_g1_wbc.py"),
        "--motion",
        str(motion.path),
        "--motion-type",
        motion.motion_type,
        "--method",
        "saved",
        "--max-steps",
        str(motion.max_steps),
        "--device",
        config.device,
    ]
    for method in DEFAULT_METHODS:
        argv.extend(["--saved-rollout", f"{method}:{_rollout_path(config, motion, method)}"])
    argv.extend(
        [
            "--saved-env-index",
            "0",
            "--panel-layout",
            "2x2",
            "--width",
            "960",
            "--height",
            "540",
            "--fps",
            "50",
            "--camera-mode",
            "ref-follow",
            "--show-root-error",
            "--output",
            str(output_path),
        ]
    )
    return RenderCommand(
        motion=motion,
        argv=argv,
        env={"PYTHONPATH": str(config.spider_root)},
        cwd=config.spider_root,
        output_path=output_path,
        log_path=log_path,
    )


def run_eval_command(
    command: EvalCommand,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command.output_dir.mkdir(parents=True, exist_ok=True)
    return _run_logged_command(command.argv, command.cwd, command.env, command.log_path, timeout_sec)


def run_render_command(
    command: RenderCommand,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command.output_path.parent.mkdir(parents=True, exist_ok=True)
    return _run_logged_command(command.argv, command.cwd, command.env, command.log_path, timeout_sec)


def classify_run_metrics(metrics: Mapping[str, Any] | None) -> RunStatus:
    if not metrics:
        return RunStatus("unknown", ("missing_required_metrics",), tuple(HARD_GATES), 0)
    missing = tuple(name for name in HARD_GATES if name not in metrics)
    if missing:
        return RunStatus("unknown", ("missing_required_metrics",), missing, 0)

    failures: list[str] = []
    borderline = False
    for name, (threshold, label) in HARD_GATES.items():
        value = float(metrics[name])
        if value >= threshold:
            failures.append(label)
        elif value >= 0.8 * threshold:
            borderline = True
    if "contact_schedule_failure" in failures:
        fp_rate = float(metrics.get("contact_false_positive_rate", 0.0))
        fn_rate = float(metrics.get("contact_false_negative_rate", 0.0))
        if fp_rate > fn_rate:
            failures.append("contact_false_positive_dominant")
        elif fn_rate > fp_rate:
            failures.append("contact_false_negative_dominant")
    if failures:
        return RunStatus("failed", tuple(failures), (), len(failures))
    if borderline:
        return RunStatus("borderline", ("borderline",), (), 0)
    return RunStatus("tracked", (), (), 0)


def compare_baseline_and_mpc(
    baseline_metrics: Mapping[str, Any] | None,
    mpc_metrics: Mapping[str, Any] | None,
) -> MpcComparison:
    labels: list[str] = []
    if not baseline_metrics or not mpc_metrics:
        return MpcComparison(None, "missing", ("missing_required_metrics",))
    if "score" not in baseline_metrics or "score" not in mpc_metrics:
        return MpcComparison(None, "missing", ("missing_required_metrics",))

    score_delta = float(mpc_metrics["score"]) - float(baseline_metrics["score"])
    if score_delta > 0:
        score_status = "improved"
    elif score_delta < 0:
        score_status = "regressed"
        labels.append("baseline_regression")
    else:
        score_status = "same"

    for name in STABILITY_METRICS:
        if (
            name in baseline_metrics
            and name in mpc_metrics
            and float(mpc_metrics[name]) > float(baseline_metrics[name])
        ):
            labels.append("stability_regression")
            break
    return MpcComparison(score_delta, score_status, tuple(labels))


def generate_benchmark_summaries(
    package_dir: Path,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    motion_ids: Sequence[str] | None = None,
) -> None:
    manifest_path = package_dir / "input_motions" / "bench_data.yaml"
    benchmark_dir = package_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    motions = load_input_manifest(manifest_path)
    if motion_ids is not None:
        selected_ids = set(motion_ids)
        motions = [motion for motion in motions if motion.id in selected_ids]
    metrics_by_key = {
        (motion.id, method): _load_run_metrics(_metrics_path(package_dir, motion, method))
        for motion in motions
        for method in methods
    }

    score_rows = _build_score_rows(motions, metrics_by_key)
    full_delta_rows = _build_full_delta_rows(motions, metrics_by_key)
    ranking_rows = _build_ranking_rows(motions, metrics_by_key, methods)
    failure_rows = _build_failure_rows(motions, metrics_by_key, methods)
    category_rows = _build_category_rows(failure_rows)
    runtime_rows = _build_runtime_rows(package_dir, motions, metrics_by_key, methods)

    _write_csv(benchmark_dir / "baseline_vs_mpc_score.csv", score_rows)
    _write_csv(benchmark_dir / "baseline_vs_mpc_full_deltas.csv", full_delta_rows)
    _write_csv(benchmark_dir / "method_ranking.csv", ranking_rows)
    _write_csv(benchmark_dir / "motion_failures.csv", failure_rows)
    _write_csv(benchmark_dir / "category_summary.csv", category_rows)
    _write_csv(benchmark_dir / "runtime_summary.csv", runtime_rows)
    (benchmark_dir / "benchmark_summary.md").write_text(
        _render_summary_md(score_rows, ranking_rows, failure_rows, category_rows)
    )


def write_package_metadata(
    motions: Sequence[BenchMotion],
    selected_motions: Sequence[BenchMotion],
    eval_commands: Sequence[EvalCommand],
    render_commands: Sequence[RenderCommand],
    config: RunnerConfig,
    *,
    mode: str,
    dry_run: bool,
) -> None:
    metadata_dir = config.package_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_bench_data_sources(metadata_dir / "bench_data_sources.csv", motions)
    _write_benchmark_config(
        metadata_dir / "benchmark_config.json",
        config,
        mode=mode,
        dry_run=dry_run,
    )
    _write_eval_commands(metadata_dir / "eval_commands.sh", eval_commands)
    _write_render_commands(metadata_dir / "render_commands.sh", render_commands)
    _write_video_manifest(config.package_dir / "videos" / "manifest.csv", render_commands)
    _write_package_summary(
        metadata_dir / "package_summary.json",
        config,
        motions=motions,
        selected_motions=selected_motions,
        eval_commands=eval_commands,
        render_commands=render_commands,
        dry_run=dry_run,
    )


def write_package_manifests(package_dir: Path) -> None:
    metadata_dir = package_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sha_path = metadata_dir / "manifest.sha256"
    csv_path = metadata_dir / "manifest.csv"

    sha_entries = []
    for path in _iter_package_files(package_dir):
        rel = path.relative_to(package_dir).as_posix()
        if rel in {"metadata/manifest.csv", "metadata/manifest.sha256"}:
            continue
        sha_entries.append((_sha256(path), rel))
    sha_entries.sort(key=lambda item: item[1])
    sha_path.write_text("".join(f"{digest}  {rel}\n" for digest, rel in sha_entries))

    rows = []
    for path in _iter_package_files(package_dir):
        rel = path.relative_to(package_dir).as_posix()
        if rel == "metadata/manifest.csv":
            continue
        rows.append({"path": rel, "bytes": str(path.stat().st_size), "sha256": _sha256(path)})
    rows.sort(key=lambda row: row["path"])
    _write_csv(csv_path, rows, fieldnames=("path", "bytes", "sha256"))


def run_benchmark(
    argv: Sequence[str] | None = None,
    *,
    run_eval: Callable[[EvalCommand, int | None], Any] = run_eval_command,
    run_render: Callable[[RenderCommand, int | None], Any] = run_render_command,
) -> int:
    args = _parse_args(argv)
    workspace_root = args.workspace_root.expanduser().resolve()
    benchmark_yaml = args.benchmark_yaml.expanduser().resolve()
    package_dir = _resolve_package_dir(args, workspace_root)
    config = RunnerConfig(
        workspace_root=workspace_root,
        package_dir=package_dir,
        python_executable=args.python_executable,
        device=args.device,
        checkpoint=args.checkpoint.expanduser().resolve(),
        reward_weights=args.reward_weights.expanduser().resolve(),
        max_steps_cap=args.max_steps_cap,
        methods=tuple(args.methods),
    )

    motions = load_bench_motions(
        benchmark_yaml,
        workspace_root=workspace_root,
        max_steps_cap=args.max_steps_cap,
    )
    selected_motions = select_run_motions(
        motions,
        mode=args.mode,
        smoke_motion_ids=args.smoke_motion_id,
        limit=args.limit,
    )
    eval_commands = [
        build_eval_command(motion, method, config)
        for motion in selected_motions
        for method in config.methods
    ]
    render_commands = [build_render_command(motion, config) for motion in selected_motions]

    _prepare_package_dirs(config.package_dir)
    write_input_manifest(
        motions,
        config.package_dir / "input_motions" / "bench_data.yaml",
        workspace_root=workspace_root,
    )
    write_package_metadata(
        motions,
        selected_motions,
        eval_commands,
        render_commands,
        config,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    failures = 0
    if not args.manifest_only and not args.dry_run:
        for command in eval_commands:
            if args.skip_existing and (command.output_dir / "metrics.json").exists():
                continue
            try:
                result = run_eval(command, args.timeout_sec)
            except subprocess.TimeoutExpired as exc:
                failures += 1
                _write_timeout_log(command.log_path, exc)
                continue
            if getattr(result, "returncode", 0) != 0:
                failures += 1
        for command in render_commands:
            if args.skip_existing and command.output_path.exists():
                continue
            if not _render_inputs_exist(command):
                failures += 1
                _write_missing_render_log(command)
                continue
            try:
                result = run_render(command, args.timeout_sec)
            except subprocess.TimeoutExpired as exc:
                failures += 1
                _write_timeout_log(command.log_path, exc)
                continue
            if getattr(result, "returncode", 0) != 0:
                failures += 1

    generate_benchmark_summaries(
        config.package_dir,
        methods=config.methods,
        motion_ids=[motion.id for motion in selected_motions],
    )
    write_package_metadata(
        motions,
        selected_motions,
        eval_commands,
        render_commands,
        config,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    write_package_manifests(config.package_dir)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_benchmark(argv)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument(
        "--benchmark-yaml",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT / "bench_data" / "benchmark.yaml",
    )
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--date", default=None, help="YYYYMMDD package date stamp.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reward-weights", type=Path, default=DEFAULT_REWARD_WEIGHTS)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--max-steps-cap", type=int, default=800)
    parser.add_argument("--smoke-motion-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args(argv)


def _mpc_argv(config: RunnerConfig) -> list[str]:
    return [
        "--mpc-preset",
        config.mpc_preset,
        "--mpc-reward-weights",
        str(config.reward_weights),
        "--mpc-samples",
        str(config.mpc_samples),
        "--mpc-iterations",
        str(config.mpc_iterations),
        "--mpc-planning-horizon-steps",
        str(config.mpc_planning_horizon_steps),
        "--mpc-control-steps",
        str(config.mpc_control_steps),
        "--mpc-sampling-mode",
        config.mpc_sampling_mode,
        "--mpc-knot-count",
        str(config.mpc_knot_count),
        "--mpc-temperature",
        str(config.mpc_temperature),
        "--mpc-root-pos-sigma",
        str(config.mpc_root_pos_sigma),
        "--mpc-root-rot-sigma",
        str(config.mpc_root_rot_sigma),
        "--mpc-joint-sigma",
        str(config.mpc_joint_sigma),
        "--mpc-smooth-passes",
        str(config.mpc_smooth_passes),
        "--mpc-command-reg-weight",
        str(config.mpc_command_reg_weight),
        "--mpc-command-smooth-weight",
        str(config.mpc_command_smooth_weight),
        "--mpc-guided-root-pos-gain",
        str(config.mpc_guided_root_pos_gain),
        "--mpc-guided-root-rot-gain",
        str(config.mpc_guided_root_rot_gain),
        "--mpc-guided-joint-gain",
        str(config.mpc_guided_joint_gain),
        "--mpc-guided-root-pos-clip",
        str(config.mpc_guided_root_pos_clip),
        "--mpc-guided-root-rot-clip",
        str(config.mpc_guided_root_rot_clip),
        "--mpc-guided-joint-clip",
        str(config.mpc_guided_joint_clip),
        "--mpc-guided-candidate",
        "--mpc-acceptance-gate",
        "--seed",
        str(config.seed),
    ]


def _validate_phase1_methods(methods: Sequence[str]) -> None:
    if tuple(methods) != DEFAULT_METHODS:
        unsupported = [method for method in methods if method not in DEFAULT_METHODS]
        if unsupported:
            raise ValueError(f"Unsupported method(s): {unsupported}")
        raise ValueError(f"Phase-1 four-panel benchmark requires methods {DEFAULT_METHODS}")


def _rollout_path(config: RunnerConfig, motion: BenchMotion, method: str) -> Path:
    if method == "no_mpc":
        return config.package_dir / "outputs" / "no_mpc" / motion.id / "rollout.npz"
    return config.package_dir / "outputs" / "spider" / motion.id / method / "rollout.npz"


def _metrics_path(package_dir: Path, motion: BenchMotion, method: str) -> Path:
    if method == "no_mpc":
        return package_dir / "outputs" / "no_mpc" / motion.id / "metrics.json"
    return package_dir / "outputs" / "spider" / motion.id / method / "metrics.json"


def _resolve_source_path(path: Path, workspace_root: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (workspace_root / path).expanduser().resolve()


def _repo_relative(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is outside workspace root {workspace_root}") from exc


def _source_group_from_repo_relative(repo_relative_path: str) -> str:
    parts = Path(repo_relative_path).parts
    if len(parts) >= 3 and parts[0] == "bench_data":
        return "/".join(parts[1:-1])
    return "/".join(parts[:-1])


def _infer_frame_count(data: np.lib.npyio.NpzFile, path: Path) -> int:
    for key in ("joint_pos", "body_pos_w", "body_quat_w", "joint_vel"):
        if key in data.files:
            value = data[key]
            if value.ndim == 0:
                continue
            return int(value.shape[0])
    raise ValueError(f"{path} has no frame-major arrays")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _format_number(value: float) -> str:
    return f"{value:.10g}"


def _run_logged_command(
    argv: Sequence[str],
    cwd: Path,
    env_delta: Mapping[str, str],
    log_path: Path,
    timeout_sec: int | None,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_delta)
    with log_path.open("w") as log_file:
        log_file.write(f"START {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(shlex.join(argv) + "\n\n")
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        log_file.write(f"\nDONE {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(f"RETURNCODE {result.returncode}\n")
    return result


def _prepare_package_dirs(package_dir: Path) -> None:
    for rel in (
        "input_motions",
        "outputs/no_mpc",
        "outputs/spider",
        "benchmark",
        "videos/four_panel",
        "logs/eval",
        "logs/render",
        "metadata",
    ):
        (package_dir / rel).mkdir(parents=True, exist_ok=True)


def _write_bench_data_sources(path: Path, motions: Sequence[BenchMotion]) -> None:
    rows = [
        {
            "motion_id": motion.id,
            "category": motion.category,
            "source_group": motion.source_group,
            "path": motion.repo_relative_path,
            "absolute_path": str(motion.path),
            "sha256": _sha256(motion.path),
            "fps": _format_number(motion.fps),
            "frames": str(motion.frames),
            "duration_sec": _format_number(motion.duration_sec),
            "max_steps": str(motion.max_steps),
            "motion_type": motion.motion_type,
        }
        for motion in motions
    ]
    _write_csv(path, rows)


def _write_benchmark_config(
    path: Path,
    config: RunnerConfig,
    *,
    mode: str,
    dry_run: bool,
) -> None:
    payload = {
        "schema_version": 1,
        "benchmark": "g1_wbc_bench_data_phase1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dry_run": dry_run,
        "workspace_root": str(config.workspace_root),
        "package_dir": str(config.package_dir),
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": _sha256_if_exists(config.checkpoint),
        "reward_weights": str(config.reward_weights),
        "reward_weights_sha256": _sha256_if_exists(config.reward_weights),
        "methods": list(config.methods),
        "panel_order": list(DEFAULT_METHODS),
        "eval": {
            "motion_type": "isaaclab",
            "max_steps_cap": config.max_steps_cap,
            "device": config.device,
            "python_executable": config.python_executable,
        },
        "mpc": {
            "preset": config.mpc_preset,
            "samples": config.mpc_samples,
            "iterations": config.mpc_iterations,
            "planning_horizon_steps": config.mpc_planning_horizon_steps,
            "control_steps": config.mpc_control_steps,
            "sampling_mode": config.mpc_sampling_mode,
            "knot_count": config.mpc_knot_count,
            "temperature": config.mpc_temperature,
            "root_pos_sigma": config.mpc_root_pos_sigma,
            "root_rot_sigma": config.mpc_root_rot_sigma,
            "joint_sigma": config.mpc_joint_sigma,
            "smooth_passes": config.mpc_smooth_passes,
            "command_reg_weight": config.mpc_command_reg_weight,
            "command_smooth_weight": config.mpc_command_smooth_weight,
            "guided_root_pos_gain": config.mpc_guided_root_pos_gain,
            "guided_root_rot_gain": config.mpc_guided_root_rot_gain,
            "guided_joint_gain": config.mpc_guided_joint_gain,
            "guided_root_pos_clip": config.mpc_guided_root_pos_clip,
            "guided_root_rot_clip": config.mpc_guided_root_rot_clip,
            "guided_joint_clip": config.mpc_guided_joint_clip,
            "guided_candidate": config.mpc_guided_candidate,
            "acceptance_gate": config.mpc_acceptance_gate,
            "seed": config.seed,
        },
        "thresholds": {
            name: threshold for name, (threshold, _label) in HARD_GATES.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_eval_commands(path: Path, commands: Sequence[EvalCommand]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if commands:
        lines.extend(_shell_context_lines(commands[0].cwd, commands[0].env))
    for command in commands:
        lines.append(f"mkdir -p {shlex.quote(str(command.log_path.parent))}")
        lines.append(
            f"{shlex.join(command.argv)} > {shlex.quote(str(command.log_path))} 2>&1"
        )
        lines.append("")
    path.write_text("\n".join(lines))


def _write_render_commands(path: Path, commands: Sequence[RenderCommand]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if commands:
        lines.extend(_shell_context_lines(commands[0].cwd, commands[0].env))
    for command in commands:
        lines.append(f"mkdir -p {shlex.quote(str(command.log_path.parent))}")
        lines.append(
            f"{shlex.join(command.argv)} > {shlex.quote(str(command.log_path))} 2>&1"
        )
        lines.append("")
    path.write_text("\n".join(lines))


def _write_video_manifest(path: Path, commands: Sequence[RenderCommand]) -> None:
    rows = []
    for index, command in enumerate(commands):
        exists = command.output_path.exists()
        rows.append(
            {
                "motion_id": command.motion.id,
                "category": command.motion.category,
                "path": command.output_path.relative_to(command.output_path.parents[2]).as_posix()
                if exists
                else str(command.output_path),
                "frames": "",
                "width": "",
                "height": "",
                "fps": "50",
                "codec": "mp4v",
                "bytes": str(command.output_path.stat().st_size) if exists else "",
                "sha256": _sha256(command.output_path) if exists else "",
                "status": "rendered" if exists else "missing",
                "log_path": str(command.log_path),
                "render_command_id": str(index),
            }
        )
    _write_csv(path, rows)


def _write_package_summary(
    path: Path,
    config: RunnerConfig,
    *,
    motions: Sequence[BenchMotion],
    selected_motions: Sequence[BenchMotion],
    eval_commands: Sequence[EvalCommand],
    render_commands: Sequence[RenderCommand],
    dry_run: bool,
) -> None:
    files = list(_iter_package_files(config.package_dir))
    payload = {
        "schema_version": 1,
        "package_dir": str(config.package_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "motion_count": len(motions),
        "selected_motion_count": len(selected_motions),
        "category_count": len({motion.category for motion in motions}),
        "expected_eval_runs": len(eval_commands),
        "completed_eval_runs": sum(
            1 for command in eval_commands if (command.output_dir / "metrics.json").exists()
        ),
        "expected_videos": len(render_commands),
        "completed_videos": sum(1 for command in render_commands if command.output_path.exists()),
        "total_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _render_inputs_exist(command: RenderCommand) -> bool:
    for index, value in enumerate(command.argv):
        if value != "--saved-rollout":
            continue
        rollout_path = Path(command.argv[index + 1].split(":", 1)[1])
        if not rollout_path.exists():
            return False
    return True


def _write_missing_render_log(command: RenderCommand) -> None:
    command.log_path.parent.mkdir(parents=True, exist_ok=True)
    command.log_path.write_text("SKIPPED missing rollout input for four-panel render\n")


def _write_timeout_log(log_path: Path, exc: subprocess.TimeoutExpired) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "TIMEOUT\n"
        f"command: {shlex.join(str(part) for part in exc.cmd)}\n"
        f"timeout_sec: {exc.timeout}\n"
    )


def _shell_context_lines(cwd: Path, env: Mapping[str, str]) -> list[str]:
    lines = [f"cd {shlex.quote(str(cwd))}"]
    for key, value in sorted(env.items()):
        if key == "PYTHONPATH":
            lines.append(
                f"export PYTHONPATH={shlex.quote(value)}:${{{key}:-}}"
            )
        else:
            lines.append(f"export {key}={shlex.quote(value)}")
    lines.append("")
    return lines


def _load_run_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    metrics = dict(payload.get("metrics", {}))
    if "success" in metrics and "success_current" not in metrics:
        metrics["success_current"] = metrics["success"]
    if "mpc" in payload:
        metrics["_mpc_num_windows"] = payload["mpc"].get("num_windows")
    return metrics


def _build_score_rows(
    motions: Sequence[BenchMotion],
    metrics_by_key: Mapping[tuple[str, str], Mapping[str, Any] | None],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for motion in motions:
        baseline = metrics_by_key.get((motion.id, "no_mpc"))
        for method in DEFAULT_METHODS[1:]:
            mpc = metrics_by_key.get((motion.id, method))
            comparison = compare_baseline_and_mpc(baseline, mpc)
            rows.append(
                {
                    "motion_id": motion.id,
                    "category": motion.category,
                    "method": method,
                    "baseline_score": _metric_text(baseline, "score"),
                    "mpc_score": _metric_text(mpc, "score"),
                    "score_delta": ""
                    if comparison.score_delta is None
                    else _format_number(comparison.score_delta),
                    "score_status": comparison.score_status,
                    "failure_labels": "|".join(comparison.failure_labels),
                }
            )
    return rows


def _build_full_delta_rows(
    motions: Sequence[BenchMotion],
    metrics_by_key: Mapping[tuple[str, str], Mapping[str, Any] | None],
) -> list[dict[str, str]]:
    metric_names = sorted(
        {
            name
            for metrics in metrics_by_key.values()
            if metrics is not None
            for name, value in metrics.items()
            if isinstance(value, (int, float, bool))
        }
    )
    if not metric_names:
        metric_names = sorted((*HARD_GATES.keys(), "score", "success_current"))
    rows: list[dict[str, str]] = []
    for motion in motions:
        baseline = metrics_by_key.get((motion.id, "no_mpc"))
        for method in DEFAULT_METHODS[1:]:
            mpc = metrics_by_key.get((motion.id, method))
            for name in metric_names:
                rows.append(
                    {
                        "motion_id": motion.id,
                        "category": motion.category,
                        "method": method,
                        "metric": name,
                        "baseline_value": _metric_text(baseline, name),
                        "mpc_value": _metric_text(mpc, name),
                        "delta": _delta_text(baseline, mpc, name),
                        "status": _delta_status(baseline, mpc, name),
                    }
                )
    return rows


def _build_ranking_rows(
    motions: Sequence[BenchMotion],
    metrics_by_key: Mapping[tuple[str, str], Mapping[str, Any] | None],
    methods: Sequence[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    method_order = {method: index for index, method in enumerate(methods)}
    for motion in motions:
        ranked = []
        for method in methods:
            metrics = metrics_by_key.get((motion.id, method))
            status = classify_run_metrics(metrics)
            ranked.append((method, metrics, status))
        ranked.sort(
            key=lambda item: (
                _success_sort_value(item[1]),
                _score_sort_value(item[1]),
                -item[2].hard_failures,
                -method_order[item[0]],
            ),
            reverse=True,
        )
        for rank, (method, metrics, status) in enumerate(ranked, start=1):
            rows.append(
                {
                    "motion_id": motion.id,
                    "category": motion.category,
                    "rank": str(rank),
                    "method": method,
                    "score": _metric_text(metrics, "score"),
                    "success_current": _metric_text(metrics, "success_current"),
                    "track_status": status.track_status,
                    "failure_labels": "|".join(status.failure_labels),
                }
            )
    return rows


def _build_failure_rows(
    motions: Sequence[BenchMotion],
    metrics_by_key: Mapping[tuple[str, str], Mapping[str, Any] | None],
    methods: Sequence[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for motion in motions:
        baseline = metrics_by_key.get((motion.id, "no_mpc"))
        for method in methods:
            metrics = metrics_by_key.get((motion.id, method))
            status = classify_run_metrics(metrics)
            labels = list(status.failure_labels)
            if method != "no_mpc":
                labels.extend(compare_baseline_and_mpc(baseline, metrics).failure_labels)
            rows.append(
                {
                    "motion_id": motion.id,
                    "category": motion.category,
                    "method": method,
                    "track_status": status.track_status,
                    "success_current": _metric_text(metrics, "success_current"),
                    "score": _metric_text(metrics, "score"),
                    "failure_labels": "|".join(dict.fromkeys(labels)),
                    "missing_metrics": "|".join(status.missing_metrics),
                }
            )
    return rows


def _build_category_rows(failure_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for row in failure_rows:
        key = (row["category"], row["method"])
        bucket = buckets.setdefault(
            key,
            {"motions": 0, "tracked": 0, "borderline": 0, "failed": 0, "unknown": 0},
        )
        bucket["motions"] += 1
        bucket[row["track_status"]] += 1
    return [
        {
            "category": category,
            "method": method,
            "motions": str(values["motions"]),
            "tracked": str(values["tracked"]),
            "borderline": str(values["borderline"]),
            "failed": str(values["failed"]),
            "unknown": str(values["unknown"]),
        }
        for (category, method), values in sorted(buckets.items())
    ]


def _build_runtime_rows(
    package_dir: Path,
    motions: Sequence[BenchMotion],
    metrics_by_key: Mapping[tuple[str, str], Mapping[str, Any] | None],
    methods: Sequence[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for motion in motions:
        for method in methods:
            metrics = metrics_by_key.get((motion.id, method))
            log_path = package_dir / "logs" / "eval" / motion.id / f"{method}.log"
            rows.append(
                {
                    "motion_id": motion.id,
                    "category": motion.category,
                    "method": method,
                    "log_path": str(log_path),
                    "num_steps": _metric_text(metrics, "num_steps"),
                    "num_windows": _metric_text(metrics, "_mpc_num_windows"),
                    "status": "present" if log_path.exists() else "missing",
                }
            )
    return rows


def _render_summary_md(
    score_rows: Sequence[Mapping[str, str]],
    ranking_rows: Sequence[Mapping[str, str]],
    failure_rows: Sequence[Mapping[str, str]],
    category_rows: Sequence[Mapping[str, str]],
) -> str:
    improved = sum(1 for row in score_rows if row["score_status"] == "improved")
    regressed = sum(1 for row in score_rows if row["score_status"] == "regressed")
    unknown = sum(1 for row in failure_rows if row["track_status"] == "unknown")
    lines = [
        "# G1 WBC Bench Data Phase 1 Summary",
        "",
        f"- MPC score improvements: {improved}",
        f"- MPC score regressions: {regressed}",
        f"- Unknown runs: {unknown}",
        "",
        "## Best Method Counts",
        "",
    ]
    best_counts: dict[str, int] = {}
    for row in ranking_rows:
        if row["rank"] == "1" and row["track_status"] != "unknown":
            best_counts[row["method"]] = best_counts.get(row["method"], 0) + 1
    for method, count in sorted(best_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {method}: {count}")
    lines.extend(["", "## Category Rows", ""])
    for row in category_rows[:20]:
        lines.append(
            f"- {row['category']} / {row['method']}: "
            f"tracked={row['tracked']}, borderline={row['borderline']}, "
            f"failed={row['failed']}, unknown={row['unknown']}"
        )
    return "\n".join(lines) + "\n"


def _metric_text(metrics: Mapping[str, Any] | None, name: str) -> str:
    if metrics is None or name not in metrics:
        return ""
    value = metrics[name]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(float(value))
    return str(value)


def _delta_text(
    baseline: Mapping[str, Any] | None,
    mpc: Mapping[str, Any] | None,
    name: str,
) -> str:
    if baseline is None or mpc is None or name not in baseline or name not in mpc:
        return ""
    if not isinstance(baseline[name], (int, float)) or not isinstance(mpc[name], (int, float)):
        return ""
    return _format_number(float(mpc[name]) - float(baseline[name]))


def _delta_status(
    baseline: Mapping[str, Any] | None,
    mpc: Mapping[str, Any] | None,
    name: str,
) -> str:
    delta = _delta_text(baseline, mpc, name)
    if not delta:
        return "missing"
    delta_value = float(delta)
    if name in {"score", "success_current"}:
        return "improved" if delta_value > 0 else "regressed" if delta_value < 0 else "same"
    return "improved" if delta_value < 0 else "regressed" if delta_value > 0 else "same"


def _success_sort_value(metrics: Mapping[str, Any] | None) -> int:
    return 1 if metrics and bool(metrics.get("success_current", metrics.get("success"))) else 0


def _score_sort_value(metrics: Mapping[str, Any] | None) -> float:
    if metrics is None or "score" not in metrics:
        return float("-inf")
    return float(metrics["score"])


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(rows[0].keys()) if rows else ("empty",)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _iter_package_files(package_dir: Path) -> Iterable[Path]:
    if not package_dir.exists():
        return []
    return sorted(path for path in package_dir.rglob("*") if path.is_file())


def _resolve_package_dir(args: argparse.Namespace, workspace_root: Path) -> Path:
    if args.package_dir is not None:
        return args.package_dir.expanduser().resolve()
    date = args.date or datetime.now().strftime("%Y%m%d")
    suffix = "_smoke" if args.mode == "smoke" else ""
    return (workspace_root / f"g1_wbc_bench_data_benchmark_{date}{suffix}").resolve()


if __name__ == "__main__":
    raise SystemExit(main())
