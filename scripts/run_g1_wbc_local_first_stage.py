#!/usr/bin/env python3
"""Plan and run the G1 WBC local-first low-sample stage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from spider.tasks.g1_wbc import local_first_stage as stage

SPIDER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SPIDER_ROOT.parent
LOW_SAMPLE_SWEEP_SCRIPT = SPIDER_ROOT / "scripts" / "run_g1_wbc_low_sample_sweep.py"
TESTBED_PACKAGE_ROOT = WORKSPACE_ROOT / "g1_wbc_testbed_motion_package_20260617"
DEFAULT_REWARD_WEIGHTS = (
    TESTBED_PACKAGE_ROOT
    / "metadata"
    / "g1_wbc_reward_weights_method_specific_v14_20260612.json"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/g1_wbc_local_first_stage")
DEFAULT_PYTHON_EXECUTABLE = (
    SPIDER_ROOT / ".venv" / "bin" / "python"
    if (SPIDER_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)

CONTACT_CANDIDATE = "jg_s128_L150_contact"
DEFAULT_SCREENING_CANDIDATES = stage.NEXT_STAGE_CANDIDATE_NAMES
DEFAULT_MOTIONS = ("jump",)
DEFAULT_PROMOTED_MOTIONS = ("walk", "qixing")
DEFAULT_TRANSFER_MOTIONS = ("walk", "qixing")
FIXED_SWEEP_ARGS = {
    "samples": "128",
    "iterations": "2",
    "horizons": "40",
    "controls": "20",
    "knot_counts": "8",
    "sigma_triplets": "0.04,0.1,0.18",
    "seeds": "0",
    "max_steps": "800",
}
REQUIRED_METRICS = (
    "score",
    "num_steps",
    *stage.GLOBAL_GUARDRAIL_METRICS,
    *stage.LOCAL_IMPROVEMENT_METRICS,
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)
REQUIRED_MPC = ("accepted", "accepted_windows", "used_baseline_fallback")
GUARDRAIL_SUMMARY_COLUMNS = [
    "candidate",
    "motion",
    "status",
    "passed",
    "failure_labels",
    "improved_local_count",
    "global_guardrail_passed",
    "smooth_guardrail_passed",
    "contact_guardrail_passed",
    "score_guardrail_passed",
    "local_guardrail_passed",
    "mpc_guardrail_passed",
    "runtime_guardrail_passed",
    "summary_csv",
]
FRONTIER_SUMMARY_COLUMNS = [
    "rank",
    "candidate",
    "motion",
    "status",
    "frontier_class",
    "failure_labels",
    "hard_guardrail_passed",
    "score_relaxed_passed",
    "global_relaxed_passed",
    "mpc_guardrail_passed",
    "runtime_guardrail_passed",
    "local_count_3",
    "local_count_5",
    "local_count_8",
    "local_composite_improvement",
    "score_delta",
    "max_global_ratio",
    "contact_delta",
    "control_delta_regression",
    "joint_acc_regression",
    "summary_csv",
]
TRANSFER_SUMMARY_COLUMNS = [
    "rank",
    "candidate",
    "motion",
    "status",
    "transfer_class",
    "failure_labels",
    "score_guardrail_passed",
    "global_guardrail_passed",
    "smooth_guardrail_passed",
    "contact_guardrail_passed",
    "local_transfer_passed",
    "mpc_guardrail_passed",
    "runtime_guardrail_passed",
    "local_count_3",
    "max_local_regression",
    "score_delta",
    "max_global_ratio",
    "contact_delta",
    "control_delta_regression",
    "joint_acc_regression",
    "summary_csv",
]
PARETO_SUMMARY_COLUMNS = [
    "rank",
    "candidate",
    "motion",
    "samples",
    "status",
    "pareto_class",
    "failure_labels",
    "repeat_count",
    "ok_count",
    "success_count",
    "accepted_count",
    "fallback_count",
    "full_run_count",
    "accepted_windows_count",
    "duration_sec_mean",
    "score_mean",
    "root_pos_error_mean_mean",
    "body_global_pos_error_mean_mean",
    "ee_global_pos_error_mean_mean",
    "ee_local_pos_error_mean_mean",
    "contact_mismatch_rate_mean",
    "control_delta_mean_mean",
    "joint_acc_mean_mean",
    "score_delta_vs_best_mean",
    "score_delta_vs_baseline_mean",
    "max_global_ratio_vs_best_mean",
    "max_global_ratio_vs_baseline_mean",
    "summary_csv",
]
FRONTIER_CLASS_RANK = {
    "local_frontier": 0,
    "near_frontier": 1,
    "dead_end": 2,
    "invalid": 3,
}
TRANSFER_CLASS_RANK = {"transfer_pass": 0, "recovery_candidate": 1, "invalid": 2}
PARETO_CLASS_RANK = {
    "baseline_close": 0,
    "promising_budget": 1,
    "stable_but_low_quality": 2,
    "unstable": 3,
    "invalid": 4,
}


def sweep_args_for_stage(samples: Sequence[int], seeds: Sequence[int]) -> dict[str, str]:
    """Return low-sample sweep arguments for the selected sample and seed ladder."""

    args = dict(FIXED_SWEEP_ARGS)
    args["samples"] = " ".join(str(value) for value in samples)
    args["seeds"] = " ".join(str(value) for value in seeds)
    return args


def plan_sweep_commands(
    *,
    output_root: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
    python_executable: str,
    device: str,
    output_group: str = "candidates",
    sweep_args: Mapping[str, str] | None = None,
) -> list[list[str]]:
    """Build low-sample sweep invocations for each local-first candidate."""

    root = Path(output_root)
    selected_sweep_args = dict(FIXED_SWEEP_ARGS if sweep_args is None else sweep_args)
    commands: list[list[str]] = []
    for candidate_name in candidate_names:
        _validate_candidate(candidate_name)
        command = [
            python_executable,
            str(LOW_SAMPLE_SWEEP_SCRIPT),
            "--execute",
            "--output-root",
            str(candidate_output_root(root, candidate_name, output_group=output_group)),
            "--samples",
            *_sweep_values(selected_sweep_args, "samples"),
            "--iterations",
            *_sweep_values(selected_sweep_args, "iterations"),
            "--horizons",
            *_sweep_values(selected_sweep_args, "horizons"),
            "--controls",
            *_sweep_values(selected_sweep_args, "controls"),
            "--knot-counts",
            *_sweep_values(selected_sweep_args, "knot_counts"),
            "--sigma-triplets",
            *_sweep_values(selected_sweep_args, "sigma_triplets"),
            "--seeds",
            *_sweep_values(selected_sweep_args, "seeds"),
            "--max-steps",
            *_sweep_values(selected_sweep_args, "max_steps"),
            "--python-executable",
            python_executable,
            "--device",
            device,
            "--mpc-reward-weights",
            str(candidate_reward_path(root, candidate_name)),
            "--motions",
            *motion_names,
        ]
        commands.append(command)
    return commands


def write_dry_run_artifacts(
    *,
    output_root: str | Path,
    base_reward_weights_path: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
    python_executable: str,
    device: str,
    promoted_motion_names: Sequence[str] = DEFAULT_PROMOTED_MOTIONS,
    local_improvement_pct: float = 3.0,
    dry_run: bool = True,
    candidate_set: str = "middle",
    assessment_mode: str = "promotion",
    sweep_args: Mapping[str, str] | None = None,
    sample_counts: Sequence[int] = (128,),
    seed_values: Sequence[int] = (0,),
) -> dict[str, Path]:
    """Write candidate rewards, a JSON experiment plan, and shell commands."""

    root = Path(output_root)
    names = tuple(candidate_names)
    local_improvement_multiplier_from_pct(local_improvement_pct)
    root.mkdir(parents=True, exist_ok=True)
    reward_paths = stage.write_candidate_reward_files(
        base_reward_weights_path,
        root / "reward_weights",
        candidate_names=names,
    )
    commands = plan_sweep_commands(
        output_root=root,
        candidate_names=names,
        motion_names=tuple(motion_names),
        python_executable=python_executable,
        device=device,
        sweep_args=sweep_args,
    )
    plan_path = root / "experiment_plan.json"
    commands_path = root / "planned_commands.sh"
    promoted_commands_path = root / "promoted_commands.sh"
    plan_payload = experiment_plan_payload(
        output_root=root,
        candidate_names=names,
        motion_names=tuple(motion_names),
        promoted_motion_names=tuple(promoted_motion_names),
        reward_paths=reward_paths,
        commands=commands,
        local_improvement_pct=local_improvement_pct,
        dry_run=dry_run,
        candidate_set=candidate_set,
        assessment_mode=assessment_mode,
        sweep_args=sweep_args,
        sample_counts=tuple(sample_counts),
        seed_values=tuple(seed_values),
    )
    plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n")
    commands_path.write_text(render_commands_script(commands, names))
    write_promoted_commands(
        output_root=root,
        guardrail_rows=(),
        candidate_names=names,
        screening_motion_name=tuple(motion_names)[0] if motion_names else "",
        promoted_motion_names=tuple(promoted_motion_names),
        python_executable=python_executable,
        device=device,
    )
    return {
        "experiment_plan": plan_path,
        "planned_commands": commands_path,
        "promoted_commands": promoted_commands_path,
        "reward_weights": root / "reward_weights",
    }


def experiment_plan_payload(
    *,
    output_root: Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
    promoted_motion_names: Sequence[str],
    reward_paths: dict[str, Path],
    commands: Sequence[Sequence[str]],
    local_improvement_pct: float,
    dry_run: bool,
    candidate_set: str = "middle",
    assessment_mode: str = "promotion",
    sweep_args: Mapping[str, str] | None = None,
    sample_counts: Sequence[int] = (128,),
    seed_values: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Return a deterministic JSON-serializable experiment plan."""

    local_improvement_multiplier = local_improvement_multiplier_from_pct(
        local_improvement_pct
    )
    selected_sweep_args = dict(FIXED_SWEEP_ARGS if sweep_args is None else sweep_args)
    return {
        "command_count": len(commands),
        "commands": [shlex.join(list(command)) for command in commands],
        "assessment_mode": assessment_mode,
        "candidate_set": candidate_set,
        "candidates": list(candidate_names),
        "dry_run": dry_run,
        "fixed_parameters": {
            "controls": int(_sweep_values(selected_sweep_args, "controls")[0]),
            "horizons": int(_sweep_values(selected_sweep_args, "horizons")[0]),
            "iterations": int(_sweep_values(selected_sweep_args, "iterations")[0]),
            "knot_counts": int(_sweep_values(selected_sweep_args, "knot_counts")[0]),
            "max_steps": int(_sweep_values(selected_sweep_args, "max_steps")[0]),
            "samples": _sample_counts_plan_value(sample_counts),
            "seeds": [int(value) for value in seed_values],
            "sigma_triplets": list(_sweep_values(selected_sweep_args, "sigma_triplets")),
        },
        "local_improvement_multiplier": local_improvement_multiplier,
        "local_improvement_pct": local_improvement_pct,
        "low_sample_sweep_script": str(LOW_SAMPLE_SWEEP_SCRIPT),
        "motions": list(motion_names),
        "output_root": str(output_root),
        "promoted_motions": list(promoted_motion_names),
        "reward_weights": {
            candidate_name: str(reward_paths[candidate_name])
            for candidate_name in candidate_names
        },
    }


def render_commands_script(
    commands: Sequence[Sequence[str]],
    candidate_names: Sequence[str],
) -> str:
    """Render planned commands as a deterministic shell script."""

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(SPIDER_ROOT))}",
        "",
    ]
    for candidate_name, command in zip(candidate_names, commands):
        lines.append(f"# candidate={candidate_name}")
        lines.append(shlex.join(list(command)))
        lines.append("")
    return "\n".join(lines)


def render_promoted_commands_script(
    commands: Sequence[Sequence[str]],
    candidate_names: Sequence[str],
    *,
    screening_motion_name: str,
    promoted_motion_names: Sequence[str],
) -> str:
    """Render promotion commands or a placeholder when screening is incomplete."""

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(SPIDER_ROOT))}",
        "",
    ]
    if not commands:
        motions = " ".join(promoted_motion_names)
        lines.extend(
            [
                "# No promoted candidates yet.",
                f"# Screening motion: {screening_motion_name or '<none>'}",
                f"# Promoted motions: {motions}",
                "# Run summarization after screening results exist to populate this file.",
                "",
            ]
        )
        return "\n".join(lines)

    for candidate_name, command in zip(candidate_names, commands):
        lines.append(
            f"# promoted_candidate={candidate_name} screening={screening_motion_name}"
        )
        lines.append(shlex.join(list(command)))
        lines.append("")
    return "\n".join(lines)


def write_guardrail_summary(
    *,
    output_root: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
    promoted_motion_names: Sequence[str] = DEFAULT_PROMOTED_MOTIONS,
    python_executable: str = str(DEFAULT_PYTHON_EXECUTABLE),
    device: str = "cuda:0",
    screening_motion_name: str | None = None,
    local_improvement_pct: float = 3.0,
) -> list[dict[str, str]]:
    """Collect per-candidate sweep summaries into guardrail_summary.csv."""

    root = Path(output_root)
    screening_motion = screening_motion_name or (motion_names[0] if motion_names else "")
    local_improvement_multiplier = local_improvement_multiplier_from_pct(
        local_improvement_pct
    )
    rows: list[dict[str, str]] = []
    control_durations = _control_durations(root)
    for candidate_name in candidate_names:
        _validate_candidate(candidate_name)
        summary_path = candidate_output_root(root, candidate_name) / "summary.csv"
        summary_rows = _read_summary_rows(summary_path)
        for motion_name in motion_names:
            source = _find_motion_row(summary_rows, motion_name)
            if source is None:
                rows.append(_planned_guardrail_row(candidate_name, motion_name, summary_path))
                continue
            rows.append(
                _guardrail_row_from_summary(
                    candidate_name,
                    motion_name,
                    summary_path,
                    source,
                    control_durations.get(motion_name),
                    local_improvement_multiplier=local_improvement_multiplier,
                )
            )

    output_path = root / "guardrail_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GUARDRAIL_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_promoted_commands(
        output_root=root,
        guardrail_rows=rows,
        candidate_names=candidate_names,
        screening_motion_name=screening_motion,
        promoted_motion_names=promoted_motion_names,
        python_executable=python_executable,
        device=device,
    )
    return rows


def write_frontier_summary(
    *,
    output_root: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
) -> list[dict[str, str]]:
    """Collect upper-bound candidate summaries into frontier_summary.csv."""

    root = Path(output_root)
    control_durations = _control_durations(root)
    sortable_rows: list[tuple[tuple[Any, ...], dict[str, str]]] = []
    source_index = 0
    for candidate_name in candidate_names:
        _validate_candidate(candidate_name)
        summary_path = candidate_output_root(root, candidate_name) / "summary.csv"
        summary_rows = _read_summary_rows(summary_path)
        for motion_name in motion_names:
            source = _find_motion_row(summary_rows, motion_name)
            if source is None:
                row = _planned_frontier_row(candidate_name, motion_name, summary_path)
                sort_key = _frontier_sort_key_for_planned(source_index)
            else:
                row, sort_key = _frontier_row_from_summary(
                    candidate_name,
                    motion_name,
                    summary_path,
                    source,
                    control_durations.get(motion_name),
                    source_index=source_index,
                )
            sortable_rows.append((sort_key, row))
            source_index += 1

    sortable_rows.sort(key=lambda item: item[0])
    rows = [row for _, row in sortable_rows]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)

    output_path = root / "frontier_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRONTIER_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_transfer_summary(
    *,
    output_root: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
) -> list[dict[str, str]]:
    """Collect transfer candidate summaries into transfer_summary.csv."""

    root = Path(output_root)
    control_durations = _control_durations(root)
    sortable_rows: list[tuple[tuple[Any, ...], dict[str, str]]] = []
    source_index = 0
    for candidate_name in candidate_names:
        _validate_candidate(candidate_name)
        summary_path = candidate_output_root(root, candidate_name) / "summary.csv"
        summary_rows = _read_summary_rows(summary_path)
        for motion_name in motion_names:
            source = _find_motion_row(summary_rows, motion_name)
            if source is None:
                row = _planned_transfer_row(candidate_name, motion_name, summary_path)
                sort_key = _transfer_sort_key_for_planned(source_index)
            else:
                row, sort_key = _transfer_row_from_summary(
                    candidate_name,
                    motion_name,
                    summary_path,
                    source,
                    control_durations.get(motion_name),
                    source_index=source_index,
                )
            sortable_rows.append((sort_key, row))
            source_index += 1

    sortable_rows.sort(key=lambda item: item[0])
    rows = [row for _, row in sortable_rows]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)

    output_path = root / "transfer_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRANSFER_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_pareto_summary(
    *,
    output_root: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
    sample_counts: Sequence[int],
    seed_values: Sequence[int],
) -> list[dict[str, str]]:
    """Collect repeated Pareto sample-budget groups into pareto_summary.csv."""

    root = Path(output_root)
    expected_seed_values = tuple(int(value) for value in seed_values)
    expected_seed_set = set(expected_seed_values)
    sortable_rows: list[tuple[tuple[Any, ...], dict[str, str]]] = []
    source_index = 0
    for candidate_name in candidate_names:
        _validate_candidate(candidate_name)
        summary_path = candidate_output_root(root, candidate_name) / "summary.csv"
        summary_rows = _read_summary_rows(summary_path)
        for motion_name in motion_names:
            for sample_count in sample_counts:
                source_rows = [
                    row
                    for row in summary_rows
                    if _motion_name(row) == motion_name
                    and _as_int(row.get("samples")) == sample_count
                    and _as_int(row.get("seed")) in expected_seed_set
                ]
                if source_rows:
                    seed_failure_labels = _pareto_seed_failure_labels(
                        source_rows,
                        expected_seed_values,
                    )
                    row, sort_key = _pareto_row_from_group(
                        candidate_name,
                        motion_name,
                        int(sample_count),
                        summary_path,
                        source_rows,
                        source_index=source_index,
                        seed_failure_labels=seed_failure_labels,
                    )
                else:
                    row = _planned_pareto_row(
                        candidate_name,
                        motion_name,
                        int(sample_count),
                        summary_path,
                    )
                    sort_key = _pareto_sort_key_for_planned(
                        int(sample_count),
                        source_index,
                    )
                sortable_rows.append((sort_key, row))
                source_index += 1

    sortable_rows.sort(key=lambda item: item[0])
    rows = [row for _, row in sortable_rows]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)

    output_path = root / "pareto_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PARETO_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_pareto_decision_report(root, rows)
    return rows


def write_pareto_decision_report(
    output_root: str | Path,
    pareto_rows: Sequence[dict[str, str]],
) -> Path:
    """Write a deterministic Markdown conclusion from pareto_summary rows."""

    root = Path(output_root)
    requested_samples = _pareto_sample_values(pareto_rows)
    completed_rows = [
        row
        for row in pareto_rows
        if row.get("status") == "completed" and _as_int(row.get("samples")) is not None
    ]
    usable_completed_rows = [
        row
        for row in completed_rows
        if row.get("pareto_class")
        in {
            "baseline_close",
            "promising_budget",
            "stable_but_low_quality",
            "unstable",
        }
    ]
    completed_classes = {row.get("pareto_class", "") for row in completed_rows}

    baseline_rows = [
        row for row in usable_completed_rows if row.get("pareto_class") == "baseline_close"
    ]
    promising_rows = [
        row
        for row in usable_completed_rows
        if row.get("pareto_class") == "promising_budget"
    ]
    unstable_rows = [
        row for row in usable_completed_rows if row.get("pareto_class") == "unstable"
    ]
    stable_low_rows = [
        row
        for row in usable_completed_rows
        if row.get("pareto_class") == "stable_but_low_quality"
    ]

    if baseline_rows:
        conclusion = "sample_budget_likely"
        sample = _pareto_sample_values(baseline_rows)[0]
        next_step = (
            f"Promote sample {sample} for transfer/visual review as the fastest "
            "baseline_close sample."
        )
    elif promising_rows:
        conclusion = "sample_budget_partial"
        samples = _pareto_sample_values(promising_rows)[:2]
        sample_text = ", ".join(str(sample) for sample in samples)
        next_step = f"Run transfer checks for promising sample(s): {sample_text}."
    elif unstable_rows:
        conclusion = "stability_likely"
        next_step = "Do stability/acceptance repair before visual promotion."
    elif (
        stable_low_rows
        and completed_classes == {"stable_but_low_quality"}
        and requested_samples
        and requested_samples[-1] in set(_pareto_sample_values(stable_low_rows))
    ):
        conclusion = "reward_gating_likely"
        next_step = (
            "Stop sample escalation and design reward/gating repair from best_s128."
        )
    else:
        conclusion = "pending"
        next_step = "rollout execution is still required before promotion decisions."

    path = root / "pareto_decision.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Pareto Decision Report",
                "",
                f"Conclusion: {conclusion}",
                "",
                f"Recommended next step: {next_step}",
                "",
            ]
        )
    )
    return path


def write_promoted_commands(
    *,
    output_root: str | Path,
    guardrail_rows: Sequence[dict[str, str]],
    candidate_names: Sequence[str],
    screening_motion_name: str,
    promoted_motion_names: Sequence[str],
    python_executable: str,
    device: str,
) -> Path:
    """Write second-stage commands for candidates that passed screening."""

    root = Path(output_root)
    promoted_names = promoted_candidate_names(
        guardrail_rows,
        candidate_names=candidate_names,
        screening_motion_name=screening_motion_name,
    )
    commands = plan_sweep_commands(
        output_root=root,
        candidate_names=promoted_names,
        motion_names=tuple(promoted_motion_names),
        python_executable=python_executable,
        device=device,
        output_group="promoted",
    )
    path = root / "promoted_commands.sh"
    path.write_text(
        render_promoted_commands_script(
            commands,
            promoted_names,
            screening_motion_name=screening_motion_name,
            promoted_motion_names=tuple(promoted_motion_names),
        )
    )
    return path


def promoted_candidate_names(
    guardrail_rows: Sequence[dict[str, str]],
    *,
    candidate_names: Sequence[str],
    screening_motion_name: str,
    limit: int = 2,
) -> tuple[str, ...]:
    """Select passed screening candidates in deterministic candidate order."""

    passed = {
        row["candidate"]
        for row in guardrail_rows
        if row.get("motion") == screening_motion_name
        and row.get("status") == "passed"
        and row.get("passed") == "true"
    }
    return tuple(candidate for candidate in candidate_names if candidate in passed)[:limit]


def selected_candidate_names(
    candidate_names: Sequence[str] | None,
    *,
    include_contact: bool = False,
    candidate_set: str = "middle",
) -> tuple[str, ...]:
    """Resolve requested candidates and optionally append the contact variant."""

    if candidate_names is not None:
        names = tuple(candidate_names)
    elif candidate_set == "middle":
        names = DEFAULT_SCREENING_CANDIDATES
    elif candidate_set == "upper-bound":
        names = stage.UPPER_BOUND_CANDIDATE_NAMES
    elif candidate_set == "transfer":
        names = stage.TRANSFER_CANDIDATE_NAMES
    elif candidate_set == "recovery":
        names = stage.RECOVERY_CANDIDATE_NAMES
    elif candidate_set == "pareto":
        names = stage.PARETO_CANDIDATE_NAMES
    else:
        raise ValueError(f"Unknown candidate set {candidate_set!r}.")
    if include_contact and candidate_names is None and candidate_set in {
        "transfer",
        "recovery",
    }:
        raise ValueError("--include-contact cannot alter transfer or recovery sets.")
    if include_contact and CONTACT_CANDIDATE not in names:
        names = (*names, CONTACT_CANDIDATE)
    for candidate_name in names:
        _validate_candidate(candidate_name)
    return names


def local_improvement_multiplier_from_pct(percent: float) -> float:
    if not math.isfinite(percent) or percent < 0.0 or percent >= 100.0:
        raise ValueError("local improvement pct must be finite and in [0, 100).")
    return 1.0 - percent / 100.0


def local_improvement_pct_arg(value: str) -> float:
    try:
        percent = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "local improvement pct must be a finite number in [0, 100)."
        ) from exc
    if not math.isfinite(percent) or percent < 0.0 or percent >= 100.0:
        raise argparse.ArgumentTypeError(
            "local improvement pct must be a finite number in [0, 100)."
        )
    return percent


def _int_arg(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc


def candidate_reward_path(output_root: str | Path, candidate_name: str) -> Path:
    return Path(output_root) / "reward_weights" / f"{candidate_name}.json"


def candidate_output_root(
    output_root: str | Path,
    candidate_name: str,
    *,
    output_group: str = "candidates",
) -> Path:
    return Path(output_root) / output_group / candidate_name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arg_tokens = tuple(sys.argv[1:] if argv is None else argv)
    assessment_mode_provided = _option_was_provided(
        arg_tokens,
        "--assessment-mode",
    )
    motions_provided = _option_was_provided(arg_tokens, "--motions")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Write deterministic artifacts without running subprocesses. Default.",
    )
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Run the planned low-sample sweep commands sequentially.",
    )
    parser.set_defaults(execute=False)

    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--base-reward-weights",
        default=str(DEFAULT_REWARD_WEIGHTS),
        help="Base v14 method-specific reward JSON used to create candidates.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="Candidate names to plan. Defaults to the next-stage screening matrix.",
    )
    parser.add_argument(
        "--candidate-set",
        choices=("middle", "upper-bound", "transfer", "recovery", "pareto"),
        default="middle",
        help="Default candidate set to use when --candidates is omitted.",
    )
    parser.add_argument(
        "--assessment-mode",
        choices=("promotion", "upper-bound", "transfer", "pareto"),
        default="promotion",
        help="Summary mode to run after execution or in --summarize-only.",
    )
    parser.add_argument("--motions", nargs="+", default=list(DEFAULT_MOTIONS))
    parser.add_argument("--samples", nargs="+", type=_int_arg, default=None)
    parser.add_argument("--seeds", nargs="+", type=_int_arg, default=None)
    parser.add_argument(
        "--promoted-motions",
        nargs="+",
        default=list(DEFAULT_PROMOTED_MOTIONS),
    )
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON_EXECUTABLE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--include-contact",
        action="store_true",
        help="Append the legacy L150 contact candidate to the selected candidate set.",
    )
    parser.add_argument(
        "--local-improvement-pct",
        type=local_improvement_pct_arg,
        default=3.0,
        help="Required local metric improvement percentage for jump screening.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only collect guardrail_summary.csv from existing candidate summaries.",
    )
    args = parser.parse_args(arg_tokens)
    if not assessment_mode_provided and args.candidate_set == "upper-bound":
        args.assessment_mode = "upper-bound"
    if not assessment_mode_provided and args.candidate_set == "transfer":
        args.assessment_mode = "transfer"
    if not assessment_mode_provided and args.candidate_set == "pareto":
        args.assessment_mode = "pareto"
    if not motions_provided and args.candidate_set == "transfer":
        args.motions = list(DEFAULT_TRANSFER_MOTIONS)
    if not motions_provided and args.candidate_set == "pareto":
        args.motions = ["jump"]
    if args.samples is None:
        args.samples = (
            tuple(stage.PARETO_SAMPLE_COUNTS)
            if args.candidate_set == "pareto"
            else (128,)
        )
    else:
        args.samples = tuple(args.samples)
    if args.seeds is None:
        args.seeds = (
            tuple(stage.PARETO_REPEAT_SEEDS)
            if args.candidate_set == "pareto"
            else (0,)
        )
    else:
        args.seeds = tuple(args.seeds)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser()
    candidate_names = selected_candidate_names(
        args.candidates,
        include_contact=args.include_contact,
        candidate_set=args.candidate_set,
    )
    motion_names = tuple(args.motions)
    sample_counts = tuple(args.samples)
    seed_values = tuple(args.seeds)
    sweep_args = sweep_args_for_stage(sample_counts, seed_values)
    local_improvement_multiplier_from_pct(args.local_improvement_pct)

    if args.summarize_only:
        if args.assessment_mode == "pareto":
            rows = write_pareto_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
                sample_counts=sample_counts,
                seed_values=seed_values,
            )
            print(f"Wrote {len(rows)} rows to {output_root / 'pareto_summary.csv'}")
        elif args.assessment_mode == "transfer":
            rows = write_transfer_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
            )
            print(f"Wrote {len(rows)} rows to {output_root / 'transfer_summary.csv'}")
        elif args.assessment_mode == "upper-bound":
            rows = write_frontier_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
            )
            print(f"Wrote {len(rows)} rows to {output_root / 'frontier_summary.csv'}")
        else:
            rows = write_guardrail_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
                promoted_motion_names=tuple(args.promoted_motions),
                python_executable=args.python_executable,
                device=args.device,
                local_improvement_pct=args.local_improvement_pct,
            )
            print(f"Wrote {len(rows)} rows to {output_root / 'guardrail_summary.csv'}")
        return 0

    artifacts = write_dry_run_artifacts(
        output_root=output_root,
        base_reward_weights_path=Path(args.base_reward_weights).expanduser(),
        candidate_names=candidate_names,
        motion_names=motion_names,
        promoted_motion_names=tuple(args.promoted_motions),
        python_executable=args.python_executable,
        device=args.device,
        local_improvement_pct=args.local_improvement_pct,
        dry_run=not args.execute,
        candidate_set=args.candidate_set,
        assessment_mode=args.assessment_mode,
        sweep_args=sweep_args,
        sample_counts=sample_counts,
        seed_values=seed_values,
    )
    commands = plan_sweep_commands(
        output_root=output_root,
        candidate_names=candidate_names,
        motion_names=motion_names,
        python_executable=args.python_executable,
        device=args.device,
        sweep_args=sweep_args,
    )

    if not args.execute:
        if args.assessment_mode == "pareto":
            rows = write_pareto_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
                sample_counts=sample_counts,
                seed_values=seed_values,
            )
            print(f"Wrote {len(rows)} planned rows to {output_root / 'pareto_summary.csv'}")
        elif args.assessment_mode == "transfer":
            rows = write_transfer_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
            )
            print(
                f"Wrote {len(rows)} planned rows to {output_root / 'transfer_summary.csv'}"
            )
        elif args.assessment_mode == "upper-bound":
            rows = write_frontier_summary(
                output_root=output_root,
                candidate_names=candidate_names,
                motion_names=motion_names,
            )
            print(f"Wrote {len(rows)} planned rows to {output_root / 'frontier_summary.csv'}")
        print(f"Wrote experiment plan: {artifacts['experiment_plan']}")
        print(f"Wrote planned commands: {artifacts['planned_commands']}")
        print(f"Wrote reward weights under: {artifacts['reward_weights']}")
        print("Pass --execute to run the planned sweeps.")
        return 0

    exit_codes: list[int] = []
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}")
        completed = subprocess.run(command, cwd=str(SPIDER_ROOT))
        exit_codes.append(completed.returncode)

    if args.assessment_mode == "transfer":
        rows = write_transfer_summary(
            output_root=output_root,
            candidate_names=candidate_names,
            motion_names=motion_names,
        )
        print(f"Wrote {len(rows)} rows to {output_root / 'transfer_summary.csv'}")
    elif args.assessment_mode == "pareto":
        rows = write_pareto_summary(
            output_root=output_root,
            candidate_names=candidate_names,
            motion_names=motion_names,
            sample_counts=sample_counts,
            seed_values=seed_values,
        )
        print(f"Wrote {len(rows)} rows to {output_root / 'pareto_summary.csv'}")
    elif args.assessment_mode == "upper-bound":
        rows = write_frontier_summary(
            output_root=output_root,
            candidate_names=candidate_names,
            motion_names=motion_names,
        )
        print(f"Wrote {len(rows)} rows to {output_root / 'frontier_summary.csv'}")
    else:
        rows = write_guardrail_summary(
            output_root=output_root,
            candidate_names=candidate_names,
            motion_names=motion_names,
            promoted_motion_names=tuple(args.promoted_motions),
            python_executable=args.python_executable,
            device=args.device,
            local_improvement_pct=args.local_improvement_pct,
        )
        print(f"Wrote {len(rows)} rows to {output_root / 'guardrail_summary.csv'}")
    return 0 if all(code == 0 for code in exit_codes) else 1


def _guardrail_row_from_summary(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
    source: dict[str, str],
    control_duration_sec: float | None,
    *,
    local_improvement_multiplier: float,
) -> dict[str, str]:
    status = source.get("status", "")
    metrics = _prefixed_values(source, "metric_")
    mpc = _prefixed_values(source, "mpc_")
    if status != "ok":
        return _incomplete_guardrail_row(candidate_name, motion_name, summary_path, status)
    if _missing_required(metrics, REQUIRED_METRICS) or _missing_required(mpc, REQUIRED_MPC):
        return _incomplete_guardrail_row(candidate_name, motion_name, summary_path, "missing")

    duration_sec = _as_float(source.get("duration_sec"))
    result = stage.assess_candidate(
        motion_name,
        metrics,
        mpc,
        duration_sec=duration_sec,
        control_duration_sec=control_duration_sec,
        local_improvement_multiplier=local_improvement_multiplier,
    )
    passed = bool(result["passed"])
    row = _empty_guardrail_row(candidate_name, motion_name, summary_path)
    row.update(
        {
            "status": "passed" if passed else "failed",
            "passed": _bool_text(passed),
            "failure_labels": ",".join(result["failure_labels"]),
            "improved_local_count": str(result["improved_local_count"]),
        }
    )
    for key in (
        "global_guardrail_passed",
        "smooth_guardrail_passed",
        "contact_guardrail_passed",
        "score_guardrail_passed",
        "local_guardrail_passed",
        "mpc_guardrail_passed",
        "runtime_guardrail_passed",
    ):
        row[key] = _bool_text(bool(result[key]))
    return row


def _frontier_row_from_summary(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
    source: dict[str, str],
    control_duration_sec: float | None,
    *,
    source_index: int,
) -> tuple[dict[str, str], tuple[Any, ...]]:
    status = source.get("status", "")
    metrics = _prefixed_values(source, "metric_")
    mpc = _prefixed_values(source, "mpc_")
    if (
        status != "ok"
        or _missing_required(metrics, REQUIRED_METRICS)
        or _missing_required(mpc, REQUIRED_MPC)
    ):
        incomplete_status = status if status != "ok" else "missing"
        row = _incomplete_frontier_row(
            candidate_name,
            motion_name,
            summary_path,
            incomplete_status,
        )
        return row, _frontier_sort_key_for_planned(source_index)

    duration_sec = _as_float(source.get("duration_sec"))
    result = stage.assess_upper_bound_candidate(
        motion_name,
        metrics,
        mpc,
        duration_sec=duration_sec,
        control_duration_sec=control_duration_sec,
    )
    row = _empty_frontier_row(candidate_name, motion_name, summary_path)
    row.update(
        {
            "status": "completed",
            "frontier_class": str(result["frontier_class"]),
            "failure_labels": ",".join(result["failure_labels"]),
            "hard_guardrail_passed": _bool_text(bool(result["hard_guardrail_passed"])),
            "score_relaxed_passed": _bool_text(bool(result["score_relaxed_passed"])),
            "global_relaxed_passed": _bool_text(bool(result["global_relaxed_passed"])),
            "mpc_guardrail_passed": _bool_text(bool(result["mpc_guardrail_passed"])),
            "runtime_guardrail_passed": _bool_text(bool(result["runtime_guardrail_passed"])),
            "local_count_3": str(result["local_count_3"]),
            "local_count_5": str(result["local_count_5"]),
            "local_count_8": str(result["local_count_8"]),
            "local_composite_improvement": _float_text(
                result["local_composite_improvement"]
            ),
            "score_delta": _float_text(result["score_delta"]),
            "max_global_ratio": _float_text(result["max_global_ratio"]),
            "contact_delta": _float_text(result["contact_delta"]),
            "control_delta_regression": _float_text(
                result["control_delta_regression"]
            ),
            "joint_acc_regression": _float_text(result["joint_acc_regression"]),
        }
    )
    return row, _frontier_sort_key_for_result(result, source_index)


def _transfer_row_from_summary(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
    source: dict[str, str],
    control_duration_sec: float | None,
    *,
    source_index: int,
) -> tuple[dict[str, str], tuple[Any, ...]]:
    status = source.get("status", "")
    metrics = _prefixed_values(source, "metric_")
    mpc = _prefixed_values(source, "mpc_")
    if (
        status != "ok"
        or _missing_required(metrics, REQUIRED_METRICS)
        or _missing_required(mpc, REQUIRED_MPC)
    ):
        incomplete_status = status if status != "ok" else "missing"
        row = _incomplete_transfer_row(
            candidate_name,
            motion_name,
            summary_path,
            incomplete_status,
        )
        return row, _transfer_sort_key_for_planned(source_index)

    duration_sec = _as_float(source.get("duration_sec"))
    result = stage.assess_transfer_candidate(
        motion_name,
        metrics,
        mpc,
        duration_sec=duration_sec,
        control_duration_sec=control_duration_sec,
    )
    row = _empty_transfer_row(candidate_name, motion_name, summary_path)
    row.update(
        {
            "status": "completed",
            "transfer_class": str(result["transfer_class"]),
            "failure_labels": ",".join(result["failure_labels"]),
            "score_guardrail_passed": _bool_text(
                bool(result["score_guardrail_passed"])
            ),
            "global_guardrail_passed": _bool_text(
                bool(result["global_guardrail_passed"])
            ),
            "smooth_guardrail_passed": _bool_text(
                bool(result["smooth_guardrail_passed"])
            ),
            "contact_guardrail_passed": _bool_text(
                bool(result["contact_guardrail_passed"])
            ),
            "local_transfer_passed": _bool_text(
                bool(result["local_transfer_passed"])
            ),
            "mpc_guardrail_passed": _bool_text(bool(result["mpc_guardrail_passed"])),
            "runtime_guardrail_passed": _bool_text(
                bool(result["runtime_guardrail_passed"])
            ),
            "local_count_3": str(result["local_count_3"]),
            "max_local_regression": _float_text(result["max_local_regression"]),
            "score_delta": _float_text(result["score_delta"]),
            "max_global_ratio": _float_text(result["max_global_ratio"]),
            "contact_delta": _float_text(result["contact_delta"]),
            "control_delta_regression": _float_text(
                result["control_delta_regression"]
            ),
            "joint_acc_regression": _float_text(result["joint_acc_regression"]),
        }
    )
    return row, _transfer_sort_key_for_result(result, source_index)


def _pareto_row_from_group(
    candidate_name: str,
    motion_name: str,
    sample_count: int,
    summary_path: Path,
    source_rows: Sequence[dict[str, str]],
    *,
    source_index: int,
    seed_failure_labels: Sequence[str] = (),
) -> tuple[dict[str, str], tuple[Any, ...]]:
    result = dict(stage.assess_pareto_group(motion_name, source_rows))
    if seed_failure_labels:
        failure_labels = list(result["failure_labels"])
        for label in seed_failure_labels:
            if label not in failure_labels:
                failure_labels.append(label)
        result["failure_labels"] = failure_labels
        result["pareto_class"] = "unstable"
    row = _empty_pareto_row(candidate_name, motion_name, sample_count, summary_path)
    row.update(
        {
            "status": "completed",
            "pareto_class": str(result["pareto_class"]),
            "failure_labels": ",".join(result["failure_labels"]),
            "repeat_count": str(result["repeat_count"]),
            "ok_count": str(result["ok_count"]),
            "success_count": str(result["success_count"]),
            "accepted_count": str(result["accepted_count"]),
            "fallback_count": str(result["fallback_count"]),
            "full_run_count": str(result["full_run_count"]),
            "accepted_windows_count": str(result["accepted_windows_count"]),
            "duration_sec_mean": _float_text(result["duration_sec_mean"]),
            "score_mean": _float_text(result["score_mean"]),
            "root_pos_error_mean_mean": _float_text(
                result["root_pos_error_mean_mean"]
            ),
            "body_global_pos_error_mean_mean": _float_text(
                result["body_global_pos_error_mean_mean"]
            ),
            "ee_global_pos_error_mean_mean": _float_text(
                result["ee_global_pos_error_mean_mean"]
            ),
            "ee_local_pos_error_mean_mean": _float_text(
                result["ee_local_pos_error_mean_mean"]
            ),
            "contact_mismatch_rate_mean": _float_text(
                result["contact_mismatch_rate_mean"]
            ),
            "control_delta_mean_mean": _float_text(
                result["control_delta_mean_mean"]
            ),
            "joint_acc_mean_mean": _float_text(result["joint_acc_mean_mean"]),
            "score_delta_vs_best_mean": _float_text(
                result["score_delta_vs_best_mean"]
            ),
            "score_delta_vs_baseline_mean": _float_text(
                result["score_delta_vs_baseline_mean"]
            ),
            "max_global_ratio_vs_best_mean": _float_text(
                result["max_global_ratio_vs_best_mean"]
            ),
            "max_global_ratio_vs_baseline_mean": _float_text(
                result["max_global_ratio_vs_baseline_mean"]
            ),
        }
    )
    return row, _pareto_sort_key_for_result(result, sample_count, source_index)


def _pareto_seed_failure_labels(
    source_rows: Sequence[dict[str, str]],
    expected_seed_values: Sequence[int],
) -> list[str]:
    expected_seed_set = set(expected_seed_values)
    seed_counts = {seed: 0 for seed in expected_seed_set}
    for row in source_rows:
        seed = _as_int(row.get("seed"))
        if seed in seed_counts:
            seed_counts[seed] += 1

    failure_labels: list[str] = []
    if any(count == 0 for count in seed_counts.values()):
        failure_labels.append("missing_seeds")
    if any(count > 1 for count in seed_counts.values()):
        failure_labels.append("duplicate_seeds")
    return failure_labels


def _pareto_sample_values(rows: Sequence[dict[str, str]]) -> list[int]:
    return sorted(
        {
            sample
            for row in rows
            for sample in [_as_int(row.get("samples"))]
            if sample is not None
        }
    )


def _planned_guardrail_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
) -> dict[str, str]:
    return _incomplete_guardrail_row(candidate_name, motion_name, summary_path, "planned")


def _incomplete_guardrail_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
    status: str,
) -> dict[str, str]:
    row = _empty_guardrail_row(candidate_name, motion_name, summary_path)
    row["status"] = _guardrail_status(status)
    row["passed"] = "false"
    return row


def _empty_guardrail_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
) -> dict[str, str]:
    return {
        "candidate": candidate_name,
        "motion": motion_name,
        "status": "",
        "passed": "false",
        "failure_labels": "",
        "improved_local_count": "",
        "global_guardrail_passed": "",
        "smooth_guardrail_passed": "",
        "contact_guardrail_passed": "",
        "score_guardrail_passed": "",
        "local_guardrail_passed": "",
        "mpc_guardrail_passed": "",
        "runtime_guardrail_passed": "",
        "summary_csv": str(summary_path),
    }


def _planned_frontier_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
) -> dict[str, str]:
    return _incomplete_frontier_row(candidate_name, motion_name, summary_path, "planned")


def _incomplete_frontier_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
    status: str,
) -> dict[str, str]:
    normalized_status = _guardrail_status(status)
    row = _empty_frontier_row(candidate_name, motion_name, summary_path)
    row.update(
        {
            "status": normalized_status,
            "frontier_class": "invalid",
            "failure_labels": normalized_status,
            "hard_guardrail_passed": "false",
            "score_relaxed_passed": "false",
            "global_relaxed_passed": "false",
            "mpc_guardrail_passed": "false",
            "runtime_guardrail_passed": "false",
            "local_count_3": "0",
            "local_count_5": "0",
            "local_count_8": "0",
        }
    )
    return row


def _empty_frontier_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
) -> dict[str, str]:
    return {
        "rank": "",
        "candidate": candidate_name,
        "motion": motion_name,
        "status": "",
        "frontier_class": "",
        "failure_labels": "",
        "hard_guardrail_passed": "",
        "score_relaxed_passed": "",
        "global_relaxed_passed": "",
        "mpc_guardrail_passed": "",
        "runtime_guardrail_passed": "",
        "local_count_3": "",
        "local_count_5": "",
        "local_count_8": "",
        "local_composite_improvement": "",
        "score_delta": "",
        "max_global_ratio": "",
        "contact_delta": "",
        "control_delta_regression": "",
        "joint_acc_regression": "",
        "summary_csv": str(summary_path),
    }


def _planned_transfer_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
) -> dict[str, str]:
    return _incomplete_transfer_row(candidate_name, motion_name, summary_path, "planned")


def _incomplete_transfer_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
    status: str,
) -> dict[str, str]:
    normalized_status = _guardrail_status(status)
    row = _empty_transfer_row(candidate_name, motion_name, summary_path)
    row.update(
        {
            "status": normalized_status,
            "transfer_class": "invalid",
            "failure_labels": normalized_status,
            "score_guardrail_passed": "false",
            "global_guardrail_passed": "false",
            "smooth_guardrail_passed": "false",
            "contact_guardrail_passed": "false",
            "local_transfer_passed": "false",
            "mpc_guardrail_passed": "false",
            "runtime_guardrail_passed": "false",
            "local_count_3": "0",
        }
    )
    return row


def _empty_transfer_row(
    candidate_name: str,
    motion_name: str,
    summary_path: Path,
) -> dict[str, str]:
    return {
        "rank": "",
        "candidate": candidate_name,
        "motion": motion_name,
        "status": "",
        "transfer_class": "",
        "failure_labels": "",
        "score_guardrail_passed": "",
        "global_guardrail_passed": "",
        "smooth_guardrail_passed": "",
        "contact_guardrail_passed": "",
        "local_transfer_passed": "",
        "mpc_guardrail_passed": "",
        "runtime_guardrail_passed": "",
        "local_count_3": "",
        "max_local_regression": "",
        "score_delta": "",
        "max_global_ratio": "",
        "contact_delta": "",
        "control_delta_regression": "",
        "joint_acc_regression": "",
        "summary_csv": str(summary_path),
    }


def _planned_pareto_row(
    candidate_name: str,
    motion_name: str,
    sample_count: int,
    summary_path: Path,
) -> dict[str, str]:
    row = _empty_pareto_row(candidate_name, motion_name, sample_count, summary_path)
    row.update(
        {
            "status": "planned",
            "pareto_class": "invalid",
            "failure_labels": "planned",
            "repeat_count": "0",
            "ok_count": "0",
            "success_count": "0",
            "accepted_count": "0",
            "fallback_count": "0",
            "full_run_count": "0",
            "accepted_windows_count": "0",
        }
    )
    return row


def _empty_pareto_row(
    candidate_name: str,
    motion_name: str,
    sample_count: int,
    summary_path: Path,
) -> dict[str, str]:
    return {
        "rank": "",
        "candidate": candidate_name,
        "motion": motion_name,
        "samples": str(sample_count),
        "status": "",
        "pareto_class": "",
        "failure_labels": "",
        "repeat_count": "",
        "ok_count": "",
        "success_count": "",
        "accepted_count": "",
        "fallback_count": "",
        "full_run_count": "",
        "accepted_windows_count": "",
        "duration_sec_mean": "",
        "score_mean": "",
        "root_pos_error_mean_mean": "",
        "body_global_pos_error_mean_mean": "",
        "ee_global_pos_error_mean_mean": "",
        "ee_local_pos_error_mean_mean": "",
        "contact_mismatch_rate_mean": "",
        "control_delta_mean_mean": "",
        "joint_acc_mean_mean": "",
        "score_delta_vs_best_mean": "",
        "score_delta_vs_baseline_mean": "",
        "max_global_ratio_vs_best_mean": "",
        "max_global_ratio_vs_baseline_mean": "",
        "summary_csv": str(summary_path),
    }


def _frontier_sort_key_for_result(
    result: dict[str, Any],
    source_index: int,
) -> tuple[Any, ...]:
    frontier_class = str(result["frontier_class"])
    return (
        0,
        FRONTIER_CLASS_RANK.get(frontier_class, FRONTIER_CLASS_RANK["invalid"]),
        -int(result["local_count_8"]),
        -int(result["local_count_5"]),
        -int(result["local_count_3"]),
        -float(result["local_composite_improvement"]),
        -float(result["score_delta"]),
        float(result["max_global_ratio"]),
        source_index,
    )


def _frontier_sort_key_for_planned(source_index: int) -> tuple[Any, ...]:
    return (
        1,
        FRONTIER_CLASS_RANK["invalid"],
        0,
        0,
        0,
        0.0,
        0.0,
        float("inf"),
        source_index,
    )


def _transfer_sort_key_for_result(
    result: dict[str, Any],
    source_index: int,
) -> tuple[Any, ...]:
    transfer_class = str(result["transfer_class"])
    return (
        TRANSFER_CLASS_RANK.get(transfer_class, TRANSFER_CLASS_RANK["invalid"]),
        -int(result["local_count_3"]),
        float(result["max_local_regression"]),
        -float(result["score_delta"]),
        float(result["max_global_ratio"]),
        source_index,
    )


def _transfer_sort_key_for_planned(source_index: int) -> tuple[Any, ...]:
    return (
        TRANSFER_CLASS_RANK["invalid"],
        0,
        float("inf"),
        0.0,
        float("inf"),
        source_index,
    )


def _pareto_sort_key_for_result(
    result: dict[str, Any],
    sample_count: int,
    source_index: int,
) -> tuple[Any, ...]:
    pareto_class = str(result["pareto_class"])
    return (
        PARETO_CLASS_RANK.get(pareto_class, PARETO_CLASS_RANK["invalid"]),
        -_sortable_float(result["score_delta_vs_baseline_mean"], float("-inf")),
        -_sortable_float(result["score_delta_vs_best_mean"], float("-inf")),
        _sortable_float(result["max_global_ratio_vs_baseline_mean"], float("inf")),
        sample_count,
        source_index,
    )


def _pareto_sort_key_for_planned(
    sample_count: int,
    source_index: int,
) -> tuple[Any, ...]:
    return (
        PARETO_CLASS_RANK["invalid"],
        float("inf"),
        float("inf"),
        float("inf"),
        sample_count,
        source_index,
    )


def _control_durations(output_root: Path) -> dict[str, float]:
    summary_path = candidate_output_root(output_root, "jg_s128_v14_control") / "summary.csv"
    durations: dict[str, float] = {}
    for row in _read_summary_rows(summary_path):
        if row.get("status") != "ok":
            continue
        motion = _motion_name(row)
        duration_sec = _as_float(row.get("duration_sec"))
        if motion and duration_sec is not None:
            durations[motion] = duration_sec
    return durations


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _find_motion_row(
    rows: Sequence[dict[str, str]],
    motion_name: str,
) -> dict[str, str] | None:
    for row in rows:
        if _motion_name(row) == motion_name:
            return row
    return None


def _motion_name(row: dict[str, str]) -> str:
    return row.get("motion_name") or row.get("motion") or ""


def _prefixed_values(row: dict[str, str], prefix: str) -> dict[str, str]:
    return {
        key[len(prefix) :]: value
        for key, value in row.items()
        if key.startswith(prefix) and value not in (None, "")
    }


def _missing_required(values: dict[str, str], required_keys: Sequence[str]) -> bool:
    return any(key not in values or values[key] == "" for key in required_keys)


def _sweep_values(sweep_args: Mapping[str, str], key: str) -> tuple[str, ...]:
    return tuple(str(sweep_args[key]).split())


def _sample_counts_plan_value(sample_counts: Sequence[int]) -> list[int]:
    return [int(value) for value in sample_counts]


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _float_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"


def _sortable_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _guardrail_status(status: str) -> str:
    if status in ("", "missing_metrics"):
        return "missing"
    return status


def _option_was_provided(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def _validate_candidate(candidate_name: str) -> None:
    if candidate_name not in stage.CANDIDATE_DEFINITIONS:
        raise ValueError(f"Unknown candidate {candidate_name!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
