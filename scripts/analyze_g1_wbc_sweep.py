#!/usr/bin/env python3
"""Rank low-sample G1 WBC sweep summary.csv files."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_GROUP_COLUMNS = (
    "samples",
    "iterations",
    "horizon",
    "control",
    "knot_count",
    "root_pos_sigma",
    "root_rot_sigma",
    "joint_sigma",
)
LATENCY_COLUMNS = ("duration_sec", "wall_time_sec", "latency_sec")
HIGHER_IS_BETTER_DELTAS = {
    "score",
    "success",
    "final_scores_mean",
    "final_scores_max",
    "final_candidate_score",
}
OK_STATUSES = {"", "ok", "success", "succeeded"}
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class SweepRow:
    """One CSV row with source location metadata."""

    source: Path
    line_number: int
    values: dict[str, str]


@dataclass
class CandidateStats:
    """Aggregated stats for one candidate configuration."""

    key: tuple[tuple[str, str], ...]
    rows: int = 0
    ok_rows: int = 0
    failed_rows: int = 0
    fallback_rows: int = 0
    latency_values: list[float] = field(default_factory=list)
    quality_values: list[float] = field(default_factory=list)
    quality_terms: int = 0
    sources: set[Path] = field(default_factory=set)
    motions: set[str] = field(default_factory=set)

    def add(self, row: SweepRow, *, latency: float | None, quality: float | None, quality_terms: int) -> None:
        row_ok = is_ok(row.values)
        self.rows += 1
        self.sources.add(row.source)
        motion = row.values.get("motion_name") or row.values.get("motion")
        if motion:
            self.motions.add(motion)
        if row_ok:
            self.ok_rows += 1
        else:
            self.failed_rows += 1
        if is_fallback(row.values):
            self.fallback_rows += 1
        if row_ok and latency is not None:
            self.latency_values.append(latency)
        if row_ok and quality is not None:
            self.quality_values.append(quality)
            self.quality_terms += quality_terms

    @property
    def mean_latency(self) -> float | None:
        return mean_or_none(self.latency_values)

    @property
    def mean_quality_delta(self) -> float | None:
        return mean_or_none(self.quality_values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_csv",
        nargs="+",
        type=Path,
        help="One or more low-sample sweep summary.csv files.",
    )
    parser.add_argument("--top", type=positive_int, default=10)
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help=(
            "Candidate columns to aggregate over. Missing columns are ignored; "
            "pass one or more column names."
        ),
    )
    parser.add_argument(
        "--failure-limit",
        type=nonnegative_int,
        default=20,
        help="Maximum failure/fallback rows to print. Use 0 to hide details.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, warnings = load_rows(args.summary_csv)
    if not rows:
        print("No rows loaded.")
        print_warnings(warnings)
        return 1

    column_names = all_column_names(rows)
    group_columns = [name for name in args.group_by if name in column_names]
    missing_group_columns = [name for name in args.group_by if name not in column_names]
    if missing_group_columns:
        warnings.append(
            "Ignoring missing group columns: " + ", ".join(missing_group_columns)
        )
    if not group_columns:
        warnings.append("No requested group columns found; ranking individual CSV rows.")

    latency_column = first_present(LATENCY_COLUMNS, column_names)
    if latency_column is None:
        warnings.append(
            "No latency column found; expected one of: " + ", ".join(LATENCY_COLUMNS)
        )

    delta_columns = sorted(name for name in column_names if name.startswith("delta_"))
    if not delta_columns:
        warnings.append("No delta_* columns found; quality_delta is unavailable.")

    candidates = aggregate_candidates(
        rows,
        group_columns=group_columns,
        latency_column=latency_column,
        delta_columns=delta_columns,
    )
    failure_rows = [row for row in rows if not is_ok(row.values) or is_fallback(row.values)]

    print(f"Loaded {len(rows)} rows from {len(set(row.source for row in rows))} file(s).")
    if group_columns:
        print("Grouping by: " + ", ".join(group_columns))
    else:
        print("Grouping by: source_csv, line_number")
    if latency_column:
        print(f"Latency: mean {latency_column} over successful rows.")
    print_quality_note(delta_columns)
    print()

    print_candidates(
        "Fastest Candidates",
        sorted(
            candidates.values(),
            key=lambda item: (
                none_last(item.mean_latency),
                none_last_desc(item.mean_quality_delta),
                candidate_label(item),
            ),
        )[: args.top],
    )
    print()
    print_candidates(
        "Best Quality Delta Candidates",
        sorted(
            candidates.values(),
            key=lambda item: (
                none_last_desc(item.mean_quality_delta),
                none_last(item.mean_latency),
                candidate_label(item),
            ),
        )[: args.top],
    )
    print()
    print_failures(failure_rows, limit=args.failure_limit)
    print_warnings(warnings)
    return 0


def load_rows(paths: Iterable[Path]) -> tuple[list[SweepRow], list[str]]:
    rows: list[SweepRow] = []
    warnings: list[str] = []
    for path in paths:
        expanded = path.expanduser()
        if not expanded.is_file():
            warnings.append(f"{path}: not found or not a file")
            continue
        try:
            with expanded.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    warnings.append(f"{path}: missing CSV header")
                    continue
                for line_number, row in enumerate(reader, start=2):
                    rows.append(
                        SweepRow(
                            source=expanded,
                            line_number=line_number,
                            values={key: value or "" for key, value in row.items() if key is not None},
                        )
                    )
        except OSError as exc:
            warnings.append(f"{path}: {exc}")
    return rows, warnings


def aggregate_candidates(
    rows: Sequence[SweepRow],
    *,
    group_columns: Sequence[str],
    latency_column: str | None,
    delta_columns: Sequence[str],
) -> dict[tuple[tuple[str, str], ...], CandidateStats]:
    candidates: dict[tuple[tuple[str, str], ...], CandidateStats] = {}
    for row in rows:
        key = candidate_key(row, group_columns)
        candidate = candidates.setdefault(key, CandidateStats(key=key))
        latency = parse_float(row.values.get(latency_column, "")) if latency_column else None
        quality, quality_terms = row_quality_delta(row.values, delta_columns)
        candidate.add(
            row,
            latency=latency,
            quality=quality,
            quality_terms=quality_terms,
        )
    return candidates


def candidate_key(row: SweepRow, group_columns: Sequence[str]) -> tuple[tuple[str, str], ...]:
    if not group_columns:
        return (("source_csv", row.source.name), ("line_number", str(row.line_number)))
    return tuple((name, row.values.get(name, "")) for name in group_columns)


def row_quality_delta(
    row: dict[str, str],
    delta_columns: Sequence[str],
) -> tuple[float | None, int]:
    values: list[float] = []
    for column in delta_columns:
        raw_value = parse_float(row.get(column, ""))
        if raw_value is None:
            continue
        metric = column.removeprefix("delta_")
        sign = 1.0 if metric in HIGHER_IS_BETTER_DELTAS else -1.0
        values.append(sign * raw_value)
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def print_candidates(title: str, candidates: Sequence[CandidateStats]) -> None:
    print(title)
    if not candidates:
        print("  (none)")
        return
    table = [
        [
            "rank",
            "candidate",
            "rows",
            "ok",
            "fail",
            "fallback",
            "latency_mean",
            "quality_delta",
            "delta_terms",
            "motions",
        ]
    ]
    for rank, candidate in enumerate(candidates, start=1):
        table.append(
            [
                str(rank),
                candidate_label(candidate),
                str(candidate.rows),
                str(candidate.ok_rows),
                str(candidate.failed_rows),
                str(candidate.fallback_rows),
                format_number(candidate.mean_latency),
                format_number(candidate.mean_quality_delta),
                str(candidate.quality_terms),
                ",".join(sorted(candidate.motions)) or "-",
            ]
        )
    print_table(table)


def print_failures(rows: Sequence[SweepRow], *, limit: int) -> None:
    print("Failures / Fallbacks")
    if not rows:
        print("  (none)")
        return
    if limit <= 0:
        print(f"  {len(rows)} row(s) hidden by --failure-limit 0.")
        return
    table = [["source", "line", "motion", "candidate", "status", "exit", "fallback", "error"]]
    for row in rows[:limit]:
        table.append(
            [
                row.source.name,
                str(row.line_number),
                row.values.get("motion_name") or row.values.get("motion") or "-",
                short_candidate(row.values),
                row.values.get("status", "") or "-",
                row.values.get("exit_code", "") or "-",
                "yes" if is_fallback(row.values) else "no",
                truncate(row.values.get("error", ""), 96),
            ]
        )
    print_table(table)
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more row(s) hidden by --failure-limit {limit}.")


def print_warnings(warnings: Sequence[str]) -> None:
    if not warnings:
        return
    print()
    print("Warnings")
    for warning in warnings:
        print(f"  - {warning}")


def print_quality_note(delta_columns: Sequence[str]) -> None:
    if not delta_columns:
        return
    positive_columns = [
        column for column in delta_columns if column.removeprefix("delta_") in HIGHER_IS_BETTER_DELTAS
    ]
    lower_columns = [column for column in delta_columns if column not in positive_columns]
    print(
        "Quality delta: mean oriented delta_*; score-like deltas are higher-is-better, "
        "other deltas are treated lower-is-better."
    )
    print("  Higher-is-better: " + (", ".join(positive_columns) if positive_columns else "-"))
    print("  Lower-is-better: " + (", ".join(lower_columns) if lower_columns else "-"))


def print_table(rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row_index, row in enumerate(rows):
        print(
            "  "
            + "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
        if row_index == 0:
            print("  " + "  ".join("-" * width for width in widths))


def all_column_names(rows: Sequence[SweepRow]) -> set[str]:
    return {name for row in rows for name in row.values}


def first_present(candidates: Sequence[str], columns: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def is_ok(row: dict[str, str]) -> bool:
    status = row.get("status", "").strip().lower()
    exit_code = row.get("exit_code", "").strip()
    if exit_code not in ("", "0"):
        return False
    return status in OK_STATUSES


def is_fallback(row: dict[str, str]) -> bool:
    return row.get("mpc_used_baseline_fallback", "").strip().lower() in TRUE_VALUES


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def none_last(value: float | None) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, value)


def none_last_desc(value: float | None) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, -value)


def candidate_label(candidate: CandidateStats) -> str:
    return ", ".join(f"{name}={value or '-'}" for name, value in candidate.key)


def short_candidate(row: dict[str, str]) -> str:
    parts = [
        f"{name}={row[name]}"
        for name in DEFAULT_GROUP_COLUMNS
        if row.get(name, "") != ""
    ]
    return ", ".join(parts) or "-"


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def truncate(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text or "-"
    return text[: max_chars - 1] + "..."


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
