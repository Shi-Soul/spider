# G1 WBC Fast-Quality Pareto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and verify a reproducible fast-quality Pareto diagnostic stage for SPIDER G1 WBC low-sample MPC.

**Architecture:** Extend the existing pure `local_first_stage.py` helper module with Pareto classification helpers and extend `run_g1_wbc_local_first_stage.py` with Pareto command planning, sample/seed overrides, and `pareto_summary.csv`. Keep the existing low-sample sweep runner as the execution backend; do not modify `evaluate.py`.

**Tech Stack:** Python stdlib, existing G1 WBC low-sample sweep runner, `unittest`, local CUDA rollout outside sandbox for real experiments.

---

## File Structure

- Modify `spider/tasks/g1_wbc/local_first_stage.py`
  - Add Pareto constants.
  - Add `assess_pareto_group`.
  - Add helpers for primary metric means and ratio diagnostics.
- Modify `scripts/run_g1_wbc_local_first_stage.py`
  - Add `pareto` candidate set and assessment mode.
  - Add `--samples` and `--seeds` stage-runner overrides.
  - Pass sweep overrides into low-sample sweep commands and plan payloads.
  - Write `pareto_summary.csv`.
- Modify `tests/tasks/g1_wbc/test_local_first_stage.py`
  - Add pure Pareto classification tests.
- Modify `tests/tasks/g1_wbc/test_local_transfer_runner.py`
  - Add runner tests for Pareto dry-run, command generation, and summary aggregation.
- Keep `scripts/run_g1_wbc_low_sample_sweep.py` unchanged.

## Task 1: Pure Pareto Classification Helpers

**Files:**

- Modify: `spider/tasks/g1_wbc/local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing tests for Pareto group classification**

Add a test class to `tests/tasks/g1_wbc/test_local_first_stage.py`:

```python
class FastQualityParetoAssessmentTest(unittest.TestCase):
    def test_pareto_group_classifies_baseline_close_budget(self) -> None:
        best = stage.BEST_S128_BASELINES["jump"]
        reference = {
            "score": -2.078,
            "root_pos_error_mean": 0.0425,
            "body_global_pos_error_mean": 0.0530,
            "ee_global_pos_error_mean": 0.0608,
        }
        rows = [
            _pareto_source_row(
                "jump",
                samples=192,
                seed=seed,
                score=-2.12,
                root_pos_error_mean=0.050,
                body_global_pos_error_mean=0.060,
                ee_global_pos_error_mean=0.070,
                reference=reference,
            )
            for seed in (0, 1, 2)
        ]

        result = stage.assess_pareto_group("jump", rows)

        self.assertEqual(result["pareto_class"], "baseline_close")
        self.assertEqual(result["repeat_count"], 3)
        self.assertEqual(result["ok_count"], 3)
        self.assertEqual(result["success_count"], 3)
        self.assertEqual(result["accepted_count"], 3)
        self.assertEqual(result["fallback_count"], 0)
        self.assertGreater(result["score_delta_vs_best_mean"], 0.0)
        self.assertGreaterEqual(result["score_delta_vs_baseline_mean"], -0.25)
        self.assertLessEqual(result["max_global_ratio_vs_baseline_mean"], 1.35)
        self.assertTrue(_expected_pareto_keys() <= set(result))

    def test_pareto_group_classifies_promising_budget_against_best_s128(self) -> None:
        best = stage.BEST_S128_BASELINES["jump"]
        reference = {
            "score": -2.078,
            "root_pos_error_mean": 0.0425,
            "body_global_pos_error_mean": 0.0530,
            "ee_global_pos_error_mean": 0.0608,
        }
        rows = [
            _pareto_source_row(
                "jump",
                samples=192,
                seed=seed,
                score=best["score"] + 0.10,
                root_pos_error_mean=best["root_pos_error_mean"] * 0.95,
                body_global_pos_error_mean=best["body_global_pos_error_mean"] * 0.95,
                ee_global_pos_error_mean=best["ee_global_pos_error_mean"] * 0.95,
                reference=reference,
            )
            for seed in (0, 1, 2)
        ]

        result = stage.assess_pareto_group("jump", rows)

        self.assertEqual(result["pareto_class"], "promising_budget")
        self.assertGreater(result["score_delta_vs_best_mean"], 0.0)
        self.assertLessEqual(result["max_global_ratio_vs_best_mean"], 1.0)

    def test_pareto_group_classifies_unstable_when_a_repeat_fails(self) -> None:
        rows = [
            _pareto_source_row("jump", samples=128, seed=0),
            _pareto_source_row("jump", samples=128, seed=1, metric_success=False),
            _pareto_source_row("jump", samples=128, seed=2),
        ]

        result = stage.assess_pareto_group("jump", rows)

        self.assertEqual(result["pareto_class"], "unstable")
        self.assertEqual(result["success_count"], 2)
        self.assertIn("success", result["failure_labels"])

    def test_pareto_group_classifies_invalid_without_usable_repeats(self) -> None:
        result = stage.assess_pareto_group("jump", [])

        self.assertEqual(result["pareto_class"], "invalid")
        self.assertEqual(result["repeat_count"], 0)
        self.assertIn("no_repeats", result["failure_labels"])
```

Add helper functions near the bottom of the test file:

```python
def _pareto_source_row(
    motion: str,
    *,
    samples: int,
    seed: int,
    status: str = "ok",
    metric_success: bool = True,
    score: float | None = None,
    root_pos_error_mean: float | None = None,
    body_global_pos_error_mean: float | None = None,
    ee_global_pos_error_mean: float | None = None,
    reference: dict[str, float] | None = None,
) -> dict[str, object]:
    baseline = stage.BEST_S128_BASELINES[motion]
    reference_values = reference or {
        "score": baseline["score"] + 0.20,
        "root_pos_error_mean": baseline["root_pos_error_mean"] * 0.80,
        "body_global_pos_error_mean": baseline["body_global_pos_error_mean"] * 0.80,
        "ee_global_pos_error_mean": baseline["ee_global_pos_error_mean"] * 0.80,
    }
    return {
        "motion_name": motion,
        "samples": samples,
        "seed": seed,
        "status": status,
        "duration_sec": 100.0 + samples / 100.0,
        "metric_success": metric_success,
        "metric_num_steps": 800,
        "metric_score": baseline["score"] if score is None else score,
        "metric_root_pos_error_mean": (
            baseline["root_pos_error_mean"]
            if root_pos_error_mean is None
            else root_pos_error_mean
        ),
        "metric_body_global_pos_error_mean": (
            baseline["body_global_pos_error_mean"]
            if body_global_pos_error_mean is None
            else body_global_pos_error_mean
        ),
        "metric_ee_global_pos_error_mean": (
            baseline["ee_global_pos_error_mean"]
            if ee_global_pos_error_mean is None
            else ee_global_pos_error_mean
        ),
        "metric_ee_local_pos_error_mean": baseline["ee_local_pos_error_mean"],
        "metric_contact_mismatch_rate": baseline["contact_mismatch_rate"],
        "metric_control_delta_mean": baseline["control_delta_mean"],
        "metric_joint_acc_mean": baseline["joint_acc_mean"],
        "mpc_accepted": True,
        "mpc_accepted_windows": 40,
        "mpc_used_baseline_fallback": False,
        "ref_score": reference_values["score"],
        "ref_root_pos_error_mean": reference_values["root_pos_error_mean"],
        "ref_body_global_pos_error_mean": reference_values["body_global_pos_error_mean"],
        "ref_ee_global_pos_error_mean": reference_values["ee_global_pos_error_mean"],
    }


def _expected_pareto_keys() -> set[str]:
    return {
        "pareto_class",
        "failure_labels",
        "repeat_count",
        "ok_count",
        "success_count",
        "accepted_count",
        "fallback_count",
        "duration_sec_mean",
        "score_mean",
        "score_delta_vs_best_mean",
        "score_delta_vs_baseline_mean",
        "max_global_ratio_vs_best_mean",
        "max_global_ratio_vs_baseline_mean",
    }
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: fail because `assess_pareto_group` does not exist.

- [ ] **Step 3: Implement Pareto helpers**

In `spider/tasks/g1_wbc/local_first_stage.py`, add constants near the other stage constants:

```python
PARETO_CANDIDATE_NAMES = ("jg_s128_v14_control",)
PARETO_SAMPLE_COUNTS = (64, 96, 128, 192, 256)
PARETO_REPEAT_SEEDS = (0, 1, 2)
PARETO_BASELINE_SCORE_TOLERANCE = 0.25
PARETO_BASELINE_GLOBAL_RATIO_LIMIT = 1.35
PARETO_PRIMARY_METRICS = (
    "score",
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
    "ee_local_pos_error_mean",
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)
```

Add these helpers before `_get_candidate`:

```python
def assess_pareto_group(
    motion: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify repeated runs for one motion/sample budget."""

    if not rows:
        return _empty_pareto_result("invalid", ["no_repeats"])

    baseline = _get_baseline(motion)
    repeat_count = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    success_count = sum(int(_as_bool(row.get("metric_success")) is True) for row in rows)
    accepted_count = sum(int(_as_bool(row.get("mpc_accepted")) is True) for row in rows)
    fallback_count = sum(
        int(_as_bool(row.get("mpc_used_baseline_fallback")) is True) for row in rows
    )
    full_run_count = sum(
        int(_as_float(row.get("metric_num_steps")) == FULL_RUN_STEPS) for row in rows
    )
    accepted_windows_count = sum(
        int(_as_float(row.get("mpc_accepted_windows")) == FULL_RUN_ACCEPTED_WINDOWS)
        for row in rows
    )

    failure_labels: list[str] = []
    if len(ok_rows) != repeat_count:
        failure_labels.append("status")
    if success_count != repeat_count:
        failure_labels.append("success")
    if accepted_count != repeat_count:
        failure_labels.append("mpc_accepted")
    if fallback_count:
        failure_labels.append("baseline_fallback")
    if full_run_count != repeat_count:
        failure_labels.append("full_run")
    if accepted_windows_count != repeat_count:
        failure_labels.append("accepted_windows")

    score_mean = _mean_prefixed(rows, "metric_score")
    duration_sec_mean = _mean_prefixed(rows, "duration_sec")
    score_delta_vs_best = (
        score_mean - baseline["score"] if score_mean is not None else float("-inf")
    )
    reference_score_mean = _mean_prefixed(rows, "ref_score")
    score_delta_vs_baseline = (
        score_mean - reference_score_mean
        if score_mean is not None and reference_score_mean is not None
        else float("-inf")
    )
    max_global_ratio_vs_best = _mean_max_global_ratio(rows, prefix="metric_", baseline=baseline)
    max_global_ratio_vs_baseline = _mean_max_global_ratio(rows, prefix="metric_", ref_prefix="ref_")

    metrics = {
        f"{metric}_mean": _mean_prefixed(rows, f"metric_{metric}")
        for metric in PARETO_PRIMARY_METRICS
    }
    metrics.update(
        {
            "duration_sec_mean": duration_sec_mean,
            "score_delta_vs_best_mean": score_delta_vs_best,
            "score_delta_vs_baseline_mean": score_delta_vs_baseline,
            "max_global_ratio_vs_best_mean": max_global_ratio_vs_best,
            "max_global_ratio_vs_baseline_mean": max_global_ratio_vs_baseline,
        }
    )

    if not ok_rows:
        pareto_class = "invalid"
    elif failure_labels:
        pareto_class = "unstable"
    elif (
        score_delta_vs_baseline >= -PARETO_BASELINE_SCORE_TOLERANCE
        and max_global_ratio_vs_baseline <= PARETO_BASELINE_GLOBAL_RATIO_LIMIT
    ):
        pareto_class = "baseline_close"
    elif score_delta_vs_best > 0.0 and max_global_ratio_vs_best <= 1.0:
        pareto_class = "promising_budget"
    else:
        pareto_class = "stable_but_low_quality"

    return {
        "pareto_class": pareto_class,
        "failure_labels": failure_labels,
        "repeat_count": repeat_count,
        "ok_count": len(ok_rows),
        "success_count": success_count,
        "accepted_count": accepted_count,
        "fallback_count": fallback_count,
        "full_run_count": full_run_count,
        "accepted_windows_count": accepted_windows_count,
        **metrics,
    }
```

Add the support helpers:

```python
def _empty_pareto_result(pareto_class: str, failure_labels: Sequence[str]) -> dict[str, Any]:
    return {
        "pareto_class": pareto_class,
        "failure_labels": list(failure_labels),
        "repeat_count": 0,
        "ok_count": 0,
        "success_count": 0,
        "accepted_count": 0,
        "fallback_count": 0,
        "full_run_count": 0,
        "accepted_windows_count": 0,
        "duration_sec_mean": float("nan"),
        "score_mean": float("nan"),
        "score_delta_vs_best_mean": float("-inf"),
        "score_delta_vs_baseline_mean": float("-inf"),
        "max_global_ratio_vs_best_mean": float("inf"),
        "max_global_ratio_vs_baseline_mean": float("inf"),
    }


def _mean_prefixed(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _mean_max_global_ratio(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    baseline: Mapping[str, float] | None = None,
    ref_prefix: str | None = None,
) -> float:
    ratios: list[float] = []
    for row in rows:
        row_ratios: list[float] = []
        for metric in GLOBAL_GUARDRAIL_METRICS:
            value = _as_float(row.get(f"{prefix}{metric}"))
            denominator = (
                baseline[metric]
                if baseline is not None
                else _as_float(row.get(f"{ref_prefix}{metric}"))
            )
            if value is None or denominator in (None, 0.0):
                continue
            row_ratios.append(value / float(denominator))
        if row_ratios:
            ratios.append(max(row_ratios))
    if not ratios:
        return float("inf")
    return sum(ratios) / len(ratios)
```

Add `_as_bool` near `_as_float`:

```python
def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None
```

- [ ] **Step 4: Run targeted tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: all local-first tests pass.

## Task 2: Pareto Runner Planning And Summary

**Files:**

- Modify: `scripts/run_g1_wbc_local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_transfer_runner.py`

- [ ] **Step 1: Write failing runner tests for Pareto mode**

Add tests to `LocalFirstTransferRunnerTest` in `tests/tasks/g1_wbc/test_local_transfer_runner.py`:

```python
    def test_selected_candidate_names_supports_pareto_set(self) -> None:
        runner = _load_runner()

        self.assertEqual(
            runner.selected_candidate_names(None, candidate_set="pareto"),
            stage.PARETO_CANDIDATE_NAMES,
        )

    def test_pareto_dry_run_writes_ladder_plan_and_planned_summary(self) -> None:
        runner = _load_runner()
        base = {"g1_wbc_joint_global": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--candidate-set",
                        "pareto",
                        "--dry-run",
                        "--output-root",
                        str(output_root),
                        "--base-reward-weights",
                        str(base_path),
                        "--python-executable",
                        "/opt/python",
                        "--device",
                        "cpu",
                    ]
                )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())
            planned_commands = (output_root / "planned_commands.sh").read_text()
            with (output_root / "pareto_summary.csv").open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["candidate_set"], "pareto")
        self.assertEqual(experiment_plan["assessment_mode"], "pareto")
        self.assertEqual(experiment_plan["motions"], ["jump"])
        self.assertEqual(experiment_plan["fixed_parameters"]["samples"], [64, 96, 128, 192, 256])
        self.assertEqual(experiment_plan["fixed_parameters"]["seeds"], [0, 1, 2])
        self.assertIn("--samples 64 96 128 192 256", planned_commands)
        self.assertIn("--seeds 0 1 2", planned_commands)
        self.assertEqual(len(rows), len(stage.PARETO_SAMPLE_COUNTS))
        self.assertEqual({row["status"] for row in rows}, {"planned"})
        self.assertEqual({row["pareto_class"] for row in rows}, {"invalid"})
        self.assertEqual(fieldnames, runner.PARETO_SUMMARY_COLUMNS)

    def test_pareto_summary_aggregates_by_motion_and_samples(self) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["jump"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root / "candidates" / "jg_s128_v14_control" / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        _candidate_metrics(
                            "jump",
                            score=str(baseline["score"] + 0.1),
                            root_pos_error_mean=str(baseline["root_pos_error_mean"] * 0.9),
                            body_global_pos_error_mean=str(baseline["body_global_pos_error_mean"] * 0.9),
                            ee_global_pos_error_mean=str(baseline["ee_global_pos_error_mean"] * 0.9),
                        ),
                        _accepted_mpc(),
                        samples="192",
                        seed=str(seed),
                        duration_sec=str(120.0 + seed),
                    )
                    for seed in (0, 1, 2)
                ],
            )

            rows = runner.write_pareto_summary(
                output_root=output_root,
                candidate_names=stage.PARETO_CANDIDATE_NAMES,
                motion_names=("jump",),
                sample_counts=(192,),
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["motion"], "jump")
        self.assertEqual(rows[0]["samples"], "192")
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["repeat_count"], "3")
        self.assertEqual(rows[0]["pareto_class"], "promising_budget")
        self.assertEqual(rows[0]["duration_sec_mean"], "121")
```

Update `_summary_row` in the same file to accept `samples` and `seed` keyword
arguments and include them in the returned row:

```python
def _summary_row(
    motion: str,
    metrics: dict[str, str],
    mpc: dict[str, str],
    *,
    status: str = "ok",
    samples: str = "128",
    seed: str = "0",
    duration_sec: str = "100.0",
) -> dict[str, str]:
    row = {
        "motion_name": motion,
        "samples": samples,
        "seed": seed,
        "status": status,
        "duration_sec": duration_sec,
    }
    ...
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_transfer_runner
```

Expected: failures for missing Pareto mode and `write_pareto_summary`.

- [ ] **Step 3: Implement runner Pareto mode**

In `scripts/run_g1_wbc_local_first_stage.py`, add columns after
`TRANSFER_SUMMARY_COLUMNS`:

```python
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
PARETO_CLASS_RANK = {
    "baseline_close": 0,
    "promising_budget": 1,
    "stable_but_low_quality": 2,
    "unstable": 3,
    "invalid": 4,
}
```

Change `FIXED_SWEEP_ARGS` handling by adding:

```python
def sweep_args_for_stage(*, samples: Sequence[int], seeds: Sequence[int]) -> dict[str, str]:
    args = dict(FIXED_SWEEP_ARGS)
    args["samples"] = " ".join(str(value) for value in samples)
    args["seeds"] = " ".join(str(value) for value in seeds)
    return args
```

Update `plan_sweep_commands` to accept `sweep_args: Mapping[str, str] | None = None`
and use `selected_sweep_args = dict(FIXED_SWEEP_ARGS if sweep_args is None else sweep_args)`
instead of reading `FIXED_SWEEP_ARGS` directly.

Update `write_dry_run_artifacts` and `experiment_plan_payload` to accept
`sweep_args`, `sample_counts`, and `seed_values`. Store `samples` and `seeds` as
integer lists in `fixed_parameters`.

Add:

```python
def write_pareto_summary(
    *,
    output_root: str | Path,
    candidate_names: Sequence[str],
    motion_names: Sequence[str],
    sample_counts: Sequence[int],
) -> list[dict[str, str]]:
    root = Path(output_root)
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
                ]
                if not source_rows:
                    row = _planned_pareto_row(candidate_name, motion_name, sample_count, summary_path)
                    sort_key = _pareto_sort_key_for_planned(source_index)
                else:
                    row, sort_key = _pareto_row_from_group(
                        candidate_name,
                        motion_name,
                        sample_count,
                        summary_path,
                        source_rows,
                        source_index=source_index,
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
    return rows
```

Add `_pareto_row_from_group`, `_planned_pareto_row`, `_empty_pareto_row`,
`_pareto_sort_key_for_result`, `_pareto_sort_key_for_planned`, and `_as_int`
matching the existing transfer/frontier helper style.

Add `pareto` to `selected_candidate_names`, `parse_args` choices, and `main`.
Default behavior for `--candidate-set pareto`:

- `assessment_mode="pareto"` when not explicitly provided.
- `motions=["jump"]` when `--motions` is omitted.
- `samples=stage.PARETO_SAMPLE_COUNTS` when `--samples` is omitted.
- `seeds=stage.PARETO_REPEAT_SEEDS` when `--seeds` is omitted.

- [ ] **Step 4: Run runner tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_transfer_runner
```

Expected: all transfer runner tests pass.

## Task 3: Documentation, Dry Run, And Verification

**Files:**

- Modify: `milestone.md`

- [ ] **Step 1: Run combined unit tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest \
  tests.tasks.g1_wbc.test_local_first_stage \
  tests.tasks.g1_wbc.test_local_transfer_stage \
  tests.tasks.g1_wbc.test_local_transfer_runner
```

Expected: all tests pass.

- [ ] **Step 2: Run py_compile**

Run:

```bash
python -m py_compile \
  scripts/run_g1_wbc_local_first_stage.py \
  spider/tasks/g1_wbc/local_first_stage.py \
  tests/tasks/g1_wbc/test_local_first_stage.py \
  tests/tasks/g1_wbc/test_local_transfer_runner.py
```

Expected: command exits 0.

- [ ] **Step 3: Run Pareto dry-run**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_fast_quality_pareto_dryrun_20260623 \
  --candidate-set pareto \
  --dry-run \
  --python-executable ./.venv/bin/python \
  --device cuda:0
```

Expected:

- `experiment_plan.json` exists.
- `planned_commands.sh` contains `--samples 64 96 128 192 256`.
- `planned_commands.sh` contains `--seeds 0 1 2`.
- `pareto_summary.csv` has five planned rows for `jump`.
- `pareto_decision.md` exists and says `Conclusion: pending`.

- [ ] **Step 4: Update milestone with stage plan**

Append a `2026-06-23 Fast-Quality Pareto Stage` section to `milestone.md` with:

- design path
- plan path
- default output root `/tmp/g1_wbc_fast_quality_pareto_20260623`
- Stage A repeatability matrix
- Stage B sample-budget ladder
- Stage C transfer rule
- expected interpretation of each possible result class

- [ ] **Step 5: Run diff check**

Run:

```bash
git diff --check -- \
  docs/superpowers/specs/2026-06-23-g1-wbc-fast-quality-pareto-design.md \
  docs/superpowers/plans/2026-06-23-g1-wbc-fast-quality-pareto-plan.md \
  scripts/run_g1_wbc_local_first_stage.py \
  spider/tasks/g1_wbc/local_first_stage.py \
  tests/tasks/g1_wbc/test_local_first_stage.py \
  tests/tasks/g1_wbc/test_local_transfer_runner.py \
  milestone.md
```

Expected: command exits 0.

## Task 4: GPU Execution Handoff

**Files:**

- No code changes expected.
- Outputs under `/tmp/g1_wbc_fast_quality_pareto_20260623`.

- [ ] **Step 1: Execute the Stage B jump ladder**

Run on the GPU machine:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_fast_quality_pareto_20260623 \
  --candidate-set pareto \
  --assessment-mode pareto \
  --motions jump \
  --samples 64 96 128 192 256 \
  --seeds 0 1 2 \
  --execute \
  --python-executable ./.venv/bin/python \
  --device cuda:0
```

Expected:

- `candidates/jg_s128_v14_control/summary.csv` has 15 rows.
- `pareto_summary.csv` has five completed rows.
- `pareto_decision.md` has a non-pending conclusion unless all rows are invalid.
- The top row is either `baseline_close`, `promising_budget`, `stable_but_low_quality`, or `unstable`.

- [ ] **Step 2: Decide promotion budgets**

Read `/tmp/g1_wbc_fast_quality_pareto_20260623/pareto_summary.csv`:

- If a `baseline_close` row exists, promote the fastest `baseline_close` budget.
- Otherwise promote up to two `promising_budget` rows.
- If no `promising_budget` exists through `s256`, do not execute transfer; record that sample escalation is not enough.
- If promoted rows include `unstable`, do not execute transfer; record that stability repair is required.

Also read `/tmp/g1_wbc_fast_quality_pareto_20260623/pareto_decision.md`; it
summarizes the same decision as `sample_budget_likely`,
`sample_budget_partial`, `stability_likely`, or `reward_gating_likely`.

- [ ] **Step 3: Execute transfer for promoted budgets only**

Run the same command with `--motions walk qixing` and only promoted sample values.

If Stage B promotes one fastest `baseline_close` budget at `192`, run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_fast_quality_pareto_transfer_20260623 \
  --candidate-set pareto \
  --assessment-mode pareto \
  --motions walk qixing \
  --samples 192 \
  --seeds 0 1 2 \
  --execute \
  --python-executable ./.venv/bin/python \
  --device cuda:0
```

If Stage B promotes two `promising_budget` budgets at `192` and `256`, run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_fast_quality_pareto_transfer_20260623 \
  --candidate-set pareto \
  --assessment-mode pareto \
  --motions walk qixing \
  --samples 192 256 \
  --seeds 0 1 2 \
  --execute \
  --python-executable ./.venv/bin/python \
  --device cuda:0
```

Expected:

- `pareto_summary.csv` has one row per promoted sample and transfer motion.
- Transfer rows do not regress to `unstable` or `invalid`.

- [ ] **Step 4: Update milestone with observed results**

Append observed result paths and conclusion to `milestone.md`.

Use one of these conclusions:

- Sample budget appears limiting; promote fastest stable budget to visual review.
- Stability appears limiting; design an acceptance/repeatability repair stage.
- Reward/gating appears limiting; stop increasing samples and design a repair stage from `best_s128`.

## 2026-06-23 Execution Result

Stage B completed on `/tmp/g1_wbc_fast_quality_pareto_20260623`.

- Step 1 completed: `summary.csv` has 15 jump rows and `pareto_summary.csv` has
  five completed sample-budget rows.
- Step 2 completed: all budgets are `unstable` because `metric_success` is not
  repeat-stable across seeds.
- Step 3 skipped by design: no `baseline_close` or `promising_budget` row was
  eligible for walk/qixing transfer.
- Step 4 completed in `milestone.md`.

Observed conclusion: `stability_likely`. Next stage should repair
acceptance/repeatability around the `s256` quality signal before visual
promotion or transfer.
