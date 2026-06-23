# G1 WBC Local Frontier Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the transfer/recovery experiment stage that validates top local-frontier anchors on `walk`/`qixing` and prepares a small recovery matrix for smooth/contact repair.

**Architecture:** Extend the existing pure `local_first_stage.py` module with transfer and recovery candidate sets plus a transfer assessment helper. Extend `run_g1_wbc_local_first_stage.py` with `transfer`/`recovery` candidate sets, transfer-mode summaries, and motion defaults while preserving existing middle and upper-bound behavior.

**Tech Stack:** Python stdlib, existing low-sample sweep runner, `unittest`, local CUDA rollout outside sandbox.

---

## File Structure

- Modify `spider/tasks/g1_wbc/local_first_stage.py`
  - Add `TRANSFER_CANDIDATE_NAMES`.
  - Add `RECOVERY_CANDIDATE_NAMES`.
  - Add `smooth010` and `contact010` reward overrides.
  - Add `assess_transfer_candidate`.
- Modify `scripts/run_g1_wbc_local_first_stage.py`
  - Add `transfer` and `recovery` choices to `--candidate-set`.
  - Add `transfer` choice to `--assessment-mode`.
  - Infer transfer assessment and `walk qixing` motions for transfer candidate set.
  - Write `transfer_summary.csv`.
- Add `tests/tasks/g1_wbc/test_local_transfer_stage.py`
  - Tests for transfer/recovery candidates and transfer assessment.
- Add `tests/tasks/g1_wbc/test_local_transfer_runner.py`
  - Tests for runner candidate-set plumbing, dry-run artifacts, and transfer summaries.

## Task 1: Transfer Candidates And Pure Assessment

**Files:**

- Modify: `spider/tasks/g1_wbc/local_first_stage.py`
- Add: `tests/tasks/g1_wbc/test_local_transfer_stage.py`

- [ ] **Step 1: Write failing tests for transfer and recovery candidate sets**

Create `tests/tasks/g1_wbc/test_local_transfer_stage.py` with tests asserting:

```python
self.assertEqual(
    stage.TRANSFER_CANDIDATE_NAMES,
    (
        "jg_s128_v14_control",
        "jg_s128_L148_posture",
        "jg_s128_L142_smooth005",
        "jg_s128_L145_smooth005",
    ),
)
self.assertEqual(
    stage.RECOVERY_CANDIDATE_NAMES,
    (
        "jg_s128_v14_control",
        "jg_s128_L148_posture",
        "jg_s128_L148_smooth010",
        "jg_s128_L148_contact010",
        "jg_s128_L148_smooth010_contact010",
        "jg_s128_L142_smooth005_contact010",
        "jg_s128_L145_smooth010_contact010",
    ),
)
```

Also assert reward overrides:

```python
smooth010 = {
    "control_delta": 0.99,
    "action_delta": 0.33,
    "joint_acc": 0.0033,
    "joint_jerk": 0.00088,
}
contact010 = {
    "contact_switch": 6.6,
    "contact_force_delta": 1.1,
    "contact_false_positive": 0.66,
    "contact_false_negative": 0.275,
}
self.assertEqual(
    stage.candidate_reward_weights({stage.JOINT_GLOBAL_METHOD: {}}, "jg_s128_L148_smooth010")[stage.JOINT_GLOBAL_METHOD],
    {**stage.CANDIDATE_DEFINITIONS["jg_s128_L148_posture"].overrides, **smooth010},
)
self.assertEqual(
    stage.candidate_reward_weights({stage.JOINT_GLOBAL_METHOD: {}}, "jg_s128_L148_contact010")[stage.JOINT_GLOBAL_METHOD],
    {**stage.CANDIDATE_DEFINITIONS["jg_s128_L148_posture"].overrides, **contact010},
)
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_transfer_stage
```

Expected: fail because `TRANSFER_CANDIDATE_NAMES`, `RECOVERY_CANDIDATE_NAMES`, and new candidate definitions do not exist.

- [ ] **Step 3: Implement candidate definitions**

In `local_first_stage.py`, add:

```python
TRANSFER_LOCAL_IMPROVEMENT_MULTIPLIER = 0.97
TRANSFER_LOCAL_REGRESSION_MULTIPLIER = 1.02

_SMOOTH010_OVERRIDES = {
    "control_delta": 0.99,
    "action_delta": 0.33,
    "joint_acc": 0.0033,
    "joint_jerk": 0.00088,
}

_CONTACT010_OVERRIDES = {
    "contact_switch": 6.6,
    "contact_force_delta": 1.1,
    "contact_false_positive": 0.66,
    "contact_false_negative": 0.275,
}
```

Add these candidates to `CANDIDATE_DEFINITIONS`:

```python
"jg_s128_L148_smooth010": _candidate(
    "jg_s128_L148_smooth010",
    {**_L148_POSTURE_OVERRIDES, **_SMOOTH010_OVERRIDES},
),
"jg_s128_L148_contact010": _candidate(
    "jg_s128_L148_contact010",
    {**_L148_POSTURE_OVERRIDES, **_CONTACT010_OVERRIDES},
),
"jg_s128_L148_smooth010_contact010": _candidate(
    "jg_s128_L148_smooth010_contact010",
    {**_L148_POSTURE_OVERRIDES, **_SMOOTH010_OVERRIDES, **_CONTACT010_OVERRIDES},
),
"jg_s128_L142_smooth005_contact010": _candidate(
    "jg_s128_L142_smooth005_contact010",
    {**_L142_POSTURE_OVERRIDES, **_SMOOTH005_OVERRIDES, **_CONTACT010_OVERRIDES},
),
"jg_s128_L145_smooth010_contact010": _candidate(
    "jg_s128_L145_smooth010_contact010",
    {**_L145_POSTURE_OVERRIDES, **_SMOOTH010_OVERRIDES, **_CONTACT010_OVERRIDES},
),
```

Then add exact tuples:

```python
TRANSFER_CANDIDATE_NAMES = (
    "jg_s128_v14_control",
    "jg_s128_L148_posture",
    "jg_s128_L142_smooth005",
    "jg_s128_L145_smooth005",
)

RECOVERY_CANDIDATE_NAMES = (
    "jg_s128_v14_control",
    "jg_s128_L148_posture",
    "jg_s128_L148_smooth010",
    "jg_s128_L148_contact010",
    "jg_s128_L148_smooth010_contact010",
    "jg_s128_L142_smooth005_contact010",
    "jg_s128_L145_smooth010_contact010",
)
```

- [ ] **Step 4: Write failing tests for transfer assessment**

In `tests/tasks/g1_wbc/test_local_transfer_stage.py`, add tests for:

```python
result = stage.assess_transfer_candidate(
    "walk",
    metrics_with_one_local_3pct_better_and_no_local_regression,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
)
self.assertEqual(result["transfer_class"], "transfer_pass")
self.assertTrue(result["local_transfer_passed"])
self.assertEqual(result["local_count_3"], 1)
```

Add a recovery classification test:

```python
result = stage.assess_transfer_candidate(
    "qixing",
    metrics_with_contact_worse_than_qixing_target_but_score_global_runtime_ok,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
)
self.assertEqual(result["transfer_class"], "recovery_candidate")
self.assertIn("contact_guardrail", result["failure_labels"])
```

Add an invalid classification test:

```python
result = stage.assess_transfer_candidate(
    "walk",
    metrics_with_global_ratio_1p06,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
)
self.assertEqual(result["transfer_class"], "invalid")
self.assertIn("global_guardrail", result["failure_labels"])
```

- [ ] **Step 5: Implement transfer assessment**

Add `assess_transfer_candidate(...)` to `local_first_stage.py`. It should return:

```python
{
    "transfer_class": transfer_class,
    "failure_labels": failure_labels,
    "score_guardrail_passed": score_guardrail_passed,
    "global_guardrail_passed": global_guardrail_passed,
    "smooth_guardrail_passed": smooth_guardrail_passed,
    "contact_guardrail_passed": contact_guardrail_passed,
    "local_transfer_passed": local_transfer_passed,
    "mpc_guardrail_passed": mpc_guardrail_passed,
    "runtime_guardrail_passed": runtime_guardrail_passed,
    "local_count_3": local_count_3,
    "max_local_regression": max_local_regression,
    "score_delta": score_delta,
    "max_global_ratio": max_global_ratio,
    "contact_delta": contact_delta,
    "control_delta_regression": control_delta_regression,
    "joint_acc_regression": joint_acc_regression,
}
```

Classification:

```python
hard_failures = ["score_guardrail", "global_guardrail", "mpc_guardrail", "runtime_guardrail"]
if any(label in failure_labels for label in hard_failures):
    transfer_class = "invalid"
elif smooth_guardrail_passed and contact_guardrail_passed and local_transfer_passed:
    transfer_class = "transfer_pass"
else:
    transfer_class = "recovery_candidate"
```

Local transfer logic:

```python
local_count_3 = sum(
    int(_local_threshold_improved(metrics, baseline, metric, TRANSFER_LOCAL_IMPROVEMENT_MULTIPLIER))
    for metric in LOCAL_IMPROVEMENT_METRICS
)
local_regressions = [
    _metric_ratio(metrics, metric, baseline[metric]) - 1.0
    for metric in LOCAL_IMPROVEMENT_METRICS
]
max_local_regression = max(local_regressions)
required_local_count = 2 if motion == "jump" else 1
local_transfer_passed = (
    local_count_3 >= required_local_count
    and max_local_regression <= TRANSFER_LOCAL_REGRESSION_MULTIPLIER - 1.0
)
```

- [ ] **Step 6: Run tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_transfer_stage
```

Expected: transfer-stage tests pass.

## Task 2: Runner Transfer Mode And Summary

**Files:**

- Modify: `scripts/run_g1_wbc_local_first_stage.py`
- Add: `tests/tasks/g1_wbc/test_local_transfer_runner.py`

- [ ] **Step 1: Write failing tests for candidate-set and mode parsing**

Create `tests/tasks/g1_wbc/test_local_transfer_runner.py` with tests that assert:

```python
self.assertEqual(
    runner.selected_candidate_names(None, candidate_set="transfer"),
    stage.TRANSFER_CANDIDATE_NAMES,
)
self.assertEqual(
    runner.selected_candidate_names(None, candidate_set="recovery"),
    stage.RECOVERY_CANDIDATE_NAMES,
)
```

Add a `parse_args` test:

```python
args = runner.parse_args(["--candidate-set", "transfer"])
self.assertEqual(args.assessment_mode, "transfer")
self.assertEqual(args.motions, ["walk", "qixing"])

args = runner.parse_args(["--candidate-set", "recovery"])
self.assertEqual(args.assessment_mode, "promotion")
self.assertEqual(args.motions, ["jump"])
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_transfer_runner
```

Expected: fail because runner choices and transfer mode do not exist.

- [ ] **Step 3: Implement runner option plumbing**

In `run_g1_wbc_local_first_stage.py`:

- Add `DEFAULT_TRANSFER_MOTIONS = ("walk", "qixing")`.
- Add `TRANSFER_SUMMARY_COLUMNS` with the columns from the design spec.
- Add `TRANSFER_CLASS_RANK = {"transfer_pass": 0, "recovery_candidate": 1, "invalid": 2}`.
- Update `selected_candidate_names`:

```python
elif candidate_set == "transfer":
    names = stage.TRANSFER_CANDIDATE_NAMES
elif candidate_set == "recovery":
    names = stage.RECOVERY_CANDIDATE_NAMES
```

- Update argparse choices:

```python
choices=("middle", "upper-bound", "transfer", "recovery")
choices=("promotion", "upper-bound", "transfer")
```

- Track whether `--motions` was provided using `_option_was_provided(arg_tokens, "--motions")`.
- After parsing:

```python
if not assessment_mode_provided and args.candidate_set == "upper-bound":
    args.assessment_mode = "upper-bound"
if not assessment_mode_provided and args.candidate_set == "transfer":
    args.assessment_mode = "transfer"
if not motions_provided and args.candidate_set == "transfer":
    args.motions = list(DEFAULT_TRANSFER_MOTIONS)
```

- [ ] **Step 4: Write failing tests for dry-run and transfer summary**

Add a dry-run test:

```python
exit_code = runner.main([
    "--candidate-set", "transfer",
    "--dry-run",
    "--output-root", str(output_root),
    "--base-reward-weights", str(base_path),
    "--python-executable", "/opt/python",
    "--device", "cpu",
])
self.assertEqual(exit_code, 0)
self.assertEqual(experiment_plan["candidate_set"], "transfer")
self.assertEqual(experiment_plan["assessment_mode"], "transfer")
self.assertEqual(experiment_plan["motions"], ["walk", "qixing"])
self.assertEqual(experiment_plan["candidates"], list(stage.TRANSFER_CANDIDATE_NAMES))
self.assertTrue((output_root / "transfer_summary.csv").exists())
```

Add a summary sorting test where one row is `transfer_pass`, one is
`recovery_candidate`, and one is `invalid`; assert this order appears in
`transfer_summary.csv`.

- [ ] **Step 5: Implement transfer summary writer**

Add `write_transfer_summary(...)`, `_transfer_row_from_summary(...)`,
`_planned_transfer_row(...)`, `_incomplete_transfer_row(...)`,
`_empty_transfer_row(...)`, `_transfer_sort_key_for_result(...)`, and
`_transfer_sort_key_for_planned(...)` using the same structure as
`write_frontier_summary`.

`_transfer_row_from_summary` must call:

```python
result = stage.assess_transfer_candidate(
    motion_name,
    metrics,
    mpc,
    duration_sec=duration_sec,
    control_duration_sec=control_duration_sec,
)
```

Update `main`:

- `--summarize-only` with `assessment_mode == "transfer"` writes
  `transfer_summary.csv`.
- dry-run with transfer mode writes planned `transfer_summary.csv`.
- execute with transfer mode writes completed `transfer_summary.csv`.

- [ ] **Step 6: Run runner transfer tests and existing local-stage tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest \
  tests.tasks.g1_wbc.test_local_transfer_runner \
  tests.tasks.g1_wbc.test_local_first_stage
```

Expected: tests pass.

## Task 3: Integrated Verification

**Files:**

- No new files expected beyond Task 1 and Task 2.

- [ ] **Step 1: Run all targeted unit tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest \
  tests.tasks.g1_wbc.test_local_first_stage \
  tests.tasks.g1_wbc.test_local_transfer_stage \
  tests.tasks.g1_wbc.test_local_transfer_runner
```

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m compileall \
  spider/tasks/g1_wbc/local_first_stage.py \
  scripts/run_g1_wbc_local_first_stage.py \
  tests/tasks/g1_wbc/test_local_first_stage.py \
  tests/tasks/g1_wbc/test_local_transfer_stage.py \
  tests/tasks/g1_wbc/test_local_transfer_runner.py
```

Expected: command exits `0`.

- [ ] **Step 3: Run transfer dry-run**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_transfer_dryrun \
  --candidate-set transfer \
  --dry-run
```

Expected:

- `experiment_plan.json` records `candidate_set=transfer`.
- `experiment_plan.json` records `assessment_mode=transfer`.
- `experiment_plan.json` records `motions=["walk", "qixing"]`.
- `transfer_summary.csv` exists with planned rows.

- [ ] **Step 4: Run recovery dry-run**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_recovery_dryrun \
  --candidate-set recovery \
  --assessment-mode promotion \
  --dry-run
```

Expected:

- `experiment_plan.json` records `candidate_set=recovery`.
- `experiment_plan.json` records `assessment_mode=promotion`.
- `experiment_plan.json` records `motions=["jump"]`.
- `planned_commands.sh` contains all recovery candidates.

- [ ] **Step 5: Report implementation status and GPU command**

Report:

- changed files;
- test and dry-run commands actually run;
- whether GPU execution was not run in the sandbox;
- next GPU command:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_transfer_gpu_20260622 \
  --candidate-set transfer \
  --execute
```
