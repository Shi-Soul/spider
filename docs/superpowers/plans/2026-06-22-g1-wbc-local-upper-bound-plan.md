# G1 WBC Local Upper-Bound Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute a local/posture upper-bound experiment stage that keeps the current low-sample rollout budget while ranking relaxed frontier candidates.

**Architecture:** Extend the pure `local_first_stage.py` module with upper-bound candidates and frontier assessment helpers. Extend the existing runner with explicit candidate-set and assessment-mode switches so the prior middle-gate promotion flow stays unchanged. Add tests first for candidate values, frontier classification, runner mode selection, and deterministic frontier summary output.

**Tech Stack:** Python stdlib, existing G1 WBC low-sample sweep runner, `unittest`, local RTX 4070 CUDA rollout outside sandbox.

---

## File Structure

- Modify `spider/tasks/g1_wbc/local_first_stage.py`
  - Add upper-bound candidate definitions.
  - Add `UPPER_BOUND_CANDIDATE_NAMES`.
  - Add pure frontier assessment and ranking helpers.
- Modify `scripts/run_g1_wbc_local_first_stage.py`
  - Add `--candidate-set`.
  - Add `--assessment-mode`.
  - Write `frontier_summary.csv` in upper-bound mode.
- Modify `tests/tasks/g1_wbc/test_local_first_stage.py`
  - Add candidate, frontier, runner, and summary tests.

## Task 1: Upper-Bound Candidates And Frontier Assessment

**Files:**

- Modify: `spider/tasks/g1_wbc/local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing tests for upper-bound candidates**

Add tests that assert `candidate_reward_weights` contains:

```python
expected_l138 = {
    "body_local_pos_error": 6.9,
    "body_local_rot_error": 1.202,
    "ee_local_pos_error": 5.52,
    "ee_local_rot_error": 0.8414,
    "hand_local_pos_error": 4.14,
    "joint_pos_error": 0.328,
}
expected_smooth005 = {
    "control_delta": 0.945,
    "action_delta": 0.315,
    "joint_acc": 0.00315,
    "joint_jerk": 0.00084,
}
```

Also assert `UPPER_BOUND_CANDIDATE_NAMES` equals:

```python
(
    "jg_s128_v14_control",
    "jg_s128_L138_posture",
    "jg_s128_L140_posture",
    "jg_s128_L142_posture",
    "jg_s128_L145_posture",
    "jg_s128_L148_posture",
    "jg_s128_L150_posture",
    "jg_s128_L155_posture",
    "jg_s128_L142_smooth005",
    "jg_s128_L145_smooth005",
    "jg_s128_L148_smooth005",
    "jg_s128_L150_smooth005",
)
```

- [ ] **Step 2: Write failing tests for frontier assessment**

Add tests for:

```python
local_frontier = stage.assess_upper_bound_candidate(
    "jump",
    metrics_with_two_locals_5pct_better,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
)
self.assertEqual(local_frontier["frontier_class"], "local_frontier")

near = stage.assess_upper_bound_candidate(
    "jump",
    metrics_with_two_locals_3pct_better,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
)
self.assertEqual(near["frontier_class"], "near_frontier")

invalid = stage.assess_upper_bound_candidate(
    "jump",
    metrics_with_global_ratio_1p36,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
)
self.assertEqual(invalid["frontier_class"], "invalid")
self.assertIn("relaxed_global_guardrail", invalid["failure_labels"])
```

- [ ] **Step 3: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: failures for missing upper-bound candidates and helper.

- [ ] **Step 4: Implement candidates and pure assessment**

In `local_first_stage.py`, add:

```python
UPPER_BOUND_SCORE_TOLERANCE = 0.25
UPPER_BOUND_GLOBAL_REGRESSION_MULTIPLIER = 1.35
```

Add posture overrides:

```text
L138: body=6.9, body_rot=1.202, ee=5.52, ee_rot=0.8414, hand=4.14, joint=0.328
L142: body=7.1, body_rot=1.218, ee=5.68, ee_rot=0.8526, hand=4.26, joint=0.352
L148: body=7.4, body_rot=1.242, ee=5.92, ee_rot=0.8694, hand=4.44, joint=0.388
L155: body=7.75, body_rot=1.27, ee=6.2, ee_rot=0.889, hand=4.65, joint=0.43
```

Add `_SMOOTH005_OVERRIDES` with the values from Step 1.

Add `assess_upper_bound_candidate(...)` returning at least:

```python
{
    "frontier_class": "...",
    "failure_labels": [...],
    "hard_guardrail_passed": bool,
    "score_relaxed_passed": bool,
    "global_relaxed_passed": bool,
    "mpc_guardrail_passed": bool,
    "runtime_guardrail_passed": bool,
    "local_count_3": int,
    "local_count_5": int,
    "local_count_8": int,
    "local_composite_improvement": float,
    "score_delta": float,
    "max_global_ratio": float,
    "contact_delta": float,
    "control_delta_regression": float,
    "joint_acc_regression": float,
}
```

Classification:

- `invalid` when MPC/runtime/relaxed score/relaxed global fail.
- `local_frontier` when `(local_count_5 >= 2 or local_count_3 == 3)`.
- `near_frontier` when `local_count_3 >= 2`.
- otherwise `dead_end`.

- [ ] **Step 5: Run tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: tests pass.

## Task 2: Runner Upper-Bound Mode And Frontier Summary

**Files:**

- Modify: `scripts/run_g1_wbc_local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing runner tests**

Add tests that assert:

```python
self.assertEqual(
    runner.selected_candidate_names(None, candidate_set="upper-bound"),
    stage.UPPER_BOUND_CANDIDATE_NAMES,
)
```

Add a dry-run test asserting `experiment_plan.json` includes:

```json
{
  "candidate_set": "upper-bound",
  "assessment_mode": "upper-bound"
}
```

Add a summary test where two completed candidate rows produce
`frontier_summary.csv` sorted with `local_frontier` before `near_frontier`.

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: failures for missing runner mode plumbing.

- [ ] **Step 3: Implement runner mode plumbing**

In `run_g1_wbc_local_first_stage.py`:

- Add `--candidate-set`, choices `middle` and `upper-bound`, default `middle`.
- Add `--assessment-mode`, choices `promotion` and `upper-bound`, default `promotion`.
- Update `selected_candidate_names(candidate_names, *, include_contact=False, candidate_set="middle")`.
- Add `write_frontier_summary(...)`.
- In `main`, when `assessment_mode == "upper-bound"`, write
  `frontier_summary.csv` after execution or in `--summarize-only`, and do not
  write promotion commands as the main output.
- Keep existing promotion mode behavior unchanged.

`frontier_summary.csv` columns must include:

```text
rank,candidate,motion,status,frontier_class,failure_labels,hard_guardrail_passed,
score_relaxed_passed,global_relaxed_passed,mpc_guardrail_passed,
runtime_guardrail_passed,local_count_3,local_count_5,local_count_8,
local_composite_improvement,score_delta,max_global_ratio,contact_delta,
control_delta_regression,joint_acc_regression,summary_csv
```

- [ ] **Step 4: Run tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: tests pass.

## Task 3: Verification And GPU Rollout

**Files:**

- No code changes expected.
- Outputs under `/tmp/g1_wbc_local_upper_bound_dryrun`.
- Outputs under `/tmp/g1_wbc_local_upper_bound_gpu_20260622`.

- [ ] **Step 1: Run relevant tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage tests.tasks.g1_wbc.test_mpc_warm_start tests.tasks.g1_wbc.test_bench_data_benchmark
```

Expected: all pass.

- [ ] **Step 2: Run compile and whitespace checks**

Run:

```bash
python -m py_compile scripts/run_g1_wbc_local_first_stage.py spider/tasks/g1_wbc/local_first_stage.py tests/tasks/g1_wbc/test_local_first_stage.py
git diff --check -- docs/superpowers/specs/2026-06-22-g1-wbc-local-upper-bound-design.md docs/superpowers/plans/2026-06-22-g1-wbc-local-upper-bound-plan.md scripts/run_g1_wbc_local_first_stage.py spider/tasks/g1_wbc/local_first_stage.py tests/tasks/g1_wbc/test_local_first_stage.py
```

Expected: both pass.

- [ ] **Step 3: Run dry-run**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_upper_bound_dryrun \
  --candidate-set upper-bound \
  --assessment-mode upper-bound \
  --dry-run
```

Expected: plan has 12 candidates, candidate set `upper-bound`, assessment mode
`upper-bound`.

- [ ] **Step 4: Run GPU rollout outside sandbox**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_upper_bound_gpu_20260622 \
  --candidate-set upper-bound \
  --assessment-mode upper-bound \
  --execute
```

Expected: one `jump` rollout per candidate and `frontier_summary.csv` written.

- [ ] **Step 5: Report frontier**

Report top frontier rows, failure labels, and the recovery strategy:

- if `L142/L145` are frontier, next stage should recover score/global/smooth;
- if only `L148+` are frontier, reward can pull local but tradeoff is hard;
- if no frontier appears, move to sampling shape (`joint_sigma`) before more
  reward tuning.
