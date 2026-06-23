# G1 WBC Local Middle-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute the next G1 WBC low-sample MPC experiment stage with a `3%` local improvement middle gate and a less entangled reward candidate matrix.

**Architecture:** Keep the existing `local_first_stage.py` pure helper module as the source of truth for candidate definitions and guardrail assessment. Keep `run_g1_wbc_local_first_stage.py` as the thin orchestration layer that writes reward JSON files, runs the existing low-sample sweep runner, and summarizes metrics. Add only the configuration needed for the new default matrix and configurable local improvement threshold.

**Tech Stack:** Python stdlib, existing G1 WBC low-sample sweep runner, `unittest`, local RTX 4070 CUDA rollout outside sandbox.

---

## File Structure

- Modify `spider/tasks/g1_wbc/local_first_stage.py`
  - Add next-stage candidate definitions.
  - Keep prior-stage candidates available.
  - Add a configurable local improvement multiplier to `assess_candidate`.
  - Expose strict and middle-gate local improvement constants.
- Modify `scripts/run_g1_wbc_local_first_stage.py`
  - Make the next-stage matrix the default screening matrix.
  - Add `--local-improvement-pct`, default `3.0`.
  - Include the selected threshold in `experiment_plan.json`.
  - Pass the selected threshold into guardrail summarization.
- Modify `tests/tasks/g1_wbc/test_local_first_stage.py`
  - Add tests for new candidate reward weights.
  - Add tests proving `3%` passes while `5%` would fail.
  - Add tests proving runner defaults and plan payload record the threshold.
- Add or update `docs/superpowers/specs/2026-06-22-g1-wbc-local-middle-gate-design.md`
  - Record the stage design and success criteria.

## Task 1: Pure Stage Candidate Matrix And Guardrail Threshold

**Files:**

- Modify: `spider/tasks/g1_wbc/local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing tests for the next-stage candidates**

Add tests that assert:

```python
weights = stage.candidate_reward_weights(
    {"g1_wbc_joint_global": {"control_delta": 0.9, "contact_switch": 6.0}},
    "jg_s128_v14_smooth015",
)
self.assertEqual(weights["g1_wbc_joint_global"]["control_delta"], 1.035)

weights = stage.candidate_reward_weights(
    {"g1_wbc_joint_global": {"contact_switch": 6.0}},
    "jg_s128_v14_contact015",
)
self.assertEqual(weights["g1_wbc_joint_global"]["contact_switch"], 6.9)

weights = stage.candidate_reward_weights(
    {"g1_wbc_joint_global": {"body_local_pos_error": 5.0}},
    "jg_s128_L140_posture",
)
self.assertEqual(weights["g1_wbc_joint_global"]["body_local_pos_error"], 7.0)
self.assertEqual(weights["g1_wbc_joint_global"]["joint_pos_error"], 0.34)
```

The exact values to implement are:

```text
smooth015: control_delta=1.035, action_delta=0.345, joint_acc=0.00345, joint_jerk=0.00092
smooth025: control_delta=1.125, action_delta=0.375, joint_acc=0.00375, joint_jerk=0.001
contact015: contact_switch=6.9, contact_force_delta=1.15, contact_false_positive=0.69, contact_false_negative=0.2875
L135: body_local_pos=6.75, body_local_rot=1.19, ee_local_pos=5.4, ee_local_rot=0.833, hand_local_pos=4.05, joint_pos=0.31
L140: body_local_pos=7.0, body_local_rot=1.21, ee_local_pos=5.6, ee_local_rot=0.847, hand_local_pos=4.2, joint_pos=0.34
L145: body_local_pos=7.25, body_local_rot=1.23, ee_local_pos=5.8, ee_local_rot=0.861, hand_local_pos=4.35, joint_pos=0.37
```

- [ ] **Step 2: Write failing tests for configurable local threshold**

Add a test where `jump` metrics improve exactly two local metrics by `4%`.

Expected behavior:

```python
result = stage.assess_candidate(
    "jump",
    metrics,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
    local_improvement_multiplier=stage.MIDDLE_GATE_LOCAL_IMPROVEMENT_MULTIPLIER,
)
self.assertTrue(result["local_guardrail_passed"])

strict = stage.assess_candidate(
    "jump",
    metrics,
    _accepted_mpc(),
    duration_sec=100.0,
    control_duration_sec=100.0,
    local_improvement_multiplier=stage.STRICT_LOCAL_IMPROVEMENT_MULTIPLIER,
)
self.assertFalse(strict["local_guardrail_passed"])
```

- [ ] **Step 3: Run the targeted test and verify it fails**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: failures for unknown candidate names or missing threshold arguments.

- [ ] **Step 4: Implement the candidate matrix and threshold constants**

In `local_first_stage.py`:

- Add `STRICT_LOCAL_IMPROVEMENT_MULTIPLIER = 0.95`.
- Add `MIDDLE_GATE_LOCAL_IMPROVEMENT_MULTIPLIER = 0.97`.
- Keep `LOCAL_IMPROVEMENT_MULTIPLIER` as an alias for the strict multiplier if
  needed for compatibility.
- Add the new candidate definitions listed in Step 1.
- Add `NEXT_STAGE_CANDIDATE_NAMES` containing:

```python
(
    "jg_s128_v14_control",
    "jg_s128_v14_smooth015",
    "jg_s128_v14_smooth025",
    "jg_s128_v14_contact015",
    "jg_s128_v14_smooth015_contact015",
    "jg_s128_L135_posture",
    "jg_s128_L140_posture",
    "jg_s128_L145_posture",
    "jg_s128_L140_smooth015",
    "jg_s128_L140_contact015",
    "jg_s128_L145_smooth015_contact015",
)
```

- Update `assess_candidate` to accept keyword-only
  `local_improvement_multiplier: float = STRICT_LOCAL_IMPROVEMENT_MULTIPLIER`
  and use that multiplier for local improvement counting.

- [ ] **Step 5: Run targeted tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: all local-first tests pass.

## Task 2: Runner Defaults And Plan/Summary Threshold Plumbing

**Files:**

- Modify: `scripts/run_g1_wbc_local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing runner tests**

Add tests that assert:

```python
self.assertEqual(
    runner.selected_candidate_names(None),
    stage.NEXT_STAGE_CANDIDATE_NAMES,
)
```

Add a dry-run test asserting `experiment_plan.json` includes:

```json
{
  "local_improvement_pct": 3.0,
  "local_improvement_multiplier": 0.97
}
```

Add a summary test where a `4%` local improvement row passes when
`local_improvement_pct=3.0` and fails when `local_improvement_pct=5.0`.

- [ ] **Step 2: Run targeted tests and verify they fail**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: failures for old default candidates, missing plan keys, or missing
summary threshold plumbing.

- [ ] **Step 3: Implement runner threshold plumbing**

In `run_g1_wbc_local_first_stage.py`:

- Change default candidate selection to `stage.NEXT_STAGE_CANDIDATE_NAMES`.
- Replace `CONTACT_CANDIDATE` default exclusion with no exclusion for the next
  matrix; keep explicit candidate validation.
- Add `--local-improvement-pct` to argparse, default `3.0`.
- Add a helper that converts a percent to multiplier:

```python
def local_improvement_multiplier_from_pct(percent: float) -> float:
    if percent < 0.0 or percent >= 100.0:
        raise ValueError("local improvement pct must be in [0, 100).")
    return 1.0 - percent / 100.0
```

- Include the percent and multiplier in `experiment_plan_payload`.
- Pass the multiplier from `write_guardrail_summary` to
  `stage.assess_candidate`.
- Make `main` pass the parsed percent through both execute and summarize-only
  paths.

- [ ] **Step 4: Run targeted tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: all local-first tests pass.

## Task 3: Verification And Rollout Execution

**Files:**

- No production code changes expected.
- Outputs under `/tmp/g1_wbc_local_middle_gate_dryrun`.
- Outputs under `/tmp/g1_wbc_local_middle_gate_gpu_20260622`.

- [ ] **Step 1: Run full relevant unit tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage tests.tasks.g1_wbc.test_mpc_warm_start tests.tasks.g1_wbc.test_bench_data_benchmark
```

Expected: all tests pass.

- [ ] **Step 2: Run compile and whitespace checks**

Run:

```bash
python -m py_compile scripts/run_g1_wbc_local_first_stage.py spider/tasks/g1_wbc/local_first_stage.py tests/tasks/g1_wbc/test_local_first_stage.py
git diff --check -- .gitignore docs/superpowers/specs/2026-06-22-g1-wbc-local-middle-gate-design.md docs/superpowers/plans/2026-06-22-g1-wbc-local-middle-gate-plan.md scripts/run_g1_wbc_local_first_stage.py spider/tasks/g1_wbc/local_first_stage.py tests/tasks/g1_wbc/test_local_first_stage.py
```

Expected: both commands pass.

- [ ] **Step 3: Run dry-run**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py --output-root /tmp/g1_wbc_local_middle_gate_dryrun --dry-run
```

Expected: `experiment_plan.json` has 11 candidates and `local_improvement_pct`
is `3.0`.

- [ ] **Step 4: Run GPU jump screening outside sandbox**

Run outside sandbox because CUDA is blocked inside the sandbox:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py --output-root /tmp/g1_wbc_local_middle_gate_gpu_20260622 --execute
```

Expected: one `jump` rollout per candidate and a completed
`guardrail_summary.csv`.

- [ ] **Step 5: Promote only passed candidates**

If `/tmp/g1_wbc_local_middle_gate_gpu_20260622/promoted_commands.sh` contains
actual low-sample sweep commands, run:

```bash
bash /tmp/g1_wbc_local_middle_gate_gpu_20260622/promoted_commands.sh
```

Then summarize:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py --output-root /tmp/g1_wbc_local_middle_gate_gpu_20260622 --summarize-only
```

If the script only contains the placeholder, do not run promoted rollouts.

- [ ] **Step 6: Report results**

Report:

- output root;
- candidates run;
- which candidates passed/failed and why;
- key metric deltas against `best_s128` and same-machine control;
- whether any candidate deserves a strict `5%` follow-up or sigma sweep.
