# G1 WBC Local-First Low-Sample Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local-first G1 WBC low-sample MPC experiment stage that keeps the current `best_s128` speed envelope while prioritizing local/posture quality and enforcing global, smoothness, contact, and runtime guardrails.

**Architecture:** Add a small pure Python module for candidate reward-weight generation, fixed experiment parameters, and guardrail assessment. Add a thin CLI runner that writes candidate reward JSON files, emits deterministic commands around the existing `run_g1_wbc_low_sample_sweep.py`, supports dry-run/execute modes, and summarizes completed metrics without requiring CUDA in tests.

**Tech Stack:** Python stdlib, existing `spider.tasks.g1_wbc.evaluate`, existing `scripts/run_g1_wbc_low_sample_sweep.py`, `unittest`.

---

## Experiment Objective

Keep the current best speed configuration fixed:

```text
method: g1_wbc_joint_global
samples: 128
iterations: 2
planning_horizon_steps: 40
control_steps: 20
knot_count: 8
sampling_mode: knot
temperature: 0.7
root_pos_sigma: 0.04
root_rot_sigma: 0.10
joint_sigma: 0.18
smooth_passes: 0
command_reg_weight: 0.0
command_smooth_weight: 0.0
guided_candidate: true
acceptance_gate: true
seed: 0
max_steps: 800
```

Only change reward weights for `g1_wbc_joint_global` in the first stage. The stage screens posture/local candidates on `jump`, then promotes the best candidates to `walk` and `qixing`. If `qixing` contact remains weak, run the contact variant there.

## Expected Targets

Hard gates:

- Same rollout work as `best_s128`: `128 * 2 * 40 * 40`.
- Runtime no more than `1.10x` same-machine control when real rollouts are available.
- No candidate with `mpc.used_baseline_fallback=true` can be promoted.
- `mpc.accepted=true` and `mpc.accepted_windows=40` for full 800-step runs.

Promotion gates against visualized `best_s128`:

- Score no worse by more than `0.05`.
- `root_pos_error_mean`, `body_global_pos_error_mean`, and `ee_global_pos_error_mean` no worse than `1.05x`.
- On `jump`, at least two of `joint_pos_error_mean`, `body_local_pos_error_mean`, and `ee_local_pos_error_mean` improve by at least `5%`.
- Smoothness guardrail: `control_delta_mean` no worse than current and `joint_acc_mean <= 1.02x`.
- Contact guardrail: `contact_mismatch_rate` no worse than current; `qixing` target is `<= 0.220`.

## Candidate Matrix

All omitted weights stay equal to v14.

- `jg_s128_v14_control`: same v14 reward weights, same-machine control.
- `jg_s128_L125_posture`: local/posture light increase.
  - `body_local_pos_error=6.25`
  - `body_local_rot_error=1.15`
  - `ee_local_pos_error=5.0`
  - `ee_local_rot_error=0.805`
  - `hand_local_pos_error=3.75`
  - `joint_pos_error=0.25`
- `jg_s128_L150_posture`: local/posture stronger increase.
  - `body_local_pos_error=7.5`
  - `body_local_rot_error=1.25`
  - `ee_local_pos_error=6.0`
  - `ee_local_rot_error=0.875`
  - `hand_local_pos_error=4.5`
  - `joint_pos_error=0.40`
- `jg_s128_L150_smooth`: `L150_posture` plus smoothness guardrail pressure.
  - `control_delta=1.08`
  - `action_delta=0.375`
  - `joint_acc=0.0045`
  - `joint_jerk=0.0012`
- `jg_s128_L150_contact`: `L150_posture` plus contact pressure.
  - `contact_switch=7.5`
  - `contact_force_delta=1.25`
  - `contact_false_positive=0.72`
  - `contact_false_negative=0.375`
- `jg_s128_L150_global090`: `L150_posture` plus slight global relaxation.
  - `body_global_pos_error=52.2`
  - `body_global_rot_error=6.48`
  - `ee_global_pos_error=5.4`
  - `ee_global_rot_error=0.99`
  - `hand_global_pos_error=4.5`

## Target Files

- Create `spider/tasks/g1_wbc/local_first_stage.py`.
- Create `scripts/run_g1_wbc_local_first_stage.py`.
- Create `tests/tasks/g1_wbc/test_local_first_stage.py`.
- Keep existing `scripts/run_g1_wbc_low_sample_sweep.py` unchanged unless the tests reveal an unavoidable gap.

## Task 1: Pure Experiment Spec And Guardrails

**Files:**

- Create: `spider/tasks/g1_wbc/local_first_stage.py`
- Create: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing tests for candidate generation**

Add tests that import `spider.tasks.g1_wbc.local_first_stage` and assert:

```python
base = {"g1_wbc_joint_global": {"body_local_pos_error": 5.0, "body_global_pos_error": 58.0}}
weights = stage.candidate_reward_weights(base, "jg_s128_L125_posture")
self.assertEqual(weights["g1_wbc_joint_global"]["body_local_pos_error"], 6.25)
self.assertEqual(weights["g1_wbc_joint_global"]["body_global_pos_error"], 58.0)
self.assertEqual(weights["g1_wbc_joint_global"]["joint_pos_error"], 0.25)
```

- [ ] **Step 2: Run the tests and verify they fail because the module is missing**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: import failure for `local_first_stage`.

- [ ] **Step 3: Implement candidate data and reward-weight generation**

Implement immutable candidate definitions, `candidate_reward_weights(base, candidate_name)`, and `write_candidate_reward_files(base_path, output_dir, candidate_names)`.

- [ ] **Step 4: Add guardrail assessment tests**

Use the current visualized `best_s128` baselines for `jump`, `walk`, and `qixing`. Add tests for:

- a passing `jump` result that improves two local metrics and keeps global within `1.05x`;
- a failing `jump` result that improves local but violates global guardrail;
- a failing `qixing` result that violates contact guardrail.

- [ ] **Step 5: Implement guardrail assessment**

Implement `assess_candidate(motion, metrics, mpc, duration_sec=None, control_duration_sec=None)` returning a small result object or dictionary with:

- `passed`
- `failure_labels`
- `improved_local_count`
- `global_guardrail_passed`
- `smooth_guardrail_passed`
- `contact_guardrail_passed`
- `score_guardrail_passed`

- [ ] **Step 6: Run tests and verify green**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: all local-first tests pass.

## Task 2: CLI Runner And Dry-Run Artifacts

**Files:**

- Create: `scripts/run_g1_wbc_local_first_stage.py`
- Modify: `tests/tasks/g1_wbc/test_local_first_stage.py`

- [ ] **Step 1: Write failing tests for command planning**

Add tests that call pure command-planning helpers and assert the generated command uses:

- `--samples 128`
- `--iterations 2`
- `--horizons 40`
- `--controls 20`
- `--knot-counts 8`
- `--sigma-triplets 0.04,0.1,0.18`
- `--seeds 0`
- `--max-steps 800`
- `--mpc-reward-weights <candidate-json>`

- [ ] **Step 2: Write failing tests for dry-run artifact generation**

Use `tempfile.TemporaryDirectory()` and assert dry-run writes:

- `experiment_plan.json`
- `planned_commands.sh`
- `reward_weights/<candidate>.json`

- [ ] **Step 3: Run the tests and verify they fail because the CLI/helpers are missing**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: failures for missing CLI/helper symbols.

- [ ] **Step 4: Implement the CLI runner**

The CLI should support:

- `--dry-run` default;
- `--execute`;
- `--output-root`, default `/tmp/g1_wbc_local_first_stage`;
- `--candidates`, default screening candidates except contact;
- `--motions`, default `jump`;
- `--promoted-motions`, default `walk qixing`;
- `--python-executable`;
- `--device`;
- `--include-contact`;
- `--summarize-only`.

In dry-run, write deterministic commands without invoking subprocesses. In execute mode, run commands sequentially with `subprocess.run`.

- [ ] **Step 5: Implement summary collection**

Collect existing per-candidate `summary.csv` files when present, assess guardrails, and write `guardrail_summary.csv`. Missing metrics should be reported as `planned` or `missing`, not treated as success.

- [ ] **Step 6: Run targeted tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest tests.tasks.g1_wbc.test_local_first_stage
```

Expected: all local-first tests pass.

## Task 3: Verification And Handoff

**Files:**

- No new production files expected.

- [ ] **Step 1: Run related unit tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest \
  tests.tasks.g1_wbc.test_local_first_stage \
  tests.tasks.g1_wbc.test_mpc_warm_start \
  tests.tasks.g1_wbc.test_bench_data_benchmark
```

Expected: all tests pass.

- [ ] **Step 2: Run local-first dry-run**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_first_stage_dryrun \
  --dry-run
```

Expected: generated plan, commands, and candidate reward JSON files.

- [ ] **Step 3: Report GPU limitation honestly**

If CUDA remains unavailable in this environment, do not mark real rollout experiments complete. Report that the stage is implemented and dry-run verified, with exact command for the user to launch on a CUDA machine.

## Self-Review

- The plan preserves the current speed envelope by fixing samples, iterations, horizon, control, knots, sigma, and seed.
- The plan prioritizes local/posture by changing only reward weights in the first implementation stage.
- Guardrails cover global, smoothness, contact, score, MPC acceptance, and runtime.
- The plan avoids claiming GPU rollout completion in a non-CUDA environment.
