# G1 WBC Local Frontier Recovery Experiment Design

## Goal

Turn the `jump` local/posture frontier found in the upper-bound stage into a
deployable low-sample MPC candidate. The next stage first validates whether the
best frontier anchors transfer to `walk` and `qixing`, then provides a small
recovery matrix that can recover smoothness/contact without increasing rollout
work.

This stage does not claim replacement quality by itself. It decides which anchor
deserves the next full three-motion report and which failure mode blocks it.

## Fixed Budget

All candidates keep the current low-sample budget:

```text
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
seed: 0
max_steps: 800
```

Runtime must remain no more than `1.10x` the same-machine
`jg_s128_v14_control` for the same motion.

## Transfer Anchors

The transfer validation matrix is intentionally small:

- `jg_s128_v14_control`
- `jg_s128_L148_posture`
- `jg_s128_L142_smooth005`
- `jg_s128_L145_smooth005`

Run these on `walk` and `qixing` first. `L148_posture` is the clean local anchor;
`L142_smooth005` is the strongest local-composite anchor; `L145_smooth005` is
the conservative contact-friendly anchor.

## Recovery Matrix

If transfer validation shows one anchor is close but blocked by smooth/contact,
run the recovery matrix on `jump` before broader validation:

- `jg_s128_v14_control`
- `jg_s128_L148_posture`
- `jg_s128_L148_smooth010`
- `jg_s128_L148_contact010`
- `jg_s128_L148_smooth010_contact010`
- `jg_s128_L142_smooth005_contact010`
- `jg_s128_L145_smooth010_contact010`

The `smooth010` suffix uses `+10%` smoothness terms:

```text
control_delta=0.99
action_delta=0.33
joint_acc=0.0033
joint_jerk=0.00088
```

The `contact010` suffix uses `+10%` contact terms:

```text
contact_switch=6.6
contact_force_delta=1.1
contact_false_positive=0.66
contact_false_negative=0.275
```

## Transfer Assessment

For each completed transfer row, classify it as:

- `transfer_pass`: hard constraints, score, global, smooth, contact, and local
  transfer checks all pass.
- `recovery_candidate`: hard constraints, score, and global checks pass, but at
  least one of local, smooth, or contact still needs recovery.
- `invalid`: MPC/runtime, score, global, required metrics, or required MPC data
  fail.

Hard constraints:

- `num_steps=800`
- `mpc.accepted=true`
- `mpc.accepted_windows=40`
- `mpc.used_baseline_fallback=false`
- runtime `<= 1.10x` same-machine control

Quality checks:

- score no worse than `best_s128 - 0.05`
- root/body-global/ee-global errors no worse than `1.05x best_s128`
- `control_delta_mean <= best_s128`
- `joint_acc_mean <= 1.02x best_s128`
- `contact_mismatch_rate <= best_s128`; for `qixing`, also keep the stricter
  `0.220` target already used by the local-first guardrail
- local transfer requires at least one of `joint_pos_error_mean`,
  `body_local_pos_error_mean`, and `ee_local_pos_error_mean` to improve by `3%`
  on `walk`/`qixing`, and no local metric may regress by more than `2%`

If `jump` is assessed in transfer mode, it keeps the stricter local target of at
least two local metrics improved by `3%`, plus the same `2%` local-regression
cap.

## Runner Behavior

Extend the existing local-first runner rather than adding a new script:

- `--candidate-set transfer` selects the transfer anchors.
- `--candidate-set recovery` selects the recovery matrix.
- `--assessment-mode transfer` writes `transfer_summary.csv`.
- If `--candidate-set transfer` is used and `--assessment-mode` is omitted, infer
  `transfer`.
- If `--candidate-set transfer` is used and `--motions` is omitted, default to
  `walk qixing`.
- If `--candidate-set recovery` is used and `--motions` is omitted, keep the
  existing `jump` screening default.
- Existing `middle`, `upper-bound`, explicit `--candidates`, `promotion`, and
  `upper-bound` behavior must remain unchanged.

`transfer_summary.csv` should include enough diagnostics to decide whether a
candidate passes transfer or needs recovery:

```text
rank,candidate,motion,status,transfer_class,failure_labels,
score_guardrail_passed,global_guardrail_passed,smooth_guardrail_passed,
contact_guardrail_passed,local_transfer_passed,mpc_guardrail_passed,
runtime_guardrail_passed,local_count_3,max_local_regression,score_delta,
max_global_ratio,contact_delta,control_delta_regression,joint_acc_regression,
summary_csv
```

Ranking should prioritize:

1. `transfer_pass`
2. `recovery_candidate`
3. `invalid`
4. more local metrics improved by `3%`
5. lower max local regression
6. higher score delta
7. lower max global ratio

## Execution Targets

Transfer dry run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_transfer_dryrun \
  --candidate-set transfer \
  --dry-run
```

Transfer GPU run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_transfer_gpu_20260622 \
  --candidate-set transfer \
  --execute
```

Recovery dry run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_recovery_dryrun \
  --candidate-set recovery \
  --assessment-mode promotion \
  --dry-run
```

## Success Criteria

Implementation success:

- targeted unit tests pass;
- dry-run artifacts record `candidate_set`, `assessment_mode`, candidate names,
  and motion defaults correctly;
- transfer dry-run writes planned rows to `transfer_summary.csv`;
- existing middle and upper-bound tests still pass.

Experiment success:

- at least one anchor is a `transfer_pass` on both `walk` and `qixing`, or
  `transfer_summary.csv` clearly shows whether local, smoothness, contact, or
  global tracking blocks transfer;
- recovery candidates only advance if they keep the fixed budget and pass the
  existing strict MPC/runtime/global/smooth/contact guardrails.
