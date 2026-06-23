# G1 WBC Local Upper-Bound Experiment Design

## Goal

Find the local/posture frontier for low-sample G1 WBC MPC while keeping the
current `s128/i2/h40/c20/k8` rollout budget. This stage intentionally relaxes
replacement-style global/score/smooth/contact gates so we can learn how much
local fidelity is reachable before trying to recover a deployable balance.

The stage is exploratory. A frontier candidate is not promoted to replace
`best_s128`; it is a candidate for a later recovery stage.

## Fixed Budget

All candidates keep:

```text
samples: 128
iterations: 2
planning_horizon_steps: 40
control_steps: 20
knot_count: 8
root_pos_sigma: 0.04
root_rot_sigma: 0.10
joint_sigma: 0.18
seed: 0
max_steps: 800
```

## Hard Constraints

These remain mandatory:

- `num_steps=800`
- `mpc.accepted=true`
- `mpc.accepted_windows=40`
- `mpc.used_baseline_fallback=false`
- runtime no more than `1.10x` same-machine `jg_s128_v14_control`

## Relaxed Exploration Limits

The frontier search keeps only loose feasibility limits:

- `score >= best_s128_score - 0.25`
- max of root/body-global/ee-global ratios `<= 1.35x best_s128`

Smoothness and contact are recorded but do not block frontier classification.

## Candidate Matrix

Default upper-bound candidates:

- `jg_s128_v14_control`
- `jg_s128_L138_posture`
- `jg_s128_L140_posture`
- `jg_s128_L142_posture`
- `jg_s128_L145_posture`
- `jg_s128_L148_posture`
- `jg_s128_L150_posture`
- `jg_s128_L155_posture`
- `jg_s128_L142_smooth005`
- `jg_s128_L145_smooth005`
- `jg_s128_L148_smooth005`
- `jg_s128_L150_smooth005`

The `smooth005` suffix means a `+5%` multiplier on the v14 smooth terms:

```text
control_delta=0.945
action_delta=0.315
joint_acc=0.00315
joint_jerk=0.00084
```

## Frontier Classification

For each completed candidate:

- `local_frontier`: hard constraints pass, relaxed score/global pass, and either
  at least two local metrics improve by `5%` or all three improve by `3%`.
- `near_frontier`: hard constraints pass, relaxed score/global pass, and at
  least two local metrics improve by `3%`.
- `dead_end`: hard constraints and relaxed score/global pass, but local gains are
  below near-frontier.
- `invalid`: MPC/runtime/score/global relaxed constraints fail or metrics are
  missing.

Local metrics are:

- `joint_pos_error_mean`
- `body_local_pos_error_mean`
- `ee_local_pos_error_mean`

Ranking prioritizes:

1. frontier class (`local_frontier` before `near_frontier` before `dead_end`)
2. count of local metrics improved by `8%`
3. count improved by `5%`
4. count improved by `3%`
5. local composite improvement
6. score delta
7. max global ratio

## Runner Behavior

The existing runner gains explicit upper-bound mode:

- `--candidate-set middle` keeps the current middle-gate default.
- `--candidate-set upper-bound` selects the upper-bound matrix.
- `--assessment-mode promotion` keeps existing `guardrail_summary.csv`.
- `--assessment-mode upper-bound` writes `frontier_summary.csv` and does not
  generate promoted rollout commands.

Existing explicit `--candidates` remains supported.

## Execution

Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_local_upper_bound_gpu_20260622 \
  --candidate-set upper-bound \
  --assessment-mode upper-bound \
  --execute
```

Then inspect `frontier_summary.csv`. No `walk/qixing` promotion is expected in
this stage.
