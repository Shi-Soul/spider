# G1 WBC MPC Milestone

Date: 2026-06-22

This note records the current state of the SPIDER `g1_wbc` low-sample MPC
experiment on the three testbed motions: `jump`, `walk`, and `qixing`.

## Scope

We are comparing three rollout variants:

- `no_mpc`: policy-only rollout, used as the no-MPC baseline.
- `baseline_8192_joint_global`: packaged SPIDER `g1_wbc_joint_global` baseline.
- `best_s128`: current low-sample MPC candidate using the best parameter set found
  so far.

The current low-sample candidate is:

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
reward_weights: g1_wbc_reward_weights_method_specific_v14_20260612.json
seed: 0
max_steps: 800
```

Important caveat for `jump`: the historical best `s128` metrics had
`score=-2.11644`, but that run did not save `rollout.npz`, so it cannot be
rendered directly. The visualized `jump` comparison uses a rerun with the same
parameters; that rerun has `score=-2.26573`.

## Artifacts

Rendered comparison videos, all H.264/yuv420p:

- `/home/wujs/Videos/spider-mpc/jump_no_mpc_baseline_best_params_rerun_h264.mp4`
- `/home/wujs/Videos/spider-mpc/walk_no_mpc_baseline_best_params_s128_h264.mp4`
- `/home/wujs/Videos/spider-mpc/qixing_no_mpc_baseline_best_params_s128_h264.mp4`

Full metrics comparison:

- `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/spider/outputs/g1_wbc_low_latency_tuning_20260619/render_inputs/full_metrics_no_mpc_baseline_best_s128_comparison.md`
- `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/spider/outputs/g1_wbc_low_latency_tuning_20260619/render_inputs/full_metrics_no_mpc_baseline_best_s128_comparison.csv`

Rollout outputs for the visualized `best_s128` runs:

- `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/spider/outputs/g1_wbc_low_latency_tuning_20260619/render_inputs/jump_best_s128_i2_h40_c20_k8_sig004_010_018_seed0`
- `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/spider/outputs/g1_wbc_low_latency_tuning_20260619/render_inputs/walk_best_s128_i2_h40_c20_k8_sig004_010_018_seed0`
- `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/spider/outputs/g1_wbc_low_latency_tuning_20260619/render_inputs/qixing_best_s128_i2_h40_c20_k8_sig004_010_018_seed0`

## Primary Metrics

The full `metrics.json` still keeps all metrics. For default comparison and
reporting, we use these primary metrics:

```text
score
root_pos_error_mean
body_global_pos_error_mean
ee_global_pos_error_mean
ee_local_pos_error_mean
contact_mismatch_rate
control_delta_mean
joint_acc_mean
```

These cover aggregate quality, global tracking, local tracking, contact quality,
and smoothness/control.

## Primary Metrics Table

Lower is better for all metrics except `score`, where higher is better.

### jump

| method | score | root | body global | ee global | ee local | contact mismatch | control delta | joint acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_mpc | -9.0896 | 1.5592 | 1.5591 | 1.5612 | 0.0381 | 0.3925 | 0.2546 | 147.941 |
| baseline_8192 | -2.0780 | 0.0425 | 0.0530 | 0.0608 | 0.0397 | 0.3550 | 0.3372 | 191.872 |
| best_s128_visualized | -2.2657 | 0.0801 | 0.0872 | 0.0935 | 0.0458 | 0.3538 | 0.4710 | 223.221 |
| best_s128_historical_metrics | -2.1164 | 0.0529 | 0.0621 | 0.0710 | 0.0429 | 0.3525 | 0.4525 | 224.116 |

### walk

| method | score | root | body global | ee global | ee local | contact mismatch | control delta | joint acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_mpc | -1.9265 | 0.1562 | 0.1562 | 0.1567 | 0.0332 | 0.2375 | 0.1448 | 86.072 |
| baseline_8192 | -1.0360 | 0.0432 | 0.0453 | 0.0471 | 0.0329 | 0.1344 | 0.1489 | 84.319 |
| best_s128 | -1.2671 | 0.0732 | 0.0776 | 0.0791 | 0.0355 | 0.1500 | 0.2441 | 117.789 |

### qixing

| method | score | root | body global | ee global | ee local | contact mismatch | control delta | joint acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_mpc | -1.9722 | 0.1280 | 0.1360 | 0.1407 | 0.0568 | 0.2338 | 0.1968 | 79.667 |
| baseline_8192 | -1.3722 | 0.0386 | 0.0460 | 0.0537 | 0.0470 | 0.2063 | 0.1862 | 73.073 |
| best_s128 | -1.5680 | 0.0638 | 0.0690 | 0.0751 | 0.0457 | 0.2275 | 0.2371 | 88.938 |

## Speed

The packaged baseline speed is from the 4090 offline testbed package. The
`best_s128` timings are from the current 4070 Laptop runs or recorded sweep
summary, so this is not a strict same-hardware benchmark.

| motion | baseline_8192 wall time | best_s128 wall time | observed speedup |
|---|---:|---:|---:|
| walk | 23.33 min | ~109.2 s | 12.8x |
| jump historical best | 29.18 min | 113.8 s | 15.4x |
| jump visualized rerun | 29.18 min | ~126.1 s | 13.9x |
| qixing | 22.82 min | ~109.1 s | 12.6x |

Theoretical rollout work reduction:

```text
baseline: 8192 * 2 * 80 * 40 = 52,428,800 rollout steps
best_s128: 128 * 2 * 40 * 40 = 409,600 rollout steps
ratio: 128x less rollout work
```

The candidate is still not real-time: for 800 policy steps, the visualized runs
are roughly 6.8x to 7.9x slower than real time. However, this is a large
improvement over the packaged 8192 baseline, which is roughly 85x to 109x slower
than real time.

## Interpretation

Current qualitative observation:

- The low-sample MPC result is meaningful progress.
- It is much faster than the 8192 baseline.
- The visualized motion is somewhat jittery.
- Global tracking looks reasonably good.
- Motion semantics are still slightly off.

Metrics support this interpretation with some nuance:

- Global position tracking is clearly much better than no-MPC and generally close
  to the 8192 baseline.
- Smoothness/control is consistently worse than both no-MPC and the 8192 baseline.
  This matches the observed jitter.
- Local/posture tracking is motion-dependent. It is clearly worse on `jump`,
  mixed on `walk`, and partly competitive on `qixing`.
- Contact/physics remains important. `walk` contact is close to baseline, while
  `qixing` contact mismatch is still much closer to no-MPC than to baseline.

## Current Conclusion

The current `best_s128` configuration is not yet a baseline-quality replacement,
but it is a useful milestone: it preserves much of the global tracking benefit of
8192-sample MPC while cutting observed wall time by roughly 12.5x to 15.4x.

The next optimization direction should not be simple sample reduction alone. The
main remaining issue is quality under low sample count, especially:

- reducing control jitter,
- improving local/posture fidelity on `jump`,
- improving contact quality on `qixing`,
- preserving the current global tracking gain.

Future reports should always include the full `metrics.json`, plus the primary
metrics table above for quick comparison.

## 2026-06-22 Local Upper-Bound Stage

Goal: keep the current `best_s128` rollout budget fixed
(`s128/i2/h40/c20/k8`, seed 0, 800 steps) and test whether stronger
local/posture rewards can reveal a useful local frontier without giving up MPC
acceptance, full-run coverage, or same-machine runtime.

Runner/artifacts:

- Design: `docs/superpowers/specs/2026-06-22-g1-wbc-local-upper-bound-design.md`
- Plan: `docs/superpowers/plans/2026-06-22-g1-wbc-local-upper-bound-plan.md`
- GPU rollout root: `/tmp/g1_wbc_local_upper_bound_gpu_20260622`
- Frontier summary:
  `/tmp/g1_wbc_local_upper_bound_gpu_20260622/frontier_summary.csv`

Screening rules:

- Hard constraints: MPC accepted, no baseline fallback, 40 accepted windows,
  800 rollout steps, and runtime <= 1.10x the same-machine control candidate.
- Relaxed exploration limits: score >= historical `best_s128` score - 0.25 and
  max(root/body-global/ee-global ratio) <= 1.35x.
- Frontier classes: `local_frontier` requires at least two local metrics improved
  by 5%, or all three improved by 3%; `near_frontier` requires at least two local
  metrics improved by 3%. Smooth/contact are recorded as diagnostics, not blockers
  in this upper-bound stage.

Top GPU rollout results on `jump`:

| rank | candidate | class | local 3% | local 5% | local composite | score delta | max global ratio | contact delta | control delta reg. | joint acc reg. |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `jg_s128_L142_smooth005` | `local_frontier` | 3 | 3 | 0.0767 | +0.0623 | 0.8723 | +0.00375 | -0.0468 | -0.0194 |
| 2 | `jg_s128_L148_posture` | `local_frontier` | 3 | 3 | 0.0665 | +0.1351 | 0.8420 | -0.00813 | -0.0603 | +0.00322 |
| 3 | `jg_s128_L145_smooth005` | `local_frontier` | 3 | 1 | 0.0407 | +0.0885 | 0.9730 | -0.0175 | +0.0188 | +0.0202 |
| 4 | `jg_s128_L155_posture` | `near_frontier` | 2 | 0 | 0.0311 | +0.1540 | 0.8493 | -0.0206 | -0.0587 | -0.00731 |
| 5 | `jg_s128_L148_smooth005` | `near_frontier` | 2 | 0 | 0.0304 | -0.0748 | 1.0618 | +0.0181 | +0.00020 | +0.0117 |

All 12 candidates completed with `status=ok`, `num_steps=800`,
`mpc_accepted=true`, `accepted_windows=40`, and
`mpc_used_baseline_fallback=false`. Candidate durations were 115.4-124.4 s, so
the upper-bound sweep stayed within the intended inference-time envelope.

Interpretation:

- The stage did find a local/posture frontier without increasing the rollout
  budget.
- `jg_s128_L148_posture` is the best clean local-reward anchor: it improves all
  three local metrics by at least 5%, improves score/global/contact, and only has
  a small joint-acc regression.
- `jg_s128_L142_smooth005` has the strongest local composite improvement and
  improves smoothness, but slightly worsens contact.
- `jg_s128_L145_smooth005` is more conservative and improves contact, but gives
  less local gain and slightly worsens smoothness.

## 2026-06-22 Local Frontier Visual Review Update

Follow-up transfer/recovery and saved-rollout visualization runs were completed
after the local upper-bound stage. The main visual-review candidate was
`jg_s128_L145_smooth005`, because it was the only candidate that passed both
`walk` and `qixing` transfer gates.

New review artifacts:

- Report:
  `outputs/g1_wbc_l145_three_motion_20260622/report/README.md`
- Metrics CSV:
  `outputs/g1_wbc_l145_three_motion_20260622/report/metrics_summary.csv`
- Videos:
  `outputs/g1_wbc_l145_three_motion_20260622/videos/walk_l145_rollout.mp4`
  `outputs/g1_wbc_l145_three_motion_20260622/videos/qixing_l145_rollout.mp4`
  `outputs/g1_wbc_l145_three_motion_20260622/videos/jump_l145_rollout_failed.mp4`

Saved rerun metrics for `jg_s128_L145_smooth005`:

| run | motion | success | score | root | body global | ee global | ee local | contact mismatch | control delta | joint acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| saved_walk | walk | true | -1.1314 | 0.0625 | 0.0672 | 0.0690 | 0.0350 | 0.1350 | 0.2344 | 112.318 |
| saved_qixing | qixing | true | -1.5123 | 0.0598 | 0.0662 | 0.0723 | 0.0467 | 0.2163 | 0.2333 | 94.262 |
| saved_jump | jump | false | -2.3948 | 0.0925 | 0.0999 | 0.1101 | 0.0443 | 0.3681 | 0.4593 | 226.184 |
| jump_repeat1 | jump | false | -2.7793 | 0.1817 | 0.1886 | 0.1956 | 0.0480 | 0.3625 | 0.4723 | 235.445 |

Manual render review verdict:

- The local-frontier stage did not produce a qualitative improvement over the
  previous visualized `best_s128` results.
- The metric gains in local/posture terms did not translate into better visual
  motion quality. The reviewed local-frontier outputs are judged worse than the
  previous `best_s128` experiment.
- `jg_s128_L145_smooth005` should not be promoted as a three-motion solution:
  `walk` and `qixing` are viable transfer checks, but `jump` is not stable under
  saved reruns.

Updated conclusion:

- Treat the local upper-bound / transfer stage as a negative result for quality
  improvement, even though it was useful diagnostically.
- The next optimization stage should start from the previous `best_s128`
  configuration and artifacts, not from the local-frontier variants.
- Local-frontier candidates (`jg_s128_L142_smooth005`,
  `jg_s128_L148_posture`, `jg_s128_L145_smooth005`) should remain diagnostic
  ablations unless a later run clearly beats `best_s128` in both metrics and
  rendered motion quality.

## 2026-06-23 Fast-Quality Pareto Stage Plan

Goal: determine whether the remaining gap to the packaged SPIDER
`g1_wbc_joint_global` baseline is primarily a sample-budget limit, a
repeatability/stability limit, or a reward/gating limit. This stage returns to
the original v14 `best_s128` reward anchor rather than continuing from the
local-frontier variants.

Artifacts:

- Design:
  `docs/superpowers/specs/2026-06-23-g1-wbc-fast-quality-pareto-design.md`
- Plan:
  `docs/superpowers/plans/2026-06-23-g1-wbc-fast-quality-pareto-plan.md`
- Default GPU output root:
  `/tmp/g1_wbc_fast_quality_pareto_20260623`
- Transfer output root, if budgets are promoted:
  `/tmp/g1_wbc_fast_quality_pareto_transfer_20260623`
- Decision report:
  `pareto_decision.md`, written next to `pareto_summary.csv`

Stage A repeatability matrix:

```text
candidate: jg_s128_v14_control
motions: jump, walk, qixing
samples: 128
seeds: 0,1,2
```

Stage B jump sample-budget ladder:

```text
candidate: jg_s128_v14_control
motion: jump
samples: 64,96,128,192,256
seeds: 0,1,2
fixed MPC params: i2/h40/c20/k8, sigma 0.04/0.10/0.18, max_steps 800
```

Stage C transfer rule:

- Promote the fastest `baseline_close` budget if one exists.
- Otherwise promote up to two `promising_budget` rows.
- If completed rows improve toward baseline but fail repeat hard constraints,
  classify the bottleneck as stability/repeatability first.
- If no `promising_budget` exists through `s256`, stop sample escalation and
  design reward/gating repair from `best_s128`.
- If promoted rows are `unstable`, run stability/acceptance repair before visual
  review.

Pareto result classes:

- `baseline_close`: stable repeats, score within 0.25 of the SPIDER baseline,
  and mean max-global ratio to SPIDER baseline <= 1.35.
- `promising_budget`: stable repeats, score better than `best_s128`, and mean
  max-global ratio to `best_s128` <= 1.00.
- `stable_but_low_quality`: stable repeats but not enough quality gain.
- `unstable`: at least one repeat is usable but hard constraints, expected seeds,
  or required metrics are inconsistent.
- `invalid`: no usable completed repeat exists for the group.

The runner also writes `pareto_decision.md` with one of:
`pending`, `sample_budget_likely`, `sample_budget_partial`,
`stability_likely`, or `reward_gating_likely`.

Expected interpretation:

- A monotonic quality gain from `s128` to `s192/s256` means sample budget is a
  likely bottleneck; promote the fastest stable budget to transfer and visual
  review.
- Little or no gain through `s256` means reward/gating/dynamics alignment is the
  likely bottleneck; do not keep escalating samples.
- Quality gains with missing/failed repeats mean stability is the bottleneck;
  repair acceptance/repeatability before any visual promotion.

### 2026-06-23 Fast-Quality Pareto Stage B Result

Executed on GPU:

```text
output_root: /tmp/g1_wbc_fast_quality_pareto_20260623
candidate: jg_s128_v14_control
motion: jump
samples: 64,96,128,192,256
seeds: 0,1,2
```

Artifacts:

- Raw sweep summary:
  `/tmp/g1_wbc_fast_quality_pareto_20260623/candidates/jg_s128_v14_control/summary.csv`
- Pareto summary:
  `/tmp/g1_wbc_fast_quality_pareto_20260623/pareto_summary.csv`
- Decision report:
  `/tmp/g1_wbc_fast_quality_pareto_20260623/pareto_decision.md`

Observed jump Pareto rows:

| samples | class | success | score_mean | delta_vs_baseline | max_global_ratio_vs_baseline | duration_sec_mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 64 | unstable | 2/3 | -2.304 | -0.226 | 2.166 | 110.5 |
| 96 | unstable | 2/3 | -2.322 | -0.244 | 2.177 | 111.0 |
| 128 | unstable | 1/3 | -2.239 | -0.161 | 1.718 | 115.1 |
| 192 | unstable | 1/3 | -2.231 | -0.153 | 1.731 | 126.5 |
| 256 | unstable | 1/3 | -2.143 | -0.065 | 1.343 | 141.5 |

Decision:

- `pareto_decision.md` conclusion: `stability_likely`.
- `s256` is the strongest quality signal: mean score is within `0.065` of the
  packaged SPIDER baseline and the mean max-global ratio to baseline is `1.343`,
  inside the `1.35` threshold.
- However, `s256` only succeeds on `1/3` seeds. All sample budgets fail
  repeat-stability through `metric_success`, despite clean process completion,
  `800` steps, accepted MPC, `40/40` accepted windows, and no baseline fallback.
- Do not run walk/qixing transfer from this stage. The next experiment should
  repair acceptance/repeatability around the `s256` budget before visual
  promotion.

`metric_success` definition:

- `root_pos_error_mean < 0.25`
- `root_rot_error_mean < 0.60`
- `ee_global_pos_error_mean < 0.25`
- `ee_local_pos_error_mean < 0.20`
- `contact_mismatch_rate < 0.35`

The `s256` seed-level hard-gate breakdown shows that the failures are contact
threshold misses rather than root/EE tracking failures:

| seed | success | score | root_pos | root_rot | ee_global | ee_local | contact_mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | false | -2.188 | 0.0549 | 0.1148 | 0.0750 | 0.0470 | 0.3594 |
| 1 | true | -2.125 | 0.0652 | 0.1026 | 0.0811 | 0.0435 | 0.3400 |
| 2 | false | -2.118 | 0.0510 | 0.1108 | 0.0681 | 0.0443 | 0.3525 |

Current interpretation:

- The continuous quality signal is real: `s256` is close to SPIDER baseline by
  average score and global tracking ratio.
- The repeatability failure is narrow: two failed seeds sit just above the
  `contact_mismatch_rate < 0.35` hard gate.
- The next experiment should diagnose contact false-positive/false-negative
  timing for the failed seeds and repair contact/acceptance stability before
  increasing samples further or doing walk/qixing transfer.
