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
- The L145 review is not a case where the metrics looked uniformly good but the
  render looked bad. The five metric groups already expose the main visual
  issues:
  - On `jump`, L145 is worse than the packaged SPIDER `g1_wbc_joint_global`
    baseline across aggregate/global/contact/smooth metrics. The saved run has
    score `-2.3948` versus baseline `-2.0780`, root `0.0925` versus `0.0425`,
    body global `0.0999` versus `0.0530`, ee global `0.1101` versus `0.0608`,
    contact mismatch `0.3681` versus `0.3550`, and control delta `0.4593`
    versus `0.3372`. `jump_repeat1` makes the root/global drift much worse.
  - On `walk` and `qixing`, L145 remains successful and tracking/contact quality
    is roughly competitive, but the smooth/control metrics reflect the visible
    jitter. `walk` control delta is `0.2344` versus baseline `0.1489`, and joint
    jerk is `130.4` versus `93.9`. `qixing` control delta is `0.2333` versus
    `0.1862`, and joint jerk is `106.5` versus `76.3`.
  - Some local/posture values improved relative to the low-sample search anchor,
    but the full five-class metric view does not support promotion once baseline
    smoothness and repeatability are considered.
- `jg_s128_L145_smooth005` should not be promoted as a three-motion solution:
  `walk` and `qixing` are viable transfer checks, but `jump` is not stable under
  saved reruns.

Updated conclusion:

- Treat the local upper-bound / transfer stage as a negative promotion result,
  even though it was useful diagnostically. The current five metric groups do
  reflect the major L145 visual issues; the weakness was that the promotion gate
  did not make baseline-relative smoothness and repeatability strong enough
  blockers.
- The next optimization stage should start from the previous `best_s128`
  configuration and artifacts, not from the local-frontier variants.
- Local-frontier candidates (`jg_s128_L142_smooth005`,
  `jg_s128_L148_posture`, `jg_s128_L145_smooth005`) should remain diagnostic
  ablations unless a later run clearly beats `best_s128` in the five metric
  groups, stays competitive with the packaged SPIDER baseline on smooth/control
  metrics, passes repeatability checks, and matches rendered motion quality.

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

## 2026-06-24 Mechanism Quality-Speed Stage

Maintenance note:

- `spider/milestone.md` is the active source of truth for SPIDER G1 WBC
  experiment milestones.
- The workspace-root `../milestone.md` is now treated as a historical snapshot
  and should not be maintained for new SPIDER G1 WBC conclusions.

Goal: identify the core factors that control optimization quality and inference
speed before running any full `bench_data` evaluation. The quality floor is the
current acceptable `best_s128`; the long-term target is the packaged SPIDER
`baseline_8192`.

Primary metric groups:

- global: `root_pos_error_mean`, `body_global_pos_error_mean`,
  `ee_global_pos_error_mean`
- local: `body_local_pos_error_mean`, `ee_local_pos_error_mean`
- smooth: `control_delta_mean`, `joint_acc_mean`
- contact: `contact_mismatch_rate`

Artifacts:

- Mechanism report:
  `outputs/g1_wbc_mechanism_20260623/mechanism_report.md`
- Stage A sample-budget summary:
  `outputs/g1_wbc_mechanism_20260623/stage_a_budget_curve/summary.csv`
- Jump repeat summary:
  `outputs/g1_wbc_mechanism_20260623/stage_a_jump_repeats/summary.csv`
- Stage B structure summaries:
  `outputs/g1_wbc_mechanism_20260623/stage_b_structure/**/summary.csv`
- Stage C mechanism summaries:
  `outputs/g1_wbc_mechanism_20260623/stage_c_mechanisms/**/summary.csv`
- Jump three-panel comparison video:
  `outputs/g1_wbc_mechanism_20260623/videos/jump_sweetpoint_best_s128_spider_baseline_3panel.mp4`

Completed experiment matrix:

- Stage A: cross-motion sample-budget curve on `jump`, `walk`, `qixing`;
  samples `64,96,128,192,256,512,1024,2048`; fixed
  `i2/h40/c20/k8`, sigma `0.04/0.10/0.18`, seed 0.
- Stage A repeat: jump `s128/s512/s2048`, seeds 1 and 2, merged with seed 0 for
  three-seed statistics.
- Stage B: jump structure probes at `s512`, including iterations, knot count,
  and paired horizon/control variants.
- Stage C: jump mechanism probes at `s128/s256/s512`, including warm start,
  command regularization, and command smoothness.

### Sample-Budget Result

Jump three-seed statistics for the base configuration
`i2/h40/c20/k8`, sigma `0.04/0.10/0.18`:

| samples | success | score mean | global sum | local sum | contact | control delta | joint acc | duration mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1/3 | -2.2193 | 0.2401 | 0.0716 | 0.3525 | 0.4540 | 225.1 | 115.1 s |
| 512 | 2/3 | -2.0536 | 0.1863 | 0.0696 | 0.3346 | 0.4457 | 217.4 | 188.7 s |
| 2048 | 2/3 | -2.1080 | 0.1770 | 0.0695 | 0.3533 | 0.4233 | 212.6 | 641.2 s |

Interpretation:

- `s512` is the current jump base sweetpoint: it is much better than `s128` on
  average score/global/contact, while avoiding the high cost and variance of
  `s2048`.
- `s2048` mainly buys smoother control and slightly better global tracking, but
  it is not a better score/contact Pareto point and costs roughly 3.4x the
  `s512` wall clock.
- `walk` does not justify high samples in this stage: `s128` is already near the
  score/contact sweet spot, while `s2048` costs about 6x more for negligible
  score gain.
- `qixing` shows high-sample improvement on score/contact/smooth, but its global
  tracking is non-monotonic. Its bottleneck should not be treated as pure sample
  budget without more targeted diagnosis.

Current jump sweetpoint configuration:

```text
method: g1_wbc_joint_global
samples: 512
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
max_steps: 800
```

Observed jump sweetpoint wall clock:

| seed | duration | score | contact |
| ---: | ---: | ---: | ---: |
| 0 | 203.488 s | -2.0607 | 0.3344 |
| 1 | 184.931 s | -1.9850 | 0.3156 |
| 2 | 177.787 s | -2.1152 | 0.3537 |

Mean wall clock: `188.735 s` for 800 policy steps on the current 4070 Laptop
environment.

### Structure Factors

Anchor: `s512/i2/h40/c20/k8`, score `-2.0607`, global sum `0.1947`,
contact `0.3344`, control delta `0.4401`, joint acc `217.7`, duration
`203.5 s`.

| variant | score | global sum | local sum | contact | control delta | joint acc | duration | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `i1` | -2.5858 | 0.4494 | 0.0845 | 0.3556 | 0.5078 | 244.7 | 117.4 s | Too little optimization depth. |
| `i3` | -1.9812 | 0.1453 | 0.0639 | 0.3413 | 0.4163 | 212.6 | 276.9 s | Best quality ceiling found in this stage. |
| `k4` | -2.6747 | 0.5538 | 0.0820 | 0.3419 | 0.3715 | 194.8 | 173.6 s | Smooth but under-expressive; tracking collapses. |
| `k12` | -2.3972 | 0.3229 | 0.0710 | 0.3594 | 0.5639 | 249.6 | 182.5 s | Higher-dimensional search is worse at s512. |
| `k16` | -2.8916 | 0.6371 | 0.0725 | 0.3587 | 0.6706 | 286.0 | 193.4 s | Not viable. |
| `h20/c10` | -8.8669 | 4.5793 | 0.0609 | 0.3713 | 0.2516 | 148.4 | 314.8 s | Short lookahead fails jump. |
| `h80/c20` | -2.1987 | 0.1638 | 0.0776 | 0.3662 | 0.4020 | 207.3 | 268.3 s | Better global, worse score/contact/local. |
| `h80/c40` | -2.2804 | 0.1893 | 0.0853 | 0.3775 | 0.4127 | 210.6 | 136.7 s | Faster, but below the quality floor. |

Interpretation:

- `i2/h40/c20/k8` is the current structure anchor.
- `i2` is the lowest acceptable optimization depth; `i3` is a quality-ceiling
  upgrade, not the speed anchor.
- `h40/c20` and `k8` are the tested-range sweetpoints. Nearby alternatives either
  fail tracking/contact or do not preserve score.

### Algorithm Mechanisms

Best speed-quality mechanism found: `mpc_command_reg_weight=0.005`.

| variant | samples | score | global sum | local sum | contact | control delta | joint acc | duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 128 | -2.2037 | 0.2204 | 0.0727 | 0.3562 | 0.4553 | 230.2 | 120.2 s |
| warm-best | 128 | -2.1205 | 0.1980 | 0.0724 | 0.3456 | 0.4496 | 226.9 | 113.5 s |
| warm-best | 256 | -2.1024 | 0.1729 | 0.0719 | 0.3469 | 0.4604 | 228.4 | 138.5 s |
| warm-best | 512 | -2.1267 | 0.2283 | 0.0775 | 0.3356 | 0.4501 | 222.1 | 198.3 s |
| warm-mean-0.8 | 128 | -2.2597 | 0.2565 | 0.0746 | 0.3481 | 0.4890 | 227.9 | 111.3 s |
| warm-mean-0.8 | 256 | -2.4045 | 0.2365 | 0.0715 | 0.4031 | 0.4543 | 225.0 | 133.9 s |
| warm-mean-0.8 | 512 | -2.1043 | 0.1914 | 0.0688 | 0.3450 | 0.4281 | 216.3 | 192.8 s |
| reg0.005 | 128 | -2.0561 | 0.1772 | 0.0681 | 0.3363 | 0.4372 | 221.1 | 112.0 s |
| reg0.005 | 256 | -2.0628 | 0.1804 | 0.0697 | 0.3381 | 0.4438 | 220.0 | 139.1 s |
| reg0.005 | 512 | -2.0562 | 0.2030 | 0.0671 | 0.3312 | 0.4300 | 216.1 | 184.9 s |
| smooth0.0001 | 128 | -2.1820 | 0.2150 | 0.0698 | 0.3519 | 0.4675 | 225.0 | 112.4 s |
| smooth0.0001 | 256 | -2.2210 | 0.2103 | 0.0701 | 0.3681 | 0.4397 | 221.5 | 135.9 s |
| smooth0.0001 | 512 | -2.0563 | 0.1798 | 0.0708 | 0.3400 | 0.4412 | 220.3 | 175.4 s |

Packaged jump `baseline_8192` reference:

```text
score: -2.0780
global sum: 0.1563
contact mismatch: 0.3550
control delta: 0.3372
joint acc: 191.9
```

Key comparison:

- `reg0.005/s128` reaches score `-2.0561` in `112.0 s`, beating the packaged
  baseline score and clearly improving over base `s128` on global/local/contact
  and smooth metrics. It is the strongest current speed-quality candidate.
- `i3/s512` reaches score `-1.9812` and global sum `0.1453`, beating the
  packaged baseline on score/global/contact. It is the strongest current quality
  candidate, but costs `276.9 s`.
- Both `reg0.005/s128` and `i3/s512` still lag the packaged baseline on smooth:
  control delta and joint acceleration remain higher than baseline.
- `warm-best` helps low samples but regresses at `s512`; it is a secondary
  low-sample assist, not the main solution.
- `warm-mean-0.8` is not recommended.
- `command_smooth_weight=0.0001` did not cleanly improve smoothness and can hurt
  contact/score at low samples.

Current factor ranking:

1. `command_reg_weight=0.005`: strongest quality-per-second lever.
2. `mpc_num_iterations`: `i2 -> i3` is the best quality lever and more efficient
   than increasing samples to `s2048`.
3. `mpc_num_samples`: useful up to `s512` on jump, weak/non-monotonic beyond.
4. `horizon/control`: sensitive; keep `h40/c20` as anchor.
5. `knot_count`: keep `k8`; `k4` under-expresses and `k12/k16` dilute search.
6. `warm_start`: `best` helps low samples only; `mean/0.8` is not reliable.
7. `command_smooth_weight=0.0001`: not a strong mechanism as configured.

### Video And Repeatability Notes

Three-panel jump comparison video:

```text
outputs/g1_wbc_mechanism_20260623/videos/jump_sweetpoint_best_s128_spider_baseline_3panel.mp4
```

Video panels:

1. `sweetpoint_s512_i2_h40_c20_k8_seed1`
2. `best_s128`
3. `spider_baseline_8192`

Metrics for the rendered trajectories:

| trajectory | score | root | body global | ee global | contact mismatch | control delta | joint acc | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sweetpoint s512 seed1 saved rollout | -2.0799 | 0.0568 | 0.0647 | 0.0722 | 0.3356 | 0.4359 | 215.6 | true |
| best_s128 visualized rerun | -2.2657 | 0.0801 | 0.0872 | 0.0935 | 0.3537 | 0.4710 | 223.2 | false |
| spider baseline 8192 | -2.0780 | 0.0425 | 0.0530 | 0.0608 | 0.3550 | 0.3372 | 191.9 | false |

Important repeatability finding:

- The same configuration on the same `jump` motion can produce different rollout
  results across seeds and even across saved reruns.
- Example: Stage A `s512 seed0` had score `-2.0607`, but a later saved-rollout
  rerun with the same explicit configuration produced score `-2.2609`.
- A later saved `s512 seed1` rerun produced score `-2.0799`, close to the
  sweetpoint and suitable for video review.
- Therefore, single rollout comparisons are not sufficient for configuration
  promotion. Future comparisons must use multi-seed or repeated runs and should
  report mean/std/min/max plus success rate across score/global/local/smooth/contact
  groups.

Recommended next experiments:

1. Repeat `reg0.005/s128` and `reg0.005/s256` on seeds 1 and 2, then transfer to
   `walk` and `qixing` only if jump repeatability holds.
2. Sweep command regularization around the discovered point:
   `0.001, 0.002, 0.005, 0.01`.
3. Test `reg0.005 + i3/s512` as a quality-ceiling candidate after the reg repeat
   check.
4. If smooth remains the main gap, test smaller command-smooth weights such as
   `0.00002` and `0.00005`; do not continue with `0.0001` as the default smooth
   penalty.
