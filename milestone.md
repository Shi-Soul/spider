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
