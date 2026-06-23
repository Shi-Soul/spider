# G1 WBC Fast-Quality Pareto Experiment Design

## Goal

Determine whether the current gap between low-sample G1 WBC MPC and the
packaged SPIDER `g1_wbc_joint_global` baseline is primarily a sample-budget
limit, a repeatability/stability limit, or a reward/gating limit.

The target is not to find the final replacement in one sweep. The target is to
identify the fastest budget that is still worth promoting into visual review and
to avoid more reward tuning when the bottleneck is actually sampling or
stability.

## Current Anchor

The anchor remains the previous `best_s128` setup:

```text
method: g1_wbc_joint_global
samples: 128
iterations: 2
planning_horizon_steps: 40
control_steps: 20
knot_count: 8
sigma_triplets: 0.04,0.10,0.18
temperature: 0.7
guided_candidate: true
acceptance_gate: true
reward_weights: g1_wbc_reward_weights_method_specific_v14_20260612.json
seed: 0
max_steps: 800
```

The local-frontier variants remain diagnostic ablations. They should not be the
starting point for this stage because visual review judged them worse than the
previous `best_s128` output.

## Stage A: Repeatability Baseline

Run the unmodified v14 reward anchor at `s128/i2/h40/c20/k8` on all three
motions with repeated seeds.

Default matrix:

```text
candidate: jg_s128_v14_control
motions: jump, walk, qixing
samples: 128
seeds: 0,1,2
```

Purpose:

- Measure saved-rerun variance before changing the budget.
- Check whether `jump` is consistently unstable or only fails on some seeds.
- Establish same-machine control durations for runtime ratios.

Expected outcomes:

- If `jump` has high variance at `s128`, stability/gating is a first-class
  blocker and a single-seed ladder is not trustworthy.
- If all three motions are repeatable but below baseline quality, the next
  question is whether extra samples improve quality enough.

## Stage B: Jump Sample-Budget Ladder

Run the unmodified v14 reward anchor on `jump` across a compact sample ladder.

Default matrix:

```text
candidate: jg_s128_v14_control
motion: jump
samples: 64,96,128,192,256
seeds: 0,1,2
```

Purpose:

- Estimate the quality/runtime Pareto curve on the hardest motion.
- Separate sample-budget failure from reward/gating failure.
- Identify the smallest sample count that deserves transfer checks.

Expected outcomes:

- If quality improves monotonically and `s192` or `s256` approaches the SPIDER
  baseline, the bottleneck is sample budget. Promote the fastest stable budget.
- If quality does not improve materially from `s128` to `s256`, the bottleneck
  is reward/gating/dynamics alignment. Stop increasing samples and design a
  repair stage.
- If higher budgets improve average metrics but still have failed reruns, the
  bottleneck is stability. Tune acceptance/repeatability before reward weights.

## Stage C: Transfer Check For Promising Budgets

Promote at most two budgets from Stage B to `walk` and `qixing`.

Promotion candidates are selected in this order:

1. Valid hard constraints on all completed `jump` repeats.
2. Better mean score than `best_s128` on `jump`.
3. Lower mean max-global ratio versus `best_s128` on `jump`.
4. Lower runtime, used as the tie-breaker.

Default promoted motions:

```text
motions: walk,qixing
seeds: 0,1,2
```

Purpose:

- Avoid overfitting budget choice to `jump`.
- Confirm that a budget that repairs `jump` does not regress `walk` or `qixing`.

## Classification

Every completed sample/motion group is assigned one class:

- `baseline_close`: hard constraints pass, all repeats succeed, mean score is at
  least SPIDER baseline score minus `0.25`, and the mean max-global ratio to
  SPIDER baseline is at most `1.35`.
- `promising_budget`: hard constraints pass, all repeats succeed, mean score is
  better than `best_s128`, and mean max-global ratio to `best_s128` is at most
  `1.00`.
- `stable_but_low_quality`: hard constraints pass, all repeats succeed, but the
  budget does not beat `best_s128` enough to promote.
- `unstable`: at least one repeat completes but hard constraints or success are
  inconsistent across repeats.
- `invalid`: no usable completed repeats exist for the group.

Hard constraints:

- `status=ok`
- `metric_success=true`
- `metric_num_steps=800`
- `mpc_accepted=true`
- `mpc_accepted_windows=40`
- `mpc_used_baseline_fallback=false`

## Primary Metrics

The Pareto summary records these metrics as mean values per
`motion/samples` group:

- `score`
- `root_pos_error_mean`
- `body_global_pos_error_mean`
- `ee_global_pos_error_mean`
- `ee_local_pos_error_mean`
- `contact_mismatch_rate`
- `control_delta_mean`
- `joint_acc_mean`
- `duration_sec`

It also records:

- completed repeat count
- successful repeat count
- accepted repeat count
- baseline-fallback count
- mean score delta versus `best_s128`
- mean score delta versus SPIDER baseline
- mean max-global ratio versus `best_s128`
- mean max-global ratio versus SPIDER baseline

## Runner Behavior

The existing local-first runner gains a Pareto mode:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_local_first_stage.py \
  --output-root /tmp/g1_wbc_fast_quality_pareto_20260623 \
  --candidate-set pareto \
  --assessment-mode pareto \
  --motions jump \
  --samples 64 96 128 192 256 \
  --seeds 0 1 2 \
  --execute
```

Dry-run mode writes:

- `experiment_plan.json`
- `planned_commands.sh`
- `pareto_summary.csv` with planned rows
- `pareto_decision.md` with a pending conclusion until completed rows exist
- candidate reward weights under `reward_weights/`

Summarize-only mode reads existing candidate `summary.csv` files and writes a
fresh `pareto_summary.csv` plus `pareto_decision.md`.

## Decision Rules

After Stage B:

- Promote the fastest `baseline_close` budget if one exists.
- Otherwise promote up to two `promising_budget` rows.
- If completed rows improve toward baseline but fail repeat hard constraints,
  classify the bottleneck as `stability_likely`; do not reinterpret the lack of
  a promotable `promising_budget` row as reward/gating failure.
- If no `promising_budget` exists through `s256`, stop sample escalation and
  design a reward/gating repair stage from the original `best_s128` anchor.
- If any promoted budget is `unstable`, run a stability repair before visual
  review.

The runner writes these rules into `pareto_decision.md` as one of:
`pending`, `sample_budget_likely`, `sample_budget_partial`,
`stability_likely`, or `reward_gating_likely`.

## Visual Review Gate

Metrics are not enough for final promotion. A budget can enter visual review only
after it passes the Pareto summary gates. A budget can replace `best_s128` only
after rendered motion quality is judged at least as good as the previous
visualized `best_s128` result and closer to the SPIDER baseline on all three
motions.
