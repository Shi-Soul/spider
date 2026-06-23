# G1 WBC Local Middle-Gate Experiment Design

## Goal

Run the next G1 WBC low-sample MPC experiment stage without increasing rollout
work. Keep the current `best_s128` speed envelope and use a softer intermediate
screen to find reward directions that improve local/posture quality while
preserving global tracking, smoothness, contact, and runtime.

The stage is exploratory. Passing this middle gate does not make a candidate a
replacement for `best_s128`; it only promotes candidates to broader
`walk`/`qixing` validation.

## Fixed Speed Envelope

All screening candidates keep the current low-sample settings:

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

Runtime remains a guardrail: candidate duration must be no more than `1.10x`
same-machine `jg_s128_v14_control` for the same motion.

## Intermediate Promotion Gate

The screening motion is `jump`. A candidate can be promoted when it satisfies:

- score no worse than `best_s128` by more than `0.05`;
- `root_pos_error_mean`, `body_global_pos_error_mean`, and
  `ee_global_pos_error_mean` no worse than `1.05x best_s128`;
- at least two of `joint_pos_error_mean`, `body_local_pos_error_mean`, and
  `ee_local_pos_error_mean` improve by at least `3%`;
- `control_delta_mean <= best_s128`;
- `joint_acc_mean <= 1.02x best_s128`;
- `contact_mismatch_rate <= best_s128`;
- `mpc.accepted=true`, `mpc.accepted_windows=40`, `num_steps=800`, and no
  baseline fallback;
- runtime passes the same-machine `1.10x` guardrail.

For final replacement claims, stricter validation still applies after promotion:
three-motion comparison against `best_s128`, with the original `5%` local target
or a clear smooth/contact win that justifies the tradeoff.

## Candidate Matrix

The next stage should test smaller, less entangled reward changes than the failed
`L150` round.

Default screening candidates:

- `jg_s128_v14_control`: same reward weights as current v14.
- `jg_s128_v14_smooth015`: v14 plus `+15%` smoothness weights.
- `jg_s128_v14_smooth025`: v14 plus `+25%` smoothness weights.
- `jg_s128_v14_contact015`: v14 plus `+15%` contact weights.
- `jg_s128_v14_smooth015_contact015`: v14 plus both light smooth/contact.
- `jg_s128_L135_posture`: local/posture interpolation between prior L125 and L150.
- `jg_s128_L140_posture`: local/posture interpolation between prior L125 and L150.
- `jg_s128_L145_posture`: local/posture interpolation between prior L125 and L150.
- `jg_s128_L140_smooth015`: L140 plus light smoothness.
- `jg_s128_L140_contact015`: L140 plus light contact.
- `jg_s128_L145_smooth015_contact015`: L145 plus light smooth/contact.

Legacy candidates from the prior stage can remain addressable by explicit
`--candidates`, but they should not be the default matrix for this stage.

## Runner Behavior

The existing local-first runner remains the entry point. It should:

- default to the next-stage candidate matrix above;
- support a configurable `--local-improvement-pct` argument, defaulting to `3.0`;
- include the threshold in `experiment_plan.json`;
- use the configured threshold when writing `guardrail_summary.csv`;
- keep `promoted_commands.sh` deterministic and limited to the first two passed
  screening candidates in candidate order;
- continue supporting `--summarize-only`, `--execute`, `--candidates`,
  `--motions`, and `--promoted-motions`.

## Experiment Execution

After implementation:

1. Run unit tests and compile checks in the sandbox.
2. Run a dry-run into `/tmp/g1_wbc_local_middle_gate_dryrun`.
3. Run GPU screening outside the sandbox into
   `/tmp/g1_wbc_local_middle_gate_gpu_20260622`.
4. If `promoted_commands.sh` contains real commands, run promoted
   `walk`/`qixing` rollouts outside the sandbox.
5. Re-run summarization and report final guardrail status plus key metric
   deltas against `best_s128` and same-machine control.

## Success Criteria

Implementation success:

- tests pass;
- dry-run emits the planned next-stage matrix;
- guardrail summary uses `3%` by default and can be configured to `5%`.

Experiment success:

- at least one candidate passes the middle gate, or the result clearly shows
  which guardrail blocks each direction;
- no promoted candidate is accepted without the runtime, MPC, smooth, contact,
  and global guardrails.
