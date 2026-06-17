# G1 WBC in the Spider MPC Framework

This note records what we changed for the G1 WBC MPC/RL integration. It is intentionally separate from the original `README.md`.

## Goal

The goal is to make G1 WBC MPC a Spider-native optimization task instead of keeping a separate G1-specific MPC implementation.

Concretely:

- Use Spider's generic sampled-MPC optimizer for sampling, knot interpolation, score weighting, iteration scheduling, and receding-horizon execution.
- Keep G1 WBC task logic only where it is physically necessary: motion loading, WBC policy rollout, G1 command construction, G1-specific score terms, and evaluation metrics.
- Remove the old G1-specific MPC algorithm implementation from `spider/tasks/g1_wbc/mpc.py`.
- Keep the optimization problem comparable to the old G1 WBC setting so that old-vs-new benchmark results remain meaningful.

## Optimization Problem

Each MPC window optimizes a horizon-length residual command sequence:

$$
U = \{u_0, u_1, \ldots, u_{H-1}\}.
$$

For G1 WBC, each control residual is interpreted as:

$$
u_t =
\left[
\Delta p^{root}_t,\,
\Delta r^{root}_t,\,
\Delta q^{joint}_t
\right].
$$

The residual is applied to the reference motion command:

$$
\hat{q}_t =
f(q^{ref}_t, u_t),
$$

where root position and joint coordinates are additive, root rotation is applied by converting the axis-angle residual to a quaternion and left-multiplying the reference root orientation, and joint coordinates are clamped to the G1 joint limits.

For a sampled candidate sequence \(U_i\), the WBC actor tracks the resulting command trajectory in MuJoCo/Warp rollout, then receives a scalar task score:

$$
J(U_i) =
-\sum_k w_k c_k(U_i).
$$

The terms \(c_k\) are the existing G1 WBC tracking, contact, force, and smoothness metrics.

## Spider-Native MPC Loop

The generic Spider loop is:

1. Keep a current control mean \(U\) over the planning horizon.
2. Sample noise on a small number of endpoint knots.
3. Interpolate knot noise to the dense horizon.
4. Form candidate sequences \(U_i = U + \epsilon_i\).
5. Roll out all candidates through the task adapter.
6. Compute candidate scores.
7. Select the top-scoring samples and compute softmax weights.
8. Update \(U\) by weighted mean, or by the best sample when configured.
9. Repeat for `max_num_iterations`.
10. Execute only the first `ctrl_steps`.
11. Shift the unexecuted suffix forward as the next window's warm start and append task-provided tail controls.

This is implemented in the generic optimizer layer, not in the G1 task directory.

## What Comes From Spider

The following behavior is inherited from Spider's optimizer framework:

- sampled-MPC candidate generation
- knot-based control parameterization
- endpoint knot interpolation to the dense horizon
- score normalization and top-sample softmax weighting
- `weighted_mean` and `best` control update modes
- noise schedule across optimization iterations
- receding-horizon execution
- warm start by shifting the remaining horizon
- optional rollout chunking through `rollout_batch_size`
- MPC metadata recording

Main files:

- `spider/optimizers/sampling.py`
- `spider/optimizers/sampling_fast.py`
- `spider/optimizers/receding.py`
- `spider/config.py`

## What Remains G1 WBC Specific

G1-specific code is now a task adapter, not a standalone MPC algorithm.

The adapter defines:

- how to convert residual controls into G1 qpos commands
- how to slice and pad the reference motion window
- how to run the WBC policy in MuJoCo/Warp
- how to compute G1 WBC objective scores from rollout terms
- how to execute selected controls and maintain recurrent rollout state
- how to reconstruct rollout artifacts for metrics and videos

Main file:

- `spider/simulators/g1_wbc.py`

The evaluation CLI wires the task adapter into Spider's optimizer:

- `spider/tasks/g1_wbc/evaluate.py`

## What Was Removed

The old G1-specific MPC implementation was deleted:

- `spider/tasks/g1_wbc/mpc.py`

That file used to own G1-specific sampling, presets, guided candidates, smoothing, acceptance/fallback logic, and the MPC loop itself. Those algorithmic responsibilities are no longer allowed in the G1 task package. The MPC loop now lives in Spider's generic optimizer layer.

## Objective Modes

The three existing G1 WBC modes are preserved:

- `g1_wbc_joint_global`
- `g1_wbc_joint`
- `g1_wbc_ee`

The distinction is in the task score weights, not in the MPC algorithm. The optimizer receives only scalar candidate scores and remains task-agnostic.

Reward weights can be passed from JSON via:

```bash
--mpc-reward-weights /path/to/weights.json
```

The JSON can be either a flat mapping or a mapping keyed by method name.

## Current Benchmark Setting

The controlled benchmark setting used for the old-vs-Spider comparison is:

- `samples = 8192`
- `iterations = 2`
- `planning_horizon_steps = 80`
- `control_steps = 20`
- `knot_count = 8`
- `temperature = 0.7`
- `root_pos_sigma = 0.08`
- `root_rot_sigma = 0.18`
- `joint_sigma = 0.28`
- `control_update_mode = weighted_mean`
- `torch_compile = false`
- checkpoint: `bc`
- max steps: `800`

The current completed comparison has 7 of 9 rows finished. The two pending rows are:

- `qixing/g1_wbc_joint`
- `qixing/g1_wbc_ee`

They are pending because no legal GPU was available at the time of writing. A local supervisor was left running to resume them once GPU 0 becomes free.

## Partial Quantitative Result

Completed score rows:

| motion | method | old score | Spider score | delta |
|---|---:|---:|---:|---:|
| walk | `g1_wbc_joint_global` | -1.070857 | -1.035968 | +0.034888 |
| walk | `g1_wbc_joint` | -1.777136 | -0.978463 | +0.798673 |
| walk | `g1_wbc_ee` | -0.952222 | -0.963920 | -0.011698 |
| jump | `g1_wbc_joint_global` | -2.208343 | -2.078044 | +0.130299 |
| jump | `g1_wbc_joint` | -6.402408 | -6.375596 | +0.026812 |
| jump | `g1_wbc_ee` | -2.196199 | -2.094176 | +0.102023 |
| qixing | `g1_wbc_joint_global` | -1.513477 | -1.372239 | +0.141238 |

Summary for completed rows:

- 6 of 7 completed score rows improved.
- 1 of 7 completed score rows regressed slightly.
- Completed score sum improved by `+1.222236`.
- Across completed quality metrics, 103 improved, 22 regressed, and 22 were unchanged.

## Videos

Old-vs-Spider videos were rendered from saved rollouts, without rerunning MPC. The available videos cover the same 7 completed rows and use the same reference ghost, camera, and setting.

Output directory in the local workspace:

```text
/home/bai/MPC-RL/spider_retarget_work_20260611/eval_reward_method_spider_native_current_knotfix_v14_20260617_local4090/videos_old_vs_spider_same_setting
```

Each rendered video has:

- left panel: old G1 WBC
- right panel: Spider-native G1 WBC
- cyan ghost: reference motion
- root error overlay
- 801 frames
- 1920x540 resolution
- 50 FPS

## Running a Single MPC Evaluation

Example:

```bash
python -m spider.tasks.g1_wbc.evaluate \
  --motion /home/bai/MPC-RL/data/focus_motions/walk/motion.npz \
  --motion-type isaaclab \
  --checkpoint bc \
  --method g1_wbc_joint_global \
  --max-steps 800 \
  --device cuda:0 \
  --output-dir /tmp/g1_wbc_spider_eval \
  --save-rollout \
  --mpc-knot-count 8 \
  --mpc-samples 8192 \
  --mpc-rollout-batch-size 0 \
  --mpc-iterations 2 \
  --mpc-planning-horizon-steps 80 \
  --mpc-control-steps 20 \
  --mpc-temperature 0.7 \
  --mpc-control-update-mode weighted_mean \
  --mpc-first-ctrl-noise-scale 0.5 \
  --mpc-last-ctrl-noise-scale 1.0 \
  --mpc-final-noise-scale 0.1 \
  --mpc-root-pos-sigma 0.08 \
  --mpc-root-rot-sigma 0.18 \
  --mpc-joint-sigma 0.28 \
  --no-mpc-torch-compile \
  --seed 0
```

Formal benchmark runs should use GPU. Do not run full formal evaluations on CPU.

## Design Boundary

The integration boundary is:

- Spider owns MPC.
- G1 WBC owns task semantics.

This means future changes to sampling strategy, horizon shifting, noise schedule, rollout chunking, or control update should be made in `spider/optimizers` or `spider/config.py`, not in `spider/tasks/g1_wbc`.

Task-specific changes such as WBC actor behavior, G1 command construction, body/EE metric terms, or G1 joint limits belong in `spider/simulators/g1_wbc.py` or the existing G1 WBC task modules.
