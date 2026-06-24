# G1 HOI Standard Usage

This branch has two independent standard ways to apply SPIDER to a G1
human-object interaction motion:

- `g1 retarget hoi`: run the SPIDER retargeting pipeline on a HOI dataset item.
- `g1 wbc mpc hoi`: run the G1 WBC policy-in-the-loop MPC solver on a full-state
  HOI reference.

These are parallel workflows. `g1 wbc mpc hoi` does not require the output of
`g1 retarget hoi`. Chain them only when that is explicitly the experiment you
want to run.

## Environment

Commands below assume this checkout:

```bash
cd /home/bai/MPC-RL/spider-g1-wbc-interaction
export PYTHONPATH=/home/bai/MPC-RL/spider-g1-wbc-interaction
```

This workspace has historically been run with:

```bash
/home/bai/anaconda3/bin/python
```

Use a CUDA device for full retargeting or WBC MPC runs:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Input Contracts

### Contract A: G1 Retarget HOI

Use this path when the source motion is a dataset HOI item that should be
retargeted to G1 inside SPIDER.

The standard processed dataset location is:

```text
${DATASET_DIR}/processed/${DATASET_NAME}/unitree_g1/humanoid_object/${TASK}/${DATA_ID}/
```

The retargeting runner resolves:

- `scene.xml` from the parent task directory.
- `trajectory_kinematic.npz` from the data-id directory.

The kinematic trajectory file is expected to contain `qpos` and `qvel`, and can
also contain `ctrl`, `contact`, and `contact_pos`. If your HOI source is not in
this layout, add a dataset adapter that writes this processed layout first.

### Contract B: G1 WBC MPC HOI

Use this path when the source motion is already a full-state interaction
reference for one complete MuJoCo scene.

Required inputs:

- `HOI_MODEL`: a complete interaction MuJoCo XML containing the G1 robot and
  all movable objects.
- `HOI_MOTION`: an `.npz` file whose `qpos` last dimension equals the XML
  model `nq`. `qvel` is optional; if missing, it is finite-differenced.

Accepted `HOI_MOTION` keys:

```text
qpos: (T, nq) or (..., nq)
qvel: (T, nv) or (..., nv), optional
dt: scalar, optional
fps: scalar, optional
```

If neither `dt` nor `fps` exists in the motion file, pass `--source-dt`
explicitly. The WBC runner resamples the reference to the policy timestep.

This contract is independent of the retargeting output contract. The full-state
reference can come from any source as long as it matches `HOI_MODEL`.

## Standard Usage 1: G1 Retarget HOI

For a processed humanoid-object dataset item:

```bash
DATASET_DIR=/home/bai/MPC-RL/spider/example_datasets
DATASET_NAME=omomo
TASK=move_largebox
DATA_ID=0

/home/bai/anaconda3/bin/python examples/run_mjwp.py \
  +override=humanoid_object \
  dataset_dir="${DATASET_DIR}" \
  dataset_name="${DATASET_NAME}" \
  task="${TASK}" \
  data_id="${DATA_ID}" \
  robot_type=unitree_g1 \
  embodiment_type=humanoid_object \
  viewer=mujoco-rerun \
  rerun_spawn=true \
  save_video=true \
  save_info=true \
  max_sim_steps=-1
```

Outputs are written to:

```text
${DATASET_DIR}/processed/${DATASET_NAME}/unitree_g1/humanoid_object/${TASK}/${DATA_ID}/
```

Important outputs:

- `trajectory_mjwp.npz`: physics-retargeted G1 HOI trajectory.
- `visualization_mjwp.mp4`: rendered retargeting video when `save_video=true`.
- `config.yaml`: saved run configuration when config saving is enabled.

For HDMI/mjlab tasks, use the HDMI entrypoint instead:

```bash
TASK=move_suitcase
DATA_ID=1

/home/bai/anaconda3/bin/python examples/run_hdmi.py \
  task=${TASK} \
  joint_noise_scale=0.2 \
  knot_dt=0.2 \
  ctrl_dt=0.04 \
  horizon=0.8 \
  +data_id=${DATA_ID} \
  viewer=mujoco-rerun \
  rerun_spawn=true \
  +save_rerun=true \
  +save_metrics=false \
  max_sim_steps=-1
```

HDMI outputs include:

- `trajectory_kinematic.npz`: HDMI environment reference.
- `trajectory_hdmi.npz`: HDMI retarget rollout.
- `visualization_hdmi.mp4`: rendered video when frames are produced.

## Standard Usage 2: G1 WBC MPC HOI

Set the full-state scene and motion directly:

```bash
HOI_MODEL=/path/to/interaction_scene.xml
HOI_MOTION=/path/to/full_state_hoi_reference.npz
OUTPUT_DIR=/home/bai/MPC-RL/results_spider/g1_wbc_mpc_hoi
SOURCE_DT=0.03333333333333333

/home/bai/anaconda3/bin/python -m spider.tasks.g1_wbc_interaction.evaluate \
  --method g1_wbc_interaction \
  --checkpoint bc \
  --device cuda:0 \
  --model-path "${HOI_MODEL}" \
  --motion "${HOI_MOTION}" \
  --source-dt "${SOURCE_DT}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-rollout \
  --save-video \
  --video-width 960 \
  --video-height 720 \
  --video-fps 30 \
  --no-use-cuda-graph \
  --no-forward-after-step \
  --mpc-samples 512 \
  --mpc-rollout-batch-size 128 \
  --mpc-iterations 2 \
  --mpc-planning-horizon-steps 36 \
  --mpc-control-steps 12 \
  --mpc-knot-count 6 \
  --mpc-temperature 0.25 \
  --mpc-root-pos-sigma 0.025 \
  --mpc-root-rot-sigma 0.045 \
  --mpc-joint-sigma 0.09 \
  --robot-bad-floor-contact-weight 45 \
  --robot-bad-floor-force-excess-weight 10 \
  --robot-contact-mismatch-weight 6 \
  --robot-contact-switch-weight 12 \
  --robot-contact-force-delta-weight 2.5 \
  --robot-contact-false-positive-weight 1.5 \
  --robot-contact-false-negative-weight 0.4 \
  --robot-body-global-pos-weight 4 \
  --robot-body-global-rot-weight 0.8 \
  --robot-ee-global-pos-weight 3 \
  --robot-ee-global-rot-weight 0.4 \
  --robot-ee-local-pos-weight 2 \
  --robot-ee-local-rot-weight 0.5 \
  --robot-root-pos-weight 5 \
  --robot-root-rot-weight 1.5 \
  --robot-joint-pos-weight 0.8 \
  --robot-action-delta-weight 0.6 \
  --robot-joint-acc-weight 0.006 \
  --robot-joint-jerk-weight 0.0012 \
  --control-delta-weight 1.8 \
  --object-pos-weight 70 \
  --object-rot-weight 6 \
  --object-final-pos-weight 150 \
  --object-final-rot-weight 12 \
  --object-vel-weight 0.1
```

Add `--baseline /path/to/baseline.npz` only when you want object-tracking
metrics and a baseline comparison video against another full-state trajectory.
It is not required for WBC MPC solving.

Important outputs:

- `metrics.json`: method metadata and robot/object metrics.
- `trajectory_mpc_rl_object.npz`: full-state MPC+RL HOI trajectory in
  retarget-style `qpos/qvel/time/ctrl` arrays on the standardized evaluation
  time grid.
- `trajectory_mjwp.npz`: same rollout exported under the compatibility name.
- `rollout.npz`: native rollout tensors for debugging.
- `mpc_command.npz`: optimized command/control trace and candidate scores.
- `visualization_ours_reference.mp4`: MPC+RL rollout against the reference.
- `visualization_baseline_reference.mp4`: only when `--baseline` is supplied.

Use `--reward-mode interaction` for the standard HOI objective. Use
`--reward-mode retarget` only when the experiment is pure qpos/qvel tracking.

## Standard Evaluation

All G1 HOI motion evaluation should go through
`spider.tasks.g1_wbc_interaction.evaluate`.

The evaluator now uses one source-grid rule:

- The reference `--motion` provides the canonical time grid.
- Every evaluated full-state trajectory is sampled on that source time grid.
- When comparing a main result with `--baseline`, both are evaluated on the
  same contiguous source-grid interval covered by both trajectories.
- There is no baseline-to-main and main-to-baseline bidirectional resampling.
- The main result and baseline result emit the same `metrics` key set.

To score one saved retarget output by itself:

```bash
HOI_MODEL=/path/to/interaction_scene.xml
HOI_MOTION=/path/to/full_state_hoi_reference.npz
SAVED_TRAJECTORY=/path/to/trajectory_mjwp.npz
OUTPUT_DIR=/home/bai/MPC-RL/results_spider/eval_retarget_hoi

/home/bai/anaconda3/bin/python -m spider.tasks.g1_wbc_interaction.evaluate \
  --method static_qpos \
  --device cuda:0 \
  --model-path "${HOI_MODEL}" \
  --motion "${HOI_MOTION}" \
  --saved-qpos "${SAVED_TRAJECTORY}" \
  --output-dir "${OUTPUT_DIR}"
```

To score WBC MPC and retarget with the same standardized metrics in one run,
pass the retarget trajectory as `--baseline` to the WBC MPC command. The JSON
will contain:

- `metrics`: standardized metrics for the main method.
- `baseline.metrics`: the same standardized metric keys for the baseline.
- `comparison`: scalar deltas for matching numeric metric keys.

## Applying Either Method To A New HOI Motion

For `g1 retarget hoi`, adapt the source motion into the SPIDER processed dataset
layout for `unitree_g1/humanoid_object`, then run the retarget command for that
`DATASET_NAME`, `TASK`, and `DATA_ID`.

For `g1 wbc mpc hoi`, adapt the source motion into a full-state `HOI_MOTION`
that matches one complete `HOI_MODEL`, then run the WBC MPC command directly.
This adaptation should not be described as using the retarget pipeline unless
the experiment intentionally uses retarget output as the reference.
