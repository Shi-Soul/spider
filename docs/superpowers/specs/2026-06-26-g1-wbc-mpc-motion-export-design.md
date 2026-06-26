# G1 WBC MPC Motion Export Design

## Goal

Make G1 WBC MPC evaluation outputs directly usable by `tracking-bfm` play by
exporting the optimized MPC command as a standard motion `.npz` whenever an MPC
method is evaluated with `--save-rollout`.

## Current State

`spider.tasks.g1_wbc.evaluate` currently writes two files when `--save-rollout`
is used with an MPC method:

- `rollout.npz`: the actual policy rollout produced by executing the selected
  command in the MuJoCo/Warp WBC environment.
- `mpc_command.npz`: a SPIDER debugging artifact containing the refined command
  trajectory and candidate scores.

`mpc_command.npz` is not a `tracking-bfm` motion file. Its command fields are
prefixed with `command_`, it retains a singleton environment dimension, and it
does not currently save `command_body_lin_vel_w` or
`command_body_ang_vel_w`.

## Desired Behavior

When `--save-rollout` is passed and the selected method produces `mpc_result`,
`evaluate.py` writes:

```text
rollout.npz
mpc_command.npz
mpc_motion.npz
```

`mpc_motion.npz` is the canonical `tracking-bfm` motion export. It uses
MuJoCo joint and body order and can be played with:

```bash
MOTION_FILE=/path/to/mpc_motion.npz \
MOTION_TYPE=mujoco \
./scripts/play.sh
```

No extra file is written for `no_mpc`, because the input motion is already the
reference motion and no optimized MPC command exists.

## File Schemas

`mpc_command.npz` remains a SPIDER debugging artifact and keeps its existing
fields. It gains two additional arrays:

```text
command_body_lin_vel_w  (T, 1, 30, 3)
command_body_ang_vel_w  (T, 1, 30, 3)
```

`mpc_motion.npz` contains exactly the standard motion fields needed by
`tracking-bfm`:

```text
fps              scalar float32
joint_pos        (T, 29) float32
joint_vel        (T, 29) float32
body_pos_w       (T, 30, 3) float32
body_quat_w      (T, 30, 4) float32
body_lin_vel_w   (T, 30, 3) float32
body_ang_vel_w   (T, 30, 3) float32
motion_type      scalar string, "mujoco"
```

The source for `mpc_motion.npz` is `mpc_result.command`, not `rollout.npz`.
This preserves the optimized command that MPC asked the WBC policy to track,
rather than the policy's realized tracking result.

## Data Flow

MPC optimization still operates in the existing 35-dimensional qpos-delta space:

```text
root position delta 3
root rotation axis-angle delta 3
joint position delta 29
```

Those qpos deltas are converted into a full `G1CommandBatch` through
`command_batch_from_qpos_trajectory()`. The exported `mpc_motion.npz` is a
schema conversion of that in-memory command batch:

```text
command_joint_pos[:, 0]      -> joint_pos
command_joint_vel[:, 0]      -> joint_vel
command_body_pos_w[:, 0]     -> body_pos_w
command_body_quat_w[:, 0]    -> body_quat_w
command_body_lin_vel_w[:, 0] -> body_lin_vel_w
command_body_ang_vel_w[:, 0] -> body_ang_vel_w
command.fps                  -> fps
```

The singleton environment dimension is removed because `tracking-bfm` expects a
single motion sequence, not a batched command tensor.

## Error Handling

The exporter should fail loudly if asked to export a batched command with
`num_envs != 1`. Current MPC evaluation uses `num_envs=1`, so this protects the
schema from silently exporting the wrong environment if future code changes the
batch shape.

The exporter should write float arrays as `float32` and `motion_type` as the
scalar string `"mujoco"`.

## Testing

Add focused unit tests for the export helpers in `evaluate.py`:

- `_save_mpc_result()` writes the existing fields plus
  `command_body_lin_vel_w` and `command_body_ang_vel_w`.
- `_save_tracking_bfm_motion()` writes the exact standard motion key set and
  removes the singleton environment dimension.
- `_save_tracking_bfm_motion()` rejects multi-env commands with a clear
  `ValueError`.

These tests use small fake command/result objects and do not run MuJoCo, Warp,
or policy inference.
