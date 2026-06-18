# G1 WBC Interaction Task

This task runs the G1 WBC policy-in-the-loop MPC rollout on a complete
humanoid-object MuJoCo model. The model XML is the single scene spec: robot,
objects, contacts, materials, inertial parameters, and object joints are read
from that file. The code discovers non-robot movable subtrees as objects;
object DOFs are present in simulation and scoring but never part of the MPC
decision vector.

Full-run inputs used for OMOMO `move_largebox` data id 0:

```bash
python -m spider.tasks.g1_wbc_interaction.evaluate \
  --method g1_wbc_interaction \
  --checkpoint /home/bai/MPC-RL/wxy/0608_ckpt_bc/model_8000.pt \
  --device cuda:0 \
  --model-path /home/bai/MPC-RL/spider/example_datasets/processed/omomo/unitree_g1/humanoid_object/move_largebox/scene.xml \
  --motion /home/bai/MPC-RL/spider/example_datasets/processed/omomo/unitree_g1/humanoid_object/move_largebox/0/trajectory_kinematic.npz \
  --baseline /home/bai/MPC-RL/spider/example_datasets/processed/omomo/unitree_g1/humanoid_object/move_largebox/0/trajectory_mjwp.npz \
  --output-dir /home/bai/MPC-RL-g1-wbc-interaction/results/move_largebox_0_full_stable_object_i2_s512 \
  --save-rollout --save-video \
  --video-width 960 --video-height 720 --video-fps 30 \
  --no-use-cuda-graph --no-forward-after-step \
  --mpc-samples 512 --mpc-rollout-batch-size 128 \
  --mpc-iterations 2 \
  --mpc-planning-horizon-steps 36 --mpc-control-steps 12 --mpc-knot-count 6 \
  --mpc-temperature 0.25 \
  --mpc-root-pos-sigma 0.025 --mpc-root-rot-sigma 0.045 --mpc-joint-sigma 0.09 \
  --robot-bad-floor-contact-weight 45 --robot-bad-floor-force-excess-weight 10 \
  --robot-contact-mismatch-weight 6 --robot-contact-switch-weight 12 \
  --robot-contact-force-delta-weight 2.5 \
  --robot-contact-false-positive-weight 1.5 \
  --robot-contact-false-negative-weight 0.4 \
  --robot-body-global-pos-weight 4 --robot-body-global-rot-weight 0.8 \
  --robot-ee-global-pos-weight 3 --robot-ee-global-rot-weight 0.4 \
  --robot-ee-local-pos-weight 2 --robot-ee-local-rot-weight 0.5 \
  --robot-root-pos-weight 5 --robot-root-rot-weight 1.5 \
  --robot-joint-pos-weight 0.8 --robot-action-delta-weight 0.6 \
  --robot-joint-acc-weight 0.006 --robot-joint-jerk-weight 0.0012 \
  --control-delta-weight 1.8 \
  --object-pos-weight 70 --object-rot-weight 6 \
  --object-final-pos-weight 150 --object-final-rot-weight 12 \
  --object-vel-weight 0.1
```

Outputs:

- `metrics.json`: robot metrics, object metrics, baseline object metrics, and deltas.
- `trajectory_mjwp.npz`: retarget-style `qpos/qvel/time/ctrl` export. When
  a baseline is supplied, this is resampled onto the baseline `time` grid and
  keeps the same windowed shape.
- `trajectory_mpc_rl_object.npz`: same trajectory under an explicit MPC+RL name.
- `rollout.npz`: full native 50 Hz debug rollout tensors.
- `mpc_command.npz`: optimized robot command/control trace.
- `visualization_ours_reference.mp4`: solid MPC+RL replay with cyan reference ghost.
- `visualization_baseline_reference.mp4`: solid baseline replay with cyan reference ghost when a baseline is supplied.
