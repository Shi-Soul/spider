# G1 WBC MPC 方法和使用方法

## MPC 优化问题

当前 MPC 是 **policy-in-loop 的采样优化**，不是优化控制力矩，也不是直接优化视频里的真实轨迹。

**优化变量**
优化的是给 WBC policy 的 reference command，也就是一段 qpos reference 的 perturbation。

每个窗口里，原始 reference qpos 是 base_qpos。MPC 采样 delta，然后得到 refined_qpos：

- root position: base root xyz 加 delta xyz
- root rotation: 用 delta axis-angle 左乘原始 root quaternion
- joints: 原始 29 dof joint 加 delta，并 clamp 到 MuJoCo joint limit

然后把 refined_qpos 转成 policy command，包括 joint_pos/joint_vel/body_pos/body_quat/body_vel，再让 WBC actor 在 MuJoCo Warp simulator 里真实 rollout。score 是 rollout 之后算的，不是只看 command 本身。

**滑动窗口**
现在是 receding horizon：

- planning horizon: 80 policy steps，也就是约 1.6s
- control steps: 20 policy steps，也就是约 0.4s
- 800-step 评估会跑 40 个窗口
- 每个窗口优化未来 80 step command，但只执行前 20 step，然后拿真实 sim state、policy hidden/history、last action 进入下一个窗口

所以不是整条 sequence 一次性优化。

**采样/CEM**
每个窗口：

1. 保留 sample 0 为 zero delta，也就是 no-MPC 原 command。
2. 加一个 guided candidate：先跑 no-MPC，看真实 robot 落后 reference 多少，然后用一部分误差反向修正 command。
3. 对 delta 加高斯噪声采样。
4. knot 模式下只采样少量时间 knot，然后用线性插值展开 root position 和 joint delta，用 quaternion slerp 展开 root rotation delta。
5. rollout 所有 samples。
6. 按 score 选 elite，用 softmax 权重更新 mean_delta 和 sigma。
7. 下一轮围绕新 mean/sigma 采样。

当前主实验参数：

- sampling mode: knot
- knot_count: 8
- samples: 8192
- iterations: 2
- horizon/control: 80/20
- root_pos_sigma: 0.08 m
- root_rot_sigma: 0.18 rad
- joint_sigma: 0.28 rad
- smooth_passes: 0
- command_reg_weight: 0.0
- command_smooth_weight: 0.0
- guided_root_pos_gain / guided_root_rot_gain / guided_joint_gain: 0.50 / 0.50 / 0.50，三个 method 共用同一套值
- guided_root_pos_clip / guided_root_rot_clip / guided_joint_clip: 0.05 / 0.12 / 0.35，三个 method 共用同一套值
- acceptance_gate: on

当前 reward-method tuning 的公平性约束是：`g1_wbc_joint_global`、`g1_wbc_joint`、`g1_wbc_ee` 三个 method 只允许 reward objective 不同，MPC 采样、knot/interpolation、sigma、guided candidate、clip、horizon/control、sample 数都固定一致。

**目标函数**
所有 method 都是 maximize score，也就是 minimize 一堆误差和惩罚。代码里 score 是负号，所以 score 越大越好，比如 -7 比 -9 好。

`g1_wbc_joint` 当前设计目标是偏 local tracking：

- 主要压 body local position/rotation error
- 兼顾 joint position、body global、EE local/global
- 保留 contact、control/action delta、joint acceleration/jerk penalty

`g1_wbc_ee` 当前设计目标是偏 end-effector tracking：

- 主要压 hand/EE global 和 local position/rotation error
- 兼顾 body global/local
- 保留 contact、control/action delta、joint acceleration/jerk penalty

`g1_wbc_joint_global` 当前设计目标是偏 global whole-body tracking：

- 主要压 body global position/rotation error
- 兼顾 body local 和 EE/hand global/local
- 保留 contact、control/action delta、joint acceleration/jerk penalty

所以 `joint_global` 的设计目标不是“只追 root”，而是更重视 world-frame 的 whole-body tracking；`joint` 和 `ee` 仍然使用完全相同的 MPC 搜索机制，只是 reward objective 的权重不同。

**acceptance gate 是什么**
窗口内部 sample 0 是 zero delta，但每个窗口仍可能选到比 zero delta 好的候选。

整条 rollout 完成后，还会再跑一次 baseline no-MPC，然后比较：

- final_candidate_score: MPC 整条轨迹 score
- final_baseline_score: no-MPC 整条轨迹 score

如果 acceptance gate 开启，并且 MPC 整体更差，就回退到 baseline。当前这批都没有 fallback。

**当前调参诊断**
旧版 512-sample 结果不够好，主要问题不是 MPC 机制不同，而是 sample 数偏小、reward 区分度不够强，并且个别组合触发 baseline fallback。

当前最好的一版是 v13 reward-only 对比：3 个重点 motion、3 个 MPC method 全部使用同一套非 reward MPC 参数，只有 reward weights 按 method 不同。9 个 MPC run 全部 accepted，没有 baseline fallback，stderr 为空。

v13 的 tracking 偏好已经比较清楚：`joint_global` 主要拿到更好的 body global / root tracking；`joint` 在 walk、jump 上拿到更低 body local error 和更好的平滑性，但明显牺牲 global tracking；`ee` 在 qixing 上拿到最好的 EE/hand local tracking，在 walk 上也有最好的 contact mismatch。

v13 结果目录：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_reward_method_v13_20260612
```

v13 reward weights：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/g1_wbc_reward_weights_method_specific_v13_20260612.json
```

## Repo 使用方法

### 基本路径

仓库根目录：

```bash
/home/bai/MPC-RL
```

SPIDER 子仓库：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
```

当前实验输出目录：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611
```

建议从 SPIDER 子仓库运行命令：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
```

当前这批实验使用 tracking_bfm 的 Python 环境：

```bash
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python
```

也可以先激活环境：

```bash
source /home/bai/MPC-RL/tracking_bfm/.venv/bin/activate
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
```

### 关键脚本

单条 motion 评估入口：

```bash
python -m spider.tasks.g1_wbc.evaluate
```

批量评估入口：

```bash
python scripts/batch_eval_g1_wbc.py
```

可视化入口：

```bash
python scripts/visualize_g1_wbc.py
```

和 tracking_bfm 对齐检查入口：

```bash
python -m spider.tasks.g1_wbc.compare_tracking_bfm
```

### 常用 motion 路径

walk：

```bash
/home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz/walk3_subject4/motion.npz
```

jump：

```bash
/home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz/jumps1_subject1/motion.npz
```

Qixing_Fist：

```bash
/home/bai/ARC/Dataset/TeleAI-MoCap-Hangzhou/G1-29dof-BYDnpz-50fps-segmented_2k/mocap4_interp10/homebaiARCDatasetTeleAI-MoCap-HangzhouG1-29dof-PBHCpkl-30fpsmocap4_interp10Qixing_Fist_1_seg1/motion.npz
```

### 单条 no-MPC rollout

例子：walk，BC checkpoint，跑 800 policy steps，并保存 rollout：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python -m spider.tasks.g1_wbc.evaluate \
  --motion /home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz/walk3_subject4/motion.npz \
  --motion-type isaaclab \
  --checkpoint bc \
  --method no_mpc \
  --max-steps 800 \
  --device cuda:0 \
  --output-dir /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_new/no_mpc_walk3_800 \
  --save-rollout
```

输出：

```bash
metrics.json
rollout.npz
```

### 单条 MPC rollout

例子：walk，`g1_wbc_joint_global`，保存 rollout 和 refined command：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python -m spider.tasks.g1_wbc.evaluate \
  --motion /home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz/walk3_subject4/motion.npz \
  --motion-type isaaclab \
  --checkpoint bc \
  --method g1_wbc_joint_global \
  --max-steps 800 \
  --device cuda:0 \
  --mpc-samples 2048 \
  --mpc-iterations 2 \
  --mpc-planning-horizon-steps 80 \
  --mpc-control-steps 20 \
  --output-dir /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_new/walk3_global_800_s2048_i2_h80_c20 \
  --save-rollout
```

输出：

```bash
metrics.json
rollout.npz
mpc_command.npz
```

### 常用 MPC 参数

method 可选：

```bash
no_mpc
g1_wbc_joint
g1_wbc_joint_global
g1_wbc_ee
static_qpos
```

checkpoint 可选：

```bash
bc
bcrl
```

preset 可选：

```bash
aggressive
conservative
explore
rootrot
wide
```

常用显式参数：

```bash
--mpc-samples 512
--mpc-samples 2048
--mpc-samples 4096
--mpc-iterations 2
--mpc-planning-horizon-steps 80
--mpc-control-steps 20
--mpc-root-pos-sigma 0.015
--mpc-root-rot-sigma 0.035
--mpc-joint-sigma 0.06
--mpc-smooth-passes 0
--mpc-command-reg-weight 0.0
--mpc-command-smooth-weight 0.0
--no-mpc-acceptance-gate
```

### 批量评估

例子：在指定数据集里跑前若干条 motion：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python scripts/batch_eval_g1_wbc.py \
  --datasets /home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz \
  --methods no_mpc g1_wbc_joint g1_wbc_joint_global g1_wbc_ee \
  --checkpoints bc \
  --limit 10 \
  --max-steps 800 \
  --device cuda:0 \
  --mpc-samples 512 \
  --mpc-iterations 2 \
  --mpc-planning-horizon-steps 80 \
  --mpc-control-steps 20 \
  --output /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_new/batch_lafan_bc_800.json
```

### 最终 v4 reward-method 配置

当前选定的 focus-motion 固定预算配置是 v4：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/g1_wbc_reward_weights_method_specific_v4_20260612.json
```

`spider.tasks.g1_wbc.mpc.REWARD_WEIGHT_PRESETS` 里的三种 method 默认 reward 也已经对齐到这套 v4 setting。

固定 MPC 参数：

```bash
--mpc-sampling-mode knot
--mpc-knot-count 8
--mpc-samples 512
--mpc-iterations 2
--mpc-planning-horizon-steps 80
--mpc-control-steps 20
--mpc-root-pos-sigma 0.08
--mpc-root-rot-sigma 0.18
--mpc-joint-sigma 0.28
--mpc-smooth-passes 0
--mpc-command-reg-weight 0.0
--mpc-command-smooth-weight 0.0
--mpc-reward-weights /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/g1_wbc_reward_weights_method_specific_v4_20260612.json
```

完整 focus-motion 批量运行脚本：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/run_h100_reward_jobs_v4.sh
```

本地结果和最终报告：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_reward_method_v4_20260612/reward_method_fixed_knot_summary.csv
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_reward_method_v4_20260612/reward_method_fixed_knot_summary.json
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/reward_method_v4_report_20260612.md
```

v4 结果没有 method fallback；所有 v4 run 的 `bad_floor_contact_rate` 都是 `0.0`。

最终 2x2 视频渲染脚本：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/render_reward_method_v4_videos.py
```

视频输出：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_reward_method_v4_20260612/videos/g1_wbc_walk_bc_reward_v4_4method_2x2_800.mp4
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_reward_method_v4_20260612/videos/g1_wbc_jump_bc_reward_v4_4method_2x2_800.mp4
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_reward_method_v4_20260612/videos/g1_wbc_qixing_bc_reward_v4_4method_2x2_800.mp4
```

### 2x2 可视化

可视化使用 saved rollout，不需要重新跑 policy 或 MPC。下面例子生成固定世界相机、全局光照、MuJoCo scene 内 cyan reference ghost、root error overlay、2x2 四方法对比视频。

walk：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python scripts/visualize_g1_wbc.py \
  --method saved \
  --device cpu \
  --motion /home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz/walk3_subject4/motion.npz \
  --max-steps 800 \
  --saved-rollout no_mpc:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_diag_after_fix/no_mpc_walk3_800/rollout.npz \
  --saved-rollout joint:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_methods_after_fix/walk3_joint_800_s512_i2_h80_c20/rollout.npz \
  --saved-rollout ee:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_methods_after_fix/walk3_ee_800_s512_i2_h80_c20/rollout.npz \
  --saved-rollout joint_global:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_diag_after_fix/walk3_global_800_s512_i2_h80_c20/rollout.npz \
  --camera-mode fixed \
  --camera-distance 10.0 \
  --camera-azimuth 135 \
  --camera-elevation -24 \
  --camera-lookat-x -1.43 \
  --camera-lookat-y 2.40 \
  --camera-lookat-z 1.0 \
  --show-root-error \
  --panel-layout 2x2 \
  --output /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_walk3_subject4_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
```

jump：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python scripts/visualize_g1_wbc.py \
  --method saved \
  --device cpu \
  --motion /home/bai/ARC/Dataset/LAFAN/G1-29dof-BYDnpz/jumps1_subject1/motion.npz \
  --max-steps 800 \
  --saved-rollout no_mpc:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_diag_after_fix/no_mpc_jump_800/rollout.npz \
  --saved-rollout joint:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_methods_after_fix/jump_joint_800_s512_i2_h80_c20/rollout.npz \
  --saved-rollout ee:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_methods_after_fix/jump_ee_800_s512_i2_h80_c20/rollout.npz \
  --saved-rollout joint_global:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_diag_after_fix/jump_global_800_s2048_i2_h80_c20/rollout.npz \
  --camera-mode fixed \
  --camera-distance 9.0 \
  --camera-azimuth 135 \
  --camera-elevation -24 \
  --camera-lookat-x 0.8 \
  --camera-lookat-y 0.8 \
  --camera-lookat-z 1.0 \
  --show-root-error \
  --panel-layout 2x2 \
  --output /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_jumps1_subject1_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
```

Qixing_Fist：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
/home/bai/MPC-RL/tracking_bfm/.venv/bin/python scripts/visualize_g1_wbc.py \
  --method saved \
  --device cpu \
  --motion /home/bai/ARC/Dataset/TeleAI-MoCap-Hangzhou/G1-29dof-BYDnpz-50fps-segmented_2k/mocap4_interp10/homebaiARCDatasetTeleAI-MoCap-HangzhouG1-29dof-PBHCpkl-30fpsmocap4_interp10Qixing_Fist_1_seg1/motion.npz \
  --max-steps 800 \
  --saved-rollout no_mpc:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_diag_after_fix/no_mpc_qixing_800/rollout.npz \
  --saved-rollout joint:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_methods_after_fix/qixing_joint_800_s512_i2_h80_c20/rollout.npz \
  --saved-rollout ee:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_methods_after_fix/qixing_ee_800_s512_i2_h80_c20/rollout.npz \
  --saved-rollout joint_global:/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_diag_after_fix/qixing_global_800_s512_i2_h80_c20/rollout.npz \
  --camera-mode fixed \
  --camera-distance 6.0 \
  --camera-azimuth 135 \
  --camera-elevation -24 \
  --camera-lookat-x -1.31 \
  --camera-lookat-y -0.79 \
  --camera-lookat-z 1.0 \
  --show-root-error \
  --panel-layout 2x2 \
  --output /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_qixing_fist_1_seg1_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
```

### 当前已有结果

汇总表：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_summary_20260611_focus_motions.csv
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/eval_summary_20260611_focus_motions.json
```

2x2 视频：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_walk3_subject4_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_jumps1_subject1_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_qixing_fist_1_seg1_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
```

### 视频检查

检查视频帧数、分辨率、时长：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,duration,r_frame_rate \
  -of default=noprint_wrappers=1 \
  /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_jumps1_subject1_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4
```

抽帧：

```bash
ffmpeg -y -v error -ss 12 \
  -i /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/g1_wbc_jumps1_subject1_bc_sceneghost_fixed_camera_global_light_rooterr_4method_2x2_800.mp4 \
  -frames:v 1 \
  /home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/videos/check_jump_t12.png
```

### no-MPC 与 tracking_bfm 对齐检查

已有对齐结果：

```bash
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/compare_tracking/walk3_subject4_200.json
/home/bai/MPC-RL/legacy_hoi_workflow/experiments/spider_retarget_work_20260611/compare_tracking/walk3_subject4_300.json
```

200-step 结果非常接近；300-step 已出现小漂移。排查时优先区分 policy feedback divergence 和 simulator/contact bifurcation。
