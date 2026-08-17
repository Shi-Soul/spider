# G1 WBC Command Motion 使用说明

这份文档只说明 Spider 主分支里非 HOI 的 `g1_wbc` 用法。它适用于纯 G1 机器人 motion，不涉及 object state、interaction layout，也不使用任何 `g1_wbc_interaction` 代码。

## 适用范围

当输入是一个 G1 robot motion，并且能被 `spider.tasks.g1_wbc.motion.load_motion` 读取时，使用这条路径。

输入 `.npz` 必须包含：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

可选字段：

- `fps`
- `motion_type`

支持的 motion type：

- `isaaclab`
- `mujoco`
- `auto`

当 `--motion-type isaaclab` 或 `--motion-type auto` 解析为 IsaacLab 时，loader 会把 IsaacLab 的 joint/body 顺序转换成 Spider 的 MuJoCo G1 顺序。

## Command Motion 是什么

原始 motion 是 G1 参考轨迹。Spider 的 G1 WBC MPC 不直接优化 policy action，而是在参考 motion 附近优化一条 residual command trajectory，然后把选中的 residual 转成修改后的 G1 qpos command。

每个采样控制 residual 的布局是：

$$
u_t = [\Delta p_t^{root}, \Delta r_t^{root}, \Delta q_t^{joint}]
$$

它会被应用到参考 qpos 上：

$$
\hat{q}_t = f(q_t^{ref}, u_t)
$$

其中 root position 和 joint 坐标是加法更新，root rotation 先把 axis-angle residual 转成 quaternion 再乘到参考 root orientation 上，joint 坐标会 clamp 到 G1 model 的 joint limits 内。

保存出来的 command motion 就是 WBC policy 后续应该 tracking 的优化后 qpos/qvel 轨迹。

## 主入口

从 Spider repo 根目录运行：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
```

## 运行环境

当前历史实验不是按全新 `spider/.venv` 跑的。已经存在的 G1 WBC 实验主要有两套启动方式：

- 本机 4090 / local runner：在 Spider repo 根目录运行，`PYTHONPATH` 指向 Spider repo。早期脚本默认 `/home/bai/MPC-RL/tracking_bfm/.venv/bin/python`，`20260617` 的 local current-knotfix 脚本默认 `$(command -v python)`。
- H100 runner：在远端 repo 根目录运行，Python 默认 `/data_sjy/wjx/envs/ultractrl/bin/python`。

本机复现实验时，先使用历史 local runner 方式：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
PYTHONPATH=/home/bai/MPC-RL/spider CUDA_VISIBLE_DEVICES=0 python -m spider.tasks.g1_wbc.evaluate ...
```

如果需要复现早期脚本中写死的 local Python，则使用：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
PYTHONPATH=/home/bai/MPC-RL/spider CUDA_VISIBLE_DEVICES=0 \
  /home/bai/MPC-RL/tracking_bfm/.venv/bin/python -m spider.tasks.g1_wbc.evaluate ...
```

在 H100 上复现实验时使用：

```bash
cd /data_sjy/wjx/MPC-RL/spider
PYTHONPATH=/data_sjy/wjx/MPC-RL/spider CUDA_VISIBLE_DEVICES=<gpu> \
  /data_sjy/wjx/envs/ultractrl/bin/python -m spider.tasks.g1_wbc.evaluate ...
```

当前 repo 也保留 `uv` 环境声明，可用于重新配置依赖。Python 版本是 3.12，关键运行依赖包括：

- `torch`
- `mujoco==3.7.0`
- `mujoco-warp==3.7.0.1`
- `warp-lang==1.12.1`

第一次配置环境：

```bash
cd /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider
uv sync
```

如果明确要使用 Spider repo 自己的环境，可以直接用：

```bash
uv run python -m spider.tasks.g1_wbc.evaluate ...
```

也可以激活本地 venv 后运行：

```bash
source /home/bai/MPC-RL/legacy_hoi_workflow/repositories/spider/.venv/bin/activate
python -m spider.tasks.g1_wbc.evaluate ...
```

正式 G1 WBC MPC/evaluation 需要 CUDA GPU。不要在正式运行里使用 CPU fallback。

CLI 入口是：

```bash
python -m spider.tasks.g1_wbc.evaluate
```

常用 method：

- `no_mpc`：直接让 WBC policy tracking 输入 motion。
- `g1_wbc_joint_global`：用 MPC 修改 command，偏重 global body tracking。
- `g1_wbc_joint`：用 MPC 修改 command，偏重 local body 和 joint tracking。
- `g1_wbc_ee`：用 MPC 修改 command，偏重 hand/end-effector tracking。
- `replay_command`：把已经保存的 command trajectory 重新走一遍 WBC rollout。
- `static_qpos`：把一个 qpos trajectory 当作 kinematic/static trace 评估，不跑 policy rollout。

## 从 G1 Motion 生成修改后的 Command Motion

选择一个 MPC method，并传入 `--save-rollout`。生成的 command artifact 会写到 `--output-dir/mpc_command.npz`。

```bash
python -m spider.tasks.g1_wbc.evaluate \
  --motion /path/to/g1_motion.npz \
  --motion-type auto \
  --checkpoint bc \
  --method g1_wbc_joint_global \
  --device cuda:0 \
  --output-dir /path/to/output/g1_wbc_joint_global \
  --save-rollout \
  --mpc-knot-count 8 \
  --mpc-samples 8192 \
  --mpc-rollout-batch-size 0 \
  --mpc-iterations 2 \
  --mpc-planning-horizon-steps 80 \
  --mpc-control-steps 20 \
  --mpc-temperature 0.7 \
  --mpc-control-update-mode weighted_mean \
  --mpc-root-pos-sigma 0.08 \
  --mpc-root-rot-sigma 0.18 \
  --mpc-joint-sigma 0.28 \
  --no-mpc-torch-compile \
  --seed 0
```

正式运行必须用 GPU，不要用 CPU 跑正式 G1 WBC evaluation。

完整 motion 转换不要传 `--max-steps`。`--max-steps` 只用于 debug 定位问题。

## 输出文件

带 `--save-rollout` 时，输出目录会包含：

- `metrics.json`：evaluation metadata 和 rollout metrics。
- `rollout.npz`：执行选中 command chunks 后得到的物理 WBC rollout trace。
- `mpc_command.npz`：MPC 生成的修改后 command motion。

`mpc_command.npz` 里包含：

- `refined_qpos`：优化后的单条 G1 qpos，shape 是 `(T, 36)`。
- `candidate_scores`：最后一个 MPC window 的 final sampled candidate scores。
- `command_joint_pos`：MuJoCo G1 顺序下的 command joint positions。
- `command_joint_vel`：MuJoCo G1 顺序下的 command joint velocities。
- `command_body_pos_w`：world frame 下的 command body positions。
- `command_body_quat_w`：world frame 下的 command body orientations。
- `command_qpos_trajectory`：command qpos trajectory，shape 是 `(T, 1, 36)`。
- `command_qvel_trajectory`：command qvel trajectory，shape 是 `(T, 1, 35)`。

下游如果要直接使用 command motion，优先用 `command_qpos_trajectory` 和 `command_qvel_trajectory`。如果需要 flat qpos array，就用 `refined_qpos`。

## Replay 一个已保存的 Command

用 `replay_command` 可以把已有的 `mpc_command.npz` 重新走同一条 chunked execute path。这条路径和 MPC 正式执行 command chunk 的方式一致。

```bash
python -m spider.tasks.g1_wbc.evaluate \
  --motion /path/to/g1_motion.npz \
  --motion-type auto \
  --checkpoint bc \
  --method replay_command \
  --saved-command /path/to/output/g1_wbc_joint_global/mpc_command.npz \
  --replay-task-mode g1_wbc_joint_global \
  --device cuda:0 \
  --output-dir /path/to/output/replay_g1_wbc_joint_global \
  --save-rollout
```

默认 replay 会根据保存的 qpos command 重新计算 chunk-local qvel。只有明确想使用 `mpc_command.npz` 里的 `command_qvel_trajectory` 时，才传 `--replay-use-saved-qvel`。

## 和直接 Policy Tracking 对比

如果要看 command refinement 前后的差异，可以对同一个 motion 跑 `no_mpc`：

```bash
python -m spider.tasks.g1_wbc.evaluate \
  --motion /path/to/g1_motion.npz \
  --motion-type auto \
  --checkpoint bc \
  --method no_mpc \
  --device cuda:0 \
  --output-dir /path/to/output/no_mpc \
  --save-rollout
```

重点比较：

- `no_mpc/metrics.json`
- `g1_wbc_joint_global/metrics.json`
- `g1_wbc_joint_global/mpc_command.npz`
- `g1_wbc_joint_global/rollout.npz`

## 选择哪个 MPC Objective

如果希望 global trajectory 尽量贴近输入 motion，优先用 `g1_wbc_joint_global`。

如果更关注 local body consistency 和 joint tracking，用 `g1_wbc_joint`。

如果 hand/end-effector tracking 是主要目标，用 `g1_wbc_ee`。

这三个 method 使用同一套 Spider optimizer flow，只是 reward weights 不同。

自定义 reward weights 可以通过下面的参数传入：

```bash
--mpc-reward-weights /path/to/reward_weights.json
```

这个 JSON 可以是 flat term-to-weight mapping，也可以是按 method name 分组的 mapping。

## 实现位置

这条 command motion 路径由下面几个文件组成：

- `spider/tasks/g1_wbc/evaluate.py`：CLI orchestration 和 artifact 保存。
- `spider/tasks/g1_wbc/motion.py`：输入 motion 读取、顺序转换、resampling、contact estimation。
- `spider/tasks/g1_wbc/spider_task.py`：task adapter、residual-to-qpos conversion、MPC result construction、replay command path。
- `spider/simulators/g1_wbc.py`：Spider sampled receding-horizon optimizer 使用的 WBC backend。
- `spider/tasks/g1_wbc/rollout.py`：MuJoCo/Warp rollout 和 command batch construction。
- `spider/optimizers/receding.py`：generic receding-horizon loop。
- `spider/optimizers/sampling.py`：generic sampled-MPC optimizer。

最关键的转换函数：

- `G1WbcSamplingTask.controls_to_qpos`
- `G1WbcSamplingTask.build_result`
- `command_batch_from_qpos_trajectory`

## 常见问题

如果 `load_motion` 拒绝输入，先检查必需字段是否齐全，再确认 `--motion-type` 是否和源数据顺序一致。

如果 command shape 不对，检查：

- qpos: `(T, 36)` 或 `(T, 1, 36)`
- qvel: `(T, 35)` 或 `(T, 1, 35)`

如果请求 CUDA 但机器没有 CUDA，CLI 会提前停止。正式 evaluation 应该移到 GPU 机器上跑，不要 fallback 到 CPU。

如果 replay 质量和原始 MPC run 不一致，先确认 `mpc_command.npz` 旁边的 `metrics.json` 记录了预期的 `control_steps`，并且 replay 使用同一个 checkpoint、model path 和 device family。
