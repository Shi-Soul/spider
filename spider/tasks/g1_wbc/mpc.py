"""Batched policy-in-the-loop MPC for the G1 WBC task."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import mujoco
import torch

from spider.tasks.g1_wbc.constants import (
    MUJOCO_JOINT_NAMES,
    POLICY_DT,
    QPOS_DIM,
)
from spider.tasks.g1_wbc.math_utils import (
    axis_angle_from_quat,
    normalize,
    quat_from_axis_angle,
    quat_inv,
    quat_mul,
)
from spider.tasks.g1_wbc.metrics import compute_rollout_scores
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    command_batch_from_qpos_trajectory,
    load_wbc_model,
    run_command_rollout,
)

MpcMode = Literal["g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global"]
MpcPreset = Literal["aggressive", "conservative"]


@dataclass
class G1WbcMpcConfig:
    """Sampling MPC parameters for refined WBC command optimization."""

    mode: MpcMode = "g1_wbc_joint"
    num_samples: int = 64
    num_iterations: int = 4
    elite_frac: float = 0.125
    temperature: float = 0.7
    root_pos_sigma: float = 0.015
    root_rot_sigma: float = 0.035
    joint_sigma: float = 0.06
    min_root_pos_sigma: float = 0.002
    min_root_rot_sigma: float = 0.004
    min_joint_sigma: float = 0.008
    sigma_decay: float = 0.75
    smooth_passes: int = 2
    command_reg_weight: float = 0.02
    command_smooth_weight: float = 0.00005
    use_guided_candidate: bool = True
    guided_root_pos_gain: float = 0.5
    guided_root_rot_gain: float = 0.5
    guided_joint_gain: float = 0.4
    guided_root_pos_clip: float = 0.05
    guided_root_rot_clip: float = 0.12
    guided_joint_clip: float = 0.20
    seed: int | None = 0
    freeze_first_frame: bool = True


@dataclass
class MpcIterationInfo:
    iteration: int
    best_score: float
    mean_score: float
    elite_score: float
    zero_delta_score: float
    best_index: int


@dataclass
class G1WbcMpcResult:
    command: G1CommandBatch
    rollout: RolloutResult
    refined_qpos: torch.Tensor
    scores: torch.Tensor
    history: list[MpcIterationInfo]
    accepted: bool = True
    final_candidate_score: float = 0.0
    final_baseline_score: float = 0.0


def mpc_config_from_preset(
    mode: MpcMode,
    preset: MpcPreset = "aggressive",
) -> G1WbcMpcConfig:
    """Return tuned MPC defaults for a motion class."""

    config = G1WbcMpcConfig(mode=mode)
    if preset == "aggressive":
        return config
    if preset == "conservative":
        return replace(
            config,
            num_iterations=3,
            root_pos_sigma=0.003,
            root_rot_sigma=0.008,
            joint_sigma=0.012,
            command_reg_weight=0.20,
            command_smooth_weight=0.0010,
            guided_root_pos_gain=0.25,
            guided_root_rot_gain=0.25,
            guided_joint_gain=0.20,
        )
    raise ValueError(f"Unknown MPC preset: {preset}")


def optimize_mpc_command(
    motion: G1Motion,
    actor: WbcActor,
    rollout_config: WbcRolloutConfig,
    mpc_config: G1WbcMpcConfig,
) -> G1WbcMpcResult:
    """Optimize a refined reference command using batched policy+sim rollouts."""

    if mpc_config.num_samples < 2:
        raise ValueError("MPC requires at least two samples.")
    if mpc_config.num_iterations < 1:
        raise ValueError("MPC requires at least one iteration.")

    device = torch.device(rollout_config.device)
    motion = motion.to(device)
    actor = actor.to(device).eval()
    horizon = motion.num_frames
    if rollout_config.max_steps is not None:
        horizon = min(horizon, int(rollout_config.max_steps))
    if horizon < 1:
        raise ValueError("Need at least one MPC horizon step.")

    base_qpos = motion.qpos()[:horizon].contiguous()
    initial_qpos = motion.qpos()[0]
    initial_qvel = motion.qvel()[0]
    mean_delta = torch.zeros(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    sigma = _initial_sigma(horizon, device, mpc_config)
    min_sigma = _min_sigma(horizon, device, mpc_config)
    joint_low, joint_high = _joint_limits(rollout_config, device)
    guided_delta = _guided_delta_from_no_mpc(
        motion,
        actor,
        rollout_config,
        horizon,
        base_qpos,
        mpc_config,
    )

    if mpc_config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(mpc_config.seed))
    else:
        generator = None

    best_score = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
    best_qpos = base_qpos
    best_is_template = True
    final_scores = torch.empty(0, dtype=torch.float32, device=device)
    history: list[MpcIterationInfo] = []

    batch_config = replace(
        rollout_config,
        num_envs=mpc_config.num_samples,
        max_steps=horizon,
    )

    for iteration in range(mpc_config.num_iterations):
        candidates_delta = _sample_delta(
            mean_delta,
            sigma,
            mpc_config,
            generator,
            guided_delta=guided_delta,
        )
        candidates_qpos = _apply_delta_to_qpos(
            base_qpos,
            candidates_delta,
            joint_low=joint_low,
            joint_high=joint_high,
        )
        command = command_batch_from_qpos_trajectory(
            motion,
            candidates_qpos,
            batch_config,
            preserve_template_first=True,
        )
        rollout = run_command_rollout(
            command,
            actor,
            batch_config,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
        )
        raw_scores, terms = compute_rollout_scores(motion, rollout)
        scores = _score_from_terms(terms, mpc_config.mode)
        scores = scores - _command_regularization(
            candidates_delta,
            mpc_config.command_reg_weight,
            mpc_config.command_smooth_weight,
        )
        final_scores = scores.detach().clone()

        iteration_best = torch.argmax(scores)
        if scores[iteration_best] > best_score:
            best_score = scores[iteration_best].detach().clone()
            best_qpos = candidates_qpos[:, iteration_best].detach().clone()
            best_delta = candidates_delta[:, iteration_best]
            best_is_template = bool(best_delta.abs().max().detach().cpu().item() < 1.0e-8)

        elite_count = max(1, int(round(mpc_config.num_samples * mpc_config.elite_frac)))
        elite_scores, elite_indices = torch.topk(scores, k=elite_count, largest=True)
        elite_delta = candidates_delta[:, elite_indices]
        weights = torch.softmax(
            (elite_scores - elite_scores.mean())
            / max(float(mpc_config.temperature), 1.0e-6),
            dim=0,
        )
        mean_delta = (elite_delta * weights.view(1, -1, 1)).sum(dim=1)
        centered = elite_delta - mean_delta[:, None, :]
        elite_std = torch.sqrt(
            (centered.square() * weights.view(1, -1, 1)).sum(dim=1) + 1.0e-8
        )
        sigma = torch.maximum(
            min_sigma,
            (
                0.5 * sigma
                + 0.5 * elite_std
            )
            * float(mpc_config.sigma_decay),
        )
        if mpc_config.freeze_first_frame:
            mean_delta[0] = 0.0
            sigma[0] = 0.0

        history.append(
            MpcIterationInfo(
                iteration=iteration,
                best_score=float(scores.max().detach().cpu().item()),
                mean_score=float(scores.mean().detach().cpu().item()),
                elite_score=float(elite_scores.mean().detach().cpu().item()),
                zero_delta_score=float(scores[0].detach().cpu().item()),
                best_index=int(iteration_best.detach().cpu().item()),
            )
        )
        del raw_scores

    final_config = replace(rollout_config, num_envs=1, max_steps=horizon)
    final_command = command_batch_from_qpos_trajectory(
        motion,
        best_qpos[:, None, :],
        final_config,
        preserve_template_first=best_is_template,
    )
    final_rollout = run_command_rollout(
        final_command,
        actor,
        final_config,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
    )
    final_score = _single_rollout_score(motion, final_rollout, mpc_config.mode)
    baseline_command = command_batch_from_qpos_trajectory(
        motion,
        base_qpos[:, None, :],
        final_config,
        preserve_template_first=True,
    )
    baseline_rollout = run_command_rollout(
        baseline_command,
        actor,
        final_config,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
    )
    baseline_score = _single_rollout_score(motion, baseline_rollout, mpc_config.mode)
    accepted = final_score >= baseline_score
    if not accepted:
        final_command = baseline_command
        final_rollout = baseline_rollout
        best_qpos = base_qpos
    return G1WbcMpcResult(
        command=final_command,
        rollout=final_rollout,
        refined_qpos=best_qpos,
        scores=final_scores,
        history=history,
        accepted=accepted,
        final_candidate_score=final_score,
        final_baseline_score=baseline_score,
    )


def _initial_sigma(
    horizon: int,
    device: torch.device,
    config: G1WbcMpcConfig,
) -> torch.Tensor:
    sigma = torch.empty(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    sigma[:, :3] = float(config.root_pos_sigma)
    sigma[:, 3:6] = float(config.root_rot_sigma)
    sigma[:, 6:] = float(config.joint_sigma)
    if config.freeze_first_frame:
        sigma[0] = 0.0
    return sigma


def _min_sigma(
    horizon: int,
    device: torch.device,
    config: G1WbcMpcConfig,
) -> torch.Tensor:
    sigma = torch.empty(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    sigma[:, :3] = float(config.min_root_pos_sigma)
    sigma[:, 3:6] = float(config.min_root_rot_sigma)
    sigma[:, 6:] = float(config.min_joint_sigma)
    if config.freeze_first_frame:
        sigma[0] = 0.0
    return sigma


def _sample_delta(
    mean_delta: torch.Tensor,
    sigma: torch.Tensor,
    config: G1WbcMpcConfig,
    generator: torch.Generator | None,
    *,
    guided_delta: torch.Tensor | None,
) -> torch.Tensor:
    shape = (mean_delta.shape[0], config.num_samples, mean_delta.shape[1])
    noise = torch.randn(
        shape,
        dtype=mean_delta.dtype,
        device=mean_delta.device,
        generator=generator,
    )
    delta = mean_delta[:, None, :] + noise * sigma[:, None, :]
    delta[:, 0, :] = 0.0
    next_reserved = 1
    if (
        guided_delta is not None
        and config.num_samples > next_reserved
        and guided_delta.abs().max() > 1.0e-8
    ):
        delta[:, next_reserved, :] = guided_delta
        next_reserved += 1
    if config.num_samples > next_reserved and mean_delta.abs().max() > 1.0e-8:
        delta[:, next_reserved, :] = mean_delta
    if config.smooth_passes > 0:
        delta = _smooth_delta(delta, config.smooth_passes)
    if config.freeze_first_frame:
        delta[0] = 0.0
    return delta


def _guided_delta_from_no_mpc(
    motion: G1Motion,
    actor: WbcActor,
    rollout_config: WbcRolloutConfig,
    horizon: int,
    base_qpos: torch.Tensor,
    config: G1WbcMpcConfig,
) -> torch.Tensor | None:
    if not config.use_guided_candidate or config.num_samples < 2:
        return None
    no_mpc_config = replace(rollout_config, num_envs=1, max_steps=horizon)
    no_mpc_rollout = run_command_rollout(
        motion,
        actor,
        no_mpc_config,
        initial_qpos=motion.qpos()[0],
        initial_qvel=motion.qvel()[0],
    )
    executed = no_mpc_rollout.qpos[1 : horizon + 1, 0]
    if executed.shape[0] != horizon:
        return None
    guided = torch.zeros(horizon, QPOS_DIM - 1, dtype=torch.float32, device=base_qpos.device)
    guided[:, :3] = (base_qpos[:, :3] - executed[:, :3]) * float(
        config.guided_root_pos_gain
    )
    quat_err = quat_mul(base_qpos[:, 3:7], quat_inv(executed[:, 3:7]))
    guided[:, 3:6] = axis_angle_from_quat(quat_err) * float(config.guided_root_rot_gain)
    joint_gain = (
        0.0
        if config.mode in ("g1_wbc_ee", "g1_wbc_joint_global")
        else float(config.guided_joint_gain)
    )
    guided[:, 6:] = (base_qpos[:, 7:] - executed[:, 7:]) * float(
        joint_gain
    )
    guided[:, :3] = guided[:, :3].clamp(
        -float(config.guided_root_pos_clip), float(config.guided_root_pos_clip)
    )
    guided[:, 3:6] = guided[:, 3:6].clamp(
        -float(config.guided_root_rot_clip), float(config.guided_root_rot_clip)
    )
    guided[:, 6:] = guided[:, 6:].clamp(
        -float(config.guided_joint_clip), float(config.guided_joint_clip)
    )
    if config.smooth_passes > 0:
        guided = _smooth_delta(guided[:, None, :], config.smooth_passes)[:, 0]
    if config.freeze_first_frame:
        guided[0] = 0.0
    return guided


def _smooth_delta(delta: torch.Tensor, passes: int) -> torch.Tensor:
    out = delta
    for _ in range(int(passes)):
        if out.shape[0] <= 2:
            break
        smoothed = out.clone()
        smoothed[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = smoothed
    return out


def _apply_delta_to_qpos(
    base_qpos: torch.Tensor,
    delta: torch.Tensor,
    *,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
) -> torch.Tensor:
    qpos = base_qpos[:, None, :].expand(-1, delta.shape[1], -1).clone()
    qpos[..., :3] = qpos[..., :3] + delta[..., :3]
    delta_quat = quat_from_axis_angle(delta[..., 3:6])
    qpos[..., 3:7] = normalize(quat_mul(delta_quat, qpos[..., 3:7]))
    qpos[..., 7:] = torch.clamp(qpos[..., 7:] + delta[..., 6:], joint_low, joint_high)
    return qpos.contiguous()


def _score_from_terms(
    terms: dict[str, torch.Tensor],
    mode: MpcMode,
) -> torch.Tensor:
    if mode == "g1_wbc_ee":
        return -(
            4.0 * terms["contact_false_positive"]
            + 3.0 * terms["contact_false_negative"]
            + 3.0 * terms["contact_switch"]
            + 0.6 * terms["contact_force_excess"]
            + 0.3 * terms["contact_force_delta"]
            + 5.0 * terms["ee_global_pos_error"]
            + 2.5 * terms["ee_local_pos_error"]
            + 0.6 * terms["ee_global_rot_error"]
            + 0.3 * terms["ee_local_rot_error"]
            + 1.0 * terms["root_pos_error"]
            + 0.4 * terms["root_rot_error"]
            + 0.40 * terms["control_delta"]
            + 0.0030 * terms["joint_acc"]
        )
    if mode == "g1_wbc_joint_global":
        return -(
            4.0 * terms["contact_false_positive"]
            + 3.0 * terms["contact_false_negative"]
            + 3.0 * terms["contact_switch"]
            + 0.5 * terms["contact_force_excess"]
            + 0.25 * terms["contact_force_delta"]
            + 2.5 * terms["body_global_pos_error"]
            + 0.8 * terms["body_global_rot_error"]
            + 2.0 * terms["ee_global_pos_error"]
            + 0.8 * terms["ee_global_rot_error"]
            + 1.5 * terms["root_pos_error"]
            + 0.4 * terms["root_rot_error"]
            + 0.20 * terms["control_delta"]
            + 0.0015 * terms["joint_acc"]
        )
    return -(
        4.0 * terms["contact_false_positive"]
        + 3.0 * terms["contact_false_negative"]
        + 3.0 * terms["contact_switch"]
        + 0.5 * terms["contact_force_excess"]
        + 0.25 * terms["contact_force_delta"]
        + 2.0 * terms["body_global_pos_error"]
        + 0.6 * terms["body_global_rot_error"]
        + 1.2 * terms["joint_pos_error"]
        + 1.5 * terms["ee_global_pos_error"]
        + 1.5 * terms["root_pos_error"]
        + 0.4 * terms["root_rot_error"]
        + 0.20 * terms["control_delta"]
        + 0.0015 * terms["joint_acc"]
    )


def _single_rollout_score(
    motion: G1Motion,
    rollout: RolloutResult,
    mode: MpcMode,
) -> float:
    _, terms = compute_rollout_scores(motion, rollout)
    score = _score_from_terms(terms, mode)
    if score.numel() == 0:
        return -float("inf")
    return float(torch.nan_to_num(score.float()).max().detach().cpu().item())


def _command_regularization(
    delta: torch.Tensor,
    reg_weight: float,
    smooth_weight: float,
) -> torch.Tensor:
    reg = delta.square().mean(dim=(0, 2))
    if delta.shape[0] > 1:
        smooth = torch.diff(delta, dim=0).square().mean(dim=(0, 2)) / (POLICY_DT**2)
    else:
        smooth = torch.zeros(delta.shape[1], dtype=delta.dtype, device=delta.device)
    return float(reg_weight) * reg + float(smooth_weight) * smooth


def _joint_limits(
    rollout_config: WbcRolloutConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = load_wbc_model(rollout_config.model_path)
    low = []
    high = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}"
            )
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {joint_name}")
        if int(model.jnt_limited[joint_id]):
            low.append(float(model.jnt_range[joint_id, 0]))
            high.append(float(model.jnt_range[joint_id, 1]))
        else:
            low.append(-float("inf"))
            high.append(float("inf"))
    return (
        torch.tensor(low, dtype=torch.float32, device=device),
        torch.tensor(high, dtype=torch.float32, device=device),
    )
