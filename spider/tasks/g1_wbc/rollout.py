"""Policy-in-the-loop MuJoCo Warp rollout for the G1 WBC task."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ACTUATOR_GROUPS,
    DECIMATION,
    DEFAULT_G1_MODEL_PATH,
    KNEES_BENT_JOINT_POS,
    LEFT_FOOT_BODY_NAME,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    PHYSICS_DT,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
    RIGHT_FOOT_BODY_NAME,
)
from spider.tasks.g1_wbc.motion import (
    G1CommandBatch,
    G1Motion,
    qvel_from_qpos_trajectory,
)
from spider.tasks.g1_wbc.obs import G1WbcObservationBuilder, RobotState
from spider.tasks.g1_wbc.policy import WbcActor

try:
    wp.init()
except RuntimeError:
    pass


@dataclass
class WbcRolloutConfig:
    """Configuration for batched G1 WBC rollouts."""

    model_path: str | Path = DEFAULT_G1_MODEL_PATH
    device: str = "cuda:0"
    num_envs: int = 1
    max_steps: int | None = None
    ref_offset: int = 0
    nconmax_per_env: int = 96
    njmax_per_env: int = 320
    sync_after_step: bool = True


@dataclass
class RolloutResult:
    """State and action traces from a policy rollout."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    actions: torch.Tensor
    controls: torch.Tensor
    contact_indicator: torch.Tensor
    contact_force: torch.Tensor
    ref_indices: torch.Tensor
    dt: float = POLICY_DT

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])


def default_joint_pos_tensor(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the WXY default G1 joint pose in MuJoCo joint order."""

    values = torch.zeros(ACTION_DIM, dtype=torch.float32, device=device)
    for joint_name, value in KNEES_BENT_JOINT_POS.items():
        values[MUJOCO_JOINT_NAMES.index(joint_name)] = float(value)
    return values


def _match_actuator_group(joint_name: str) -> tuple[float, float, float, float]:
    matches: list[tuple[float, float, float, float]] = []
    for patterns, kp, kd, effort, armature in ACTUATOR_GROUPS:
        if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
            matches.append((float(kp), float(kd), float(effort), float(armature)))
    if len(matches) != 1:
        raise ValueError(f"Expected one actuator group for {joint_name}, got {len(matches)}")
    return matches[0]


def joint_actuator_specs(
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Return WXY action scale and actuator gains in MuJoCo joint order."""

    kp: list[float] = []
    kd: list[float] = []
    effort: list[float] = []
    armature: list[float] = []
    action_scale: list[float] = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_kp, joint_kd, joint_effort, joint_armature = _match_actuator_group(
            joint_name
        )
        kp.append(joint_kp)
        kd.append(joint_kd)
        effort.append(joint_effort)
        armature.append(joint_armature)
        action_scale.append(joint_effort / (4.0 * joint_kp))
    return {
        "kp": torch.tensor(kp, dtype=torch.float32, device=device),
        "kd": torch.tensor(kd, dtype=torch.float32, device=device),
        "effort": torch.tensor(effort, dtype=torch.float32, device=device),
        "armature": torch.tensor(armature, dtype=torch.float32, device=device),
        "action_scale": torch.tensor(action_scale, dtype=torch.float32, device=device),
    }


def configure_wbc_model(model: mujoco.MjModel) -> None:
    """Mutate a MuJoCo model to match the WXY G1 WBC sim/action settings."""

    model.opt.timestep = float(PHYSICS_DT)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    if hasattr(model.opt, "ccd_iterations"):
        model.opt.ccd_iterations = 50
    model.opt.tolerance = 1.0e-8
    model.opt.ls_tolerance = 1.0e-2

    for joint_name in MUJOCO_JOINT_NAMES:
        kp, kd, effort, armature = _match_actuator_group(joint_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"G1 model is missing joint/actuator {joint_name}")
        dof_id = int(model.jnt_dofadr[joint_id])
        model.dof_armature[dof_id] = armature
        model.dof_damping[dof_id] = 0.0
        model.dof_frictionloss[dof_id] = 0.0

        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_ctrllimited[actuator_id] = 0


class G1WbcMujocoWarpEnv:
    """Minimal standalone batched G1 simulator for WBC policy rollout."""

    def __init__(self, config: WbcRolloutConfig):
        self.config = config
        self.device = str(config.device)
        self.torch_device = torch.device(config.device)
        self.num_envs = int(config.num_envs)

        self.model_cpu = mujoco.MjModel.from_xml_path(str(config.model_path))
        configure_wbc_model(self.model_cpu)
        self.data_cpu = mujoco.MjData(self.model_cpu)
        mujoco.mj_forward(self.model_cpu, self.data_cpu)

        self.body_ids = [
            mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in MUJOCO_BODY_NAMES
        ]
        if any(body_id < 0 for body_id in self.body_ids):
            missing = [
                name for name, body_id in zip(MUJOCO_BODY_NAMES, self.body_ids) if body_id < 0
            ]
            raise ValueError(f"G1 model is missing bodies: {missing}")
        self.root_body_id = self.body_ids[0]
        self.foot_geom_ids = self._resolve_foot_geoms()
        self.floor_geom_id = mujoco.mj_name2id(
            self.model_cpu, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        wp.set_device(self.device)
        with wp.ScopedDevice(self.device):
            self.model_wp = mjwarp.put_model(self.model_cpu)
            self.data_wp = mjwarp.put_data(
                self.model_cpu,
                self.data_cpu,
                nworld=self.num_envs,
                nconmax=int(config.nconmax_per_env),
                njmax=int(config.njmax_per_env),
            )
        self.default_joint_pos = default_joint_pos_tensor(self.torch_device)
        self.action_scale = joint_actuator_specs(self.torch_device)["action_scale"]

    def _resolve_foot_geoms(self) -> tuple[torch.Tensor, torch.Tensor]:
        foot_ids: list[torch.Tensor] = []
        for body_name in (LEFT_FOOT_BODY_NAME, RIGHT_FOOT_BODY_NAME):
            body_id = mujoco.mj_name2id(
                self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            geom_ids = [
                geom_id
                for geom_id in range(self.model_cpu.ngeom)
                if int(self.model_cpu.geom_bodyid[geom_id]) == body_id
            ]
            foot_ids.append(
                torch.tensor(geom_ids, dtype=torch.long, device=self.torch_device)
            )
        return foot_ids[0], foot_ids[1]

    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> None:
        """Set all worlds to the provided state and recompute derived quantities."""

        qpos = self._batch_state(qpos, QPOS_DIM)
        if qvel is None:
            qvel = torch.zeros(self.num_envs, QVEL_DIM, device=self.torch_device)
        else:
            qvel = self._batch_state(qvel, QVEL_DIM)
        ctrl = self.default_joint_pos.view(1, -1).expand(self.num_envs, -1)
        zeros_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        zeros_qacc = torch.zeros(self.num_envs, QVEL_DIM, device=self.torch_device)
        with wp.ScopedDevice(self.device):
            wp.copy(self.data_wp.qpos, wp.from_torch(qpos.contiguous()))
            wp.copy(self.data_wp.qvel, wp.from_torch(qvel.contiguous()))
            wp.copy(self.data_wp.ctrl, wp.from_torch(ctrl.contiguous()))
            wp.copy(self.data_wp.time, wp.from_torch(zeros_time.contiguous()))
            wp.copy(self.data_wp.qacc, wp.from_torch(zeros_qacc.contiguous()))
            mjwarp.forward(self.model_wp, self.data_wp)
        if self.config.sync_after_step:
            wp.synchronize()

    def _batch_state(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        value = value.to(self.torch_device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.view(1, dim).expand(self.num_envs, dim)
        if value.shape != (self.num_envs, dim):
            raise ValueError(f"Expected state {(self.num_envs, dim)}, got {value.shape}")
        return value.contiguous()

    def step_control(self, ctrl: torch.Tensor) -> None:
        """Advance physics for one policy step using joint position targets."""

        ctrl = ctrl.to(self.torch_device, dtype=torch.float32)
        if ctrl.ndim == 1:
            ctrl = ctrl.view(1, ACTION_DIM).expand(self.num_envs, ACTION_DIM)
        if ctrl.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"Expected ctrl {(self.num_envs, ACTION_DIM)}, got {ctrl.shape}")
        with wp.ScopedDevice(self.device):
            wp.copy(self.data_wp.ctrl, wp.from_torch(ctrl.contiguous()))
            for _ in range(DECIMATION):
                mjwarp.step(self.model_wp, self.data_wp)
            mjwarp.forward(self.model_wp, self.data_wp)
        if self.config.sync_after_step:
            wp.synchronize()

    def robot_state(self) -> RobotState:
        """Return the current robot state in tracking_bfm-compatible tensors."""

        qpos = wp.to_torch(self.data_wp.qpos).clone()
        qvel = wp.to_torch(self.data_wp.qvel).clone()
        xpos = wp.to_torch(self.data_wp.xpos)[:, self.body_ids].clone()
        xquat = wp.to_torch(self.data_wp.xquat)[:, self.body_ids].clone()
        cvel = wp.to_torch(self.data_wp.cvel)[:, self.body_ids].clone()
        root_subtree_com = wp.to_torch(self.data_wp.subtree_com)[
            :, self.root_body_id
        ].clone()
        lin_vel_c = cvel[..., 3:6]
        ang_vel_w = cvel[..., 0:3]
        lin_vel_w = lin_vel_c - torch.cross(
            ang_vel_w, root_subtree_com[:, None, :] - xpos, dim=-1
        )
        return RobotState(
            qpos=qpos,
            qvel=qvel,
            body_pos_w=xpos,
            body_quat_w=xquat,
            body_lin_vel_w=lin_vel_w,
            body_ang_vel_w=ang_vel_w,
        )

    def foot_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-world foot contact indicators and approximate normal forces."""

        indicator = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.torch_device)
        force = torch.zeros_like(indicator)
        if self.floor_geom_id < 0:
            return indicator, force

        contact = self.data_wp.contact
        geom = wp.to_torch(contact.geom).to(self.torch_device)
        worldid = wp.to_torch(contact.worldid).to(self.torch_device).long()
        dist = wp.to_torch(contact.dist).to(self.torch_device)
        includemargin = wp.to_torch(contact.includemargin).to(self.torch_device)
        address = wp.to_torch(contact.efc_address).to(self.torch_device).long()[:, 0]
        efc_force = wp.to_torch(self.data_wp.efc.force).to(self.torch_device)

        active_indicator = (
            (worldid >= 0)
            & (worldid < self.num_envs)
            & (geom[:, 0] >= 0)
            & (geom[:, 1] >= 0)
            & (dist <= includemargin + 1.0e-5)
        )
        if not torch.any(active_indicator):
            return indicator, force

        floor = torch.tensor(self.floor_geom_id, device=self.torch_device)
        for foot_idx, foot_geoms in enumerate(self.foot_geom_ids):
            has_floor = (geom[:, 0] == floor) | (geom[:, 1] == floor)
            has_foot = torch.isin(geom[:, 0], foot_geoms) | torch.isin(
                geom[:, 1], foot_geoms
            )
            mask = active_indicator & has_floor & has_foot
            if not torch.any(mask):
                continue
            env_ids = worldid[mask]
            indicator[:, foot_idx].scatter_reduce_(
                0,
                env_ids,
                torch.ones(env_ids.shape[0], dtype=torch.float32, device=self.torch_device),
                reduce="amax",
                include_self=True,
            )
            force_mask = mask & (address >= 0)
            if not torch.any(force_mask):
                continue
            force_env_ids = worldid[force_mask]
            addr = address[force_mask].clamp(min=0, max=efc_force.shape[1] - 1)
            normal_force = efc_force[force_env_ids, addr].clamp(min=0.0)
            force[:, foot_idx].scatter_add_(0, force_env_ids, normal_force)
        return indicator.clamp(max=1.0), force

    def action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        """Map raw actor action to joint position targets."""

        return action * self.action_scale.view(1, -1) + self.default_joint_pos.view(1, -1)


def run_no_mpc_rollout(
    motion: G1Motion,
    actor: WbcActor,
    config: WbcRolloutConfig,
) -> RolloutResult:
    """Roll out the WBC actor with the reference motion used directly as command."""

    device = torch.device(config.device)
    motion = motion.to(device)
    return run_command_rollout(
        motion,
        actor,
        config,
        initial_qpos=motion.qpos()[0],
        initial_qvel=motion.qvel()[0],
    )


def run_command_rollout(
    command: G1Motion | G1CommandBatch,
    actor: WbcActor,
    config: WbcRolloutConfig,
    *,
    initial_qpos: torch.Tensor,
    initial_qvel: torch.Tensor,
) -> RolloutResult:
    """Roll out the WBC actor with a motion or batched refined command source."""

    device = torch.device(config.device)
    command = command.to(device)
    actor = actor.to(device)
    actor.eval()

    if isinstance(command, G1CommandBatch) and command.num_envs != config.num_envs:
        raise ValueError(
            f"Command batch has {command.num_envs} envs, config has {config.num_envs}."
        )

    total_steps = command.num_frames
    if config.max_steps is not None:
        total_steps = min(total_steps, int(config.max_steps))
    if total_steps < 1:
        raise ValueError("Need at least one rollout step.")

    env = G1WbcMujocoWarpEnv(config)
    env.reset(initial_qpos, initial_qvel)

    obs_builder = G1WbcObservationBuilder(
        motion=command,
        num_envs=config.num_envs,
        default_joint_pos=env.default_joint_pos,
        device=device,
    )
    last_action = torch.zeros(config.num_envs, ACTION_DIM, device=device)

    qpos_trace = []
    qvel_trace = []
    body_pos_trace = []
    body_quat_trace = []
    body_lin_vel_trace = []
    body_ang_vel_trace = []
    actions = []
    controls = []
    contact_indicator = []
    contact_force = []
    ref_indices = []

    state = env.robot_state()
    foot_contact, foot_force = env.foot_contact()
    _append_state(
        state,
        foot_contact,
        foot_force,
        qpos_trace,
        qvel_trace,
        body_pos_trace,
        body_quat_trace,
        body_lin_vel_trace,
        body_ang_vel_trace,
        contact_indicator,
        contact_force,
    )
    ref_indices.append(torch.zeros(config.num_envs, dtype=torch.long, device=device))

    with torch.inference_mode():
        for step_idx in range(total_steps):
            ref_idx_scalar = min(
                max(step_idx + int(config.ref_offset), 0), command.num_frames - 1
            )
            ref_idx = torch.full(
                (config.num_envs,), ref_idx_scalar, dtype=torch.long, device=device
            )
            obs = obs_builder.compute(state, ref_idx, last_action)
            action = actor(obs)
            ctrl = env.action_to_control(action)
            env.step_control(ctrl)

            state = env.robot_state()
            foot_contact, foot_force = env.foot_contact()
            _append_state(
                state,
                foot_contact,
                foot_force,
                qpos_trace,
                qvel_trace,
                body_pos_trace,
                body_quat_trace,
                body_lin_vel_trace,
                body_ang_vel_trace,
                contact_indicator,
                contact_force,
            )
            actions.append(action.detach().clone())
            controls.append(ctrl.detach().clone())
            ref_indices.append(ref_idx)
            last_action = action

    return RolloutResult(
        qpos=torch.stack(qpos_trace, dim=0),
        qvel=torch.stack(qvel_trace, dim=0),
        body_pos_w=torch.stack(body_pos_trace, dim=0),
        body_quat_w=torch.stack(body_quat_trace, dim=0),
        body_lin_vel_w=torch.stack(body_lin_vel_trace, dim=0),
        body_ang_vel_w=torch.stack(body_ang_vel_trace, dim=0),
        actions=torch.stack(actions, dim=0),
        controls=torch.stack(controls, dim=0),
        contact_indicator=torch.stack(contact_indicator, dim=0),
        contact_force=torch.stack(contact_force, dim=0),
        ref_indices=torch.stack(ref_indices, dim=0),
    )


def command_batch_from_qpos_trajectory(
    template_motion: G1Motion,
    qpos_trajectory: torch.Tensor,
    config: WbcRolloutConfig,
    *,
    preserve_template_first: bool = False,
) -> G1CommandBatch:
    """Convert batched refined qpos trajectories into WBC command fields."""

    device = torch.device(config.device)
    template_motion = template_motion.to(device)
    qpos_trajectory = qpos_trajectory.to(device, dtype=torch.float32)
    if qpos_trajectory.ndim == 2:
        qpos_trajectory = qpos_trajectory[:, None, :]
    if qpos_trajectory.ndim != 3 or qpos_trajectory.shape[-1] != QPOS_DIM:
        raise ValueError(
            "Expected qpos trajectory shape (T, N, 36) or (T, 36), "
            f"got {qpos_trajectory.shape}."
        )

    num_envs = int(qpos_trajectory.shape[1])
    qvel_trajectory = qvel_from_qpos_trajectory(qpos_trajectory, dt=POLICY_DT)
    kin_config = replace(config, num_envs=num_envs, max_steps=None)
    env = G1WbcMujocoWarpEnv(kin_config)

    body_pos = []
    body_quat = []
    body_lin_vel = []
    body_ang_vel = []
    with torch.inference_mode():
        for frame_idx in range(qpos_trajectory.shape[0]):
            env.reset(qpos_trajectory[frame_idx], qvel_trajectory[frame_idx])
            state = env.robot_state()
            body_pos.append(state.body_pos_w.detach().clone())
            body_quat.append(state.body_quat_w.detach().clone())
            body_lin_vel.append(state.body_lin_vel_w.detach().clone())
            body_ang_vel.append(state.body_ang_vel_w.detach().clone())

    joint_pos = qpos_trajectory[..., 7:].contiguous()
    joint_vel = qvel_trajectory[..., 6:].contiguous()
    body_pos_w = torch.stack(body_pos, dim=0)
    body_quat_w = torch.stack(body_quat, dim=0)
    body_lin_vel_w = torch.stack(body_lin_vel, dim=0)
    body_ang_vel_w = torch.stack(body_ang_vel, dim=0)

    if preserve_template_first:
        frame_count = qpos_trajectory.shape[0]
        joint_pos[:, 0] = template_motion.joint_pos[:frame_count]
        joint_vel[:, 0] = template_motion.joint_vel[:frame_count]
        body_pos_w[:, 0] = template_motion.body_pos_w[:frame_count]
        body_quat_w[:, 0] = template_motion.body_quat_w[:frame_count]
        body_lin_vel_w[:, 0] = template_motion.body_lin_vel_w[:frame_count]
        body_ang_vel_w[:, 0] = template_motion.body_ang_vel_w[:frame_count]
        qpos_trajectory[:, 0] = template_motion.qpos()[:frame_count]
        qvel_trajectory[:, 0] = template_motion.qvel()[:frame_count]

    return G1CommandBatch(
        path=template_motion.path,
        motion_type=template_motion.motion_type,
        fps=template_motion.fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        qpos_trajectory=qpos_trajectory.contiguous(),
        qvel_trajectory=qvel_trajectory.contiguous(),
    )


def _append_state(
    state: RobotState,
    foot_contact: torch.Tensor,
    foot_force: torch.Tensor,
    qpos_trace: list[torch.Tensor],
    qvel_trace: list[torch.Tensor],
    body_pos_trace: list[torch.Tensor],
    body_quat_trace: list[torch.Tensor],
    body_lin_vel_trace: list[torch.Tensor],
    body_ang_vel_trace: list[torch.Tensor],
    contact_indicator: list[torch.Tensor],
    contact_force: list[torch.Tensor],
) -> None:
    qpos_trace.append(state.qpos.detach().clone())
    qvel_trace.append(state.qvel.detach().clone())
    body_pos_trace.append(state.body_pos_w.detach().clone())
    body_quat_trace.append(state.body_quat_w.detach().clone())
    body_lin_vel_trace.append(state.body_lin_vel_w.detach().clone())
    body_ang_vel_trace.append(state.body_ang_vel_w.detach().clone())
    contact_indicator.append(foot_contact.detach().clone())
    contact_force.append(foot_force.detach().clone())
