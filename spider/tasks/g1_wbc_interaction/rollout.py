"""Policy-in-the-loop rollout for full-state G1 interaction models."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    DECIMATION,
    LEFT_FOOT_BODY_NAME,
    MUJOCO_JOINT_NAMES,
    POLICY_DT,
    RIGHT_FOOT_BODY_NAME,
)
from spider.tasks.g1_wbc.obs import G1WbcObservationBuilder, RobotState
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    _geom_is_robot_collision,
    _resolve_actuator_ids_by_joint,
    default_joint_pos_tensor,
    joint_actuator_specs,
)
from spider.tasks.g1_wbc_interaction.layout import (
    InteractionModelLayout,
    discover_interaction_layout,
    load_interaction_model,
)
from spider.tasks.g1_wbc_interaction.motion import (
    InteractionMotion,
    qvel_from_full_qpos,
)

try:
    wp.init()
except RuntimeError:
    pass


@dataclass
class InteractionRolloutConfig(WbcRolloutConfig):
    """Configuration for full-state interaction rollouts."""

    model_path: str | Path = ""


def load_interaction_model_and_layout(
    model_path: str | Path,
) -> tuple[mujoco.MjModel, InteractionModelLayout]:
    model = load_interaction_model(model_path)
    layout = discover_interaction_layout(model, model_path)
    return model, layout


class G1WbcInteractionMujocoWarpEnv:
    """Batched WBC policy simulator with full robot/object state."""

    def __init__(self, config: InteractionRolloutConfig):
        self.config = config
        self.device = str(config.device)
        self.torch_device = torch.device(config.device)
        self.num_envs = int(config.num_envs)
        self.model_cpu, self.layout = load_interaction_model_and_layout(config.model_path)
        self.data_cpu = mujoco.MjData(self.model_cpu)
        mujoco.mj_forward(self.model_cpu, self.data_cpu)

        self.body_ids = list(self.layout.robot_body_ids)
        self.root_body_id = self.layout.root_body_id
        self.foot_geom_ids = self._resolve_foot_geoms()
        self.floor_contact_geom_groups = self._resolve_floor_contact_geom_groups()
        self.floor_geom_id = self._resolve_ground_geom()
        self.imu_ang_vel_slice = self._resolve_sensor_slice("imu_ang_vel")
        if self.imu_ang_vel_slice is None:
            self.imu_ang_vel_slice = self._resolve_sensor_slice("robot/imu_ang_vel")
        self.ctrl_actuator_ids = torch.tensor(
            _resolve_actuator_ids_by_joint(self.model_cpu),
            dtype=torch.long,
            device=self.torch_device,
        )

        wp.set_device(self.device)
        with wp.ScopedDevice(self.device):
            self.model_wp = mjwarp.put_model(self.model_cpu)
            self.model_wp.opt.ls_parallel = True
            self.model_wp.opt.contact_sensor_maxmatch = 64
            self.data_wp = mjwarp.put_data(
                self.model_cpu,
                self.data_cpu,
                nworld=self.num_envs,
                nconmax=int(config.nconmax_per_env),
                njmax=int(config.njmax_per_env),
            )
            self.step_graph = None
            self.forward_graph = None
            self.reset_graph = None
            self._reset_mask_wp = wp.zeros(self.num_envs, dtype=bool)
            self._reset_mask = wp.to_torch(self._reset_mask_wp)
            if config.use_cuda_graph and wp.get_device(self.device).is_cuda:
                with wp.ScopedCapture() as capture:
                    mjwarp.step(self.model_wp, self.data_wp)
                self.step_graph = capture.graph
                with wp.ScopedCapture() as capture:
                    mjwarp.forward(self.model_wp, self.data_wp)
                self.forward_graph = capture.graph
                with wp.ScopedCapture() as capture:
                    mjwarp.reset_data(
                        self.model_wp,
                        self.data_wp,
                        reset=self._reset_mask_wp,
                    )
                self.reset_graph = capture.graph
        self.default_joint_pos = default_joint_pos_tensor(self.torch_device)
        self.action_scale = joint_actuator_specs(self.torch_device)["action_scale"]

    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> None:
        qpos = self._batch_state(qpos, self.model_cpu.nq)
        if qvel is None:
            qvel = torch.zeros(self.num_envs, self.model_cpu.nv, device=self.torch_device)
        else:
            qvel = self._batch_state(qvel, self.model_cpu.nv)
        ctrl = torch.zeros(
            self.num_envs,
            ACTION_DIM,
            dtype=torch.float32,
            device=self.torch_device,
        )
        ctrl = self._joint_order_to_model_ctrl(ctrl)
        zeros_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        with wp.ScopedDevice(self.device):
            self._reset_mask.fill_(True)
            if self.reset_graph is not None:
                wp.capture_launch(self.reset_graph)
            else:
                mjwarp.reset_data(
                    self.model_wp,
                    self.data_wp,
                    reset=self._reset_mask_wp,
                )
            wp.copy(self.data_wp.qpos, wp.from_torch(qpos.contiguous()))
            wp.copy(self.data_wp.qvel, wp.from_torch(qvel.contiguous()))
            wp.copy(self.data_wp.ctrl, wp.from_torch(ctrl.contiguous()))
            wp.copy(self.data_wp.time, wp.from_torch(zeros_time.contiguous()))
            if self.forward_graph is not None:
                wp.capture_launch(self.forward_graph)
            else:
                mjwarp.forward(self.model_wp, self.data_wp)
        if self.config.sync_after_step:
            wp.synchronize()

    def step_control(self, ctrl: torch.Tensor) -> None:
        ctrl = ctrl.to(self.torch_device, dtype=torch.float32)
        if ctrl.ndim == 1:
            ctrl = ctrl.view(1, ACTION_DIM).expand(self.num_envs, ACTION_DIM)
        if ctrl.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"Expected ctrl {(self.num_envs, ACTION_DIM)}, got {ctrl.shape}")
        model_ctrl = self._joint_order_to_model_ctrl(ctrl)
        with wp.ScopedDevice(self.device):
            wp.copy(self.data_wp.ctrl, wp.from_torch(model_ctrl.contiguous()))
            for _ in range(DECIMATION):
                if self.step_graph is not None:
                    wp.capture_launch(self.step_graph)
                else:
                    mjwarp.step(self.model_wp, self.data_wp)
            if self.config.forward_after_step:
                if self.forward_graph is not None:
                    wp.capture_launch(self.forward_graph)
                else:
                    mjwarp.forward(self.model_wp, self.data_wp)
        if self.config.sync_after_step:
            wp.synchronize()

    def robot_state(self) -> RobotState:
        full_qpos = wp.to_torch(self.data_wp.qpos).clone()
        full_qvel = wp.to_torch(self.data_wp.qvel).clone()
        robot_qpos = self.layout.robot_qpos(full_qpos)
        robot_qvel = self.layout.robot_qvel(full_qvel)
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
        base_ang_vel_b = None
        if self.imu_ang_vel_slice is not None:
            base_ang_vel_b = wp.to_torch(self.data_wp.sensordata)[
                :, self.imu_ang_vel_slice
            ].clone()
        return RobotState(
            qpos=robot_qpos,
            qvel=robot_qvel,
            body_pos_w=xpos,
            body_quat_w=xquat,
            body_lin_vel_w=lin_vel_w,
            body_ang_vel_w=ang_vel_w,
            base_ang_vel_b=base_ang_vel_b,
        )

    def full_qpos_qvel(self) -> tuple[torch.Tensor, torch.Tensor]:
        return wp.to_torch(self.data_wp.qpos).clone(), wp.to_torch(self.data_wp.qvel).clone()

    def floor_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        indicator = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.torch_device)
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
        for contact_idx, contact_geoms in enumerate(self.floor_contact_geom_groups):
            if contact_geoms.numel() == 0:
                continue
            has_floor = (geom[:, 0] == floor) | (geom[:, 1] == floor)
            has_robot_part = torch.isin(geom[:, 0], contact_geoms) | torch.isin(
                geom[:, 1], contact_geoms
            )
            mask = active_indicator & has_floor & has_robot_part
            if not torch.any(mask):
                continue
            env_ids = worldid[mask]
            indicator[:, contact_idx].scatter_reduce_(
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
            force[:, contact_idx].scatter_add_(0, force_env_ids, normal_force)
        return indicator.clamp(max=1.0), force

    def action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale.view(1, -1) + self.default_joint_pos.view(1, -1)

    def _batch_state(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        value = value.to(self.torch_device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.view(1, dim).expand(self.num_envs, dim)
        if value.shape != (self.num_envs, dim):
            raise ValueError(f"Expected state {(self.num_envs, dim)}, got {value.shape}")
        return value.contiguous()

    def _joint_order_to_model_ctrl(self, ctrl: torch.Tensor) -> torch.Tensor:
        if self.ctrl_actuator_ids.numel() == self.model_cpu.nu and torch.equal(
            self.ctrl_actuator_ids,
            torch.arange(self.model_cpu.nu, dtype=torch.long, device=self.torch_device),
        ):
            return ctrl.contiguous()
        model_ctrl = torch.zeros(
            self.num_envs,
            self.model_cpu.nu,
            dtype=ctrl.dtype,
            device=self.torch_device,
        )
        model_ctrl[:, self.ctrl_actuator_ids] = ctrl
        return model_ctrl.contiguous()

    def _resolve_foot_geoms(self) -> tuple[torch.Tensor, torch.Tensor]:
        foot_ids: list[torch.Tensor] = []
        for body_name in (LEFT_FOOT_BODY_NAME, RIGHT_FOOT_BODY_NAME):
            body_id = _resolve_name_id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"G1 model is missing foot body {body_name}")
            geom_ids = [
                geom_id
                for geom_id in range(self.model_cpu.ngeom)
                if int(self.model_cpu.geom_bodyid[geom_id]) == body_id
            ]
            foot_ids.append(torch.tensor(geom_ids, dtype=torch.long, device=self.torch_device))
        return foot_ids[0], foot_ids[1]

    def _resolve_floor_contact_geom_groups(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_foot, right_foot = self._resolve_foot_geoms()
        foot_set = set(left_foot.detach().cpu().tolist()) | set(
            right_foot.detach().cpu().tolist()
        )
        robot_body_set = set(int(body_id) for body_id in self.layout.robot_body_ids)
        other_ids = [
            geom_id
            for geom_id in range(self.model_cpu.ngeom)
            if geom_id not in foot_set
            and int(self.model_cpu.geom_bodyid[geom_id]) in robot_body_set
            and (
                _geom_is_robot_collision(self.model_cpu, geom_id)
                or int(self.model_cpu.geom_contype[geom_id]) != 0
                or int(self.model_cpu.geom_conaffinity[geom_id]) != 0
            )
        ]
        other = torch.tensor(other_ids, dtype=torch.long, device=self.torch_device)
        return left_foot, right_foot, other

    def _resolve_ground_geom(self) -> int:
        for geom_name in ("terrain", "floor"):
            geom_id = _resolve_name_id(self.model_cpu, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                return geom_id
        return -1

    def _resolve_sensor_slice(self, sensor_name: str) -> slice | None:
        sensor_id = mujoco.mj_name2id(
            self.model_cpu,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )
        if sensor_id < 0:
            return None
        start = int(self.model_cpu.sensor_adr[sensor_id])
        dim = int(self.model_cpu.sensor_dim[sensor_id])
        return slice(start, start + dim)


def run_interaction_command_rollout(
    command: G1Motion | G1CommandBatch,
    actor: WbcActor,
    config: InteractionRolloutConfig,
    *,
    initial_qpos: torch.Tensor,
    initial_qvel: torch.Tensor,
    initial_last_action: torch.Tensor | None = None,
    initial_history_state: dict | None = None,
    ref_start: int = 0,
) -> RolloutResult:
    device = torch.device(config.device)
    command = command.to(device)
    actor = actor.to(device).eval()
    total_steps = command.num_frames
    if config.max_steps is not None:
        total_steps = min(total_steps, int(config.max_steps))
    if total_steps < 1:
        raise ValueError("Need at least one rollout step.")

    env = G1WbcInteractionMujocoWarpEnv(config)
    env.reset(initial_qpos, initial_qvel)
    obs_builder = G1WbcObservationBuilder(
        motion=command,
        num_envs=config.num_envs,
        default_joint_pos=env.default_joint_pos,
        device=device,
    )
    obs_builder.load_history_state_dict(initial_history_state)
    if initial_last_action is None:
        last_action = torch.zeros(config.num_envs, ACTION_DIM, device=device)
    else:
        last_action = initial_last_action.to(device, dtype=torch.float32)
        if last_action.ndim == 1:
            last_action = last_action.view(1, ACTION_DIM).expand(config.num_envs, ACTION_DIM)
        if last_action.shape != (config.num_envs, ACTION_DIM):
            raise ValueError(
                f"Expected initial_last_action {(config.num_envs, ACTION_DIM)}, "
                f"got {last_action.shape}."
            )
        last_action = last_action.contiguous()

    traces = _TraceBuffers()
    ref_indices = []
    state = env.robot_state()
    full_qpos, full_qvel = env.full_qpos_qvel()
    floor_contact, floor_force = env.floor_contact()
    traces.append(state, full_qpos, full_qvel, floor_contact, floor_force)
    ref_indices.append(
        torch.full((config.num_envs,), int(ref_start), dtype=torch.long, device=device)
    )

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
            full_qpos, full_qvel = env.full_qpos_qvel()
            floor_contact, floor_force = env.floor_contact()
            traces.append(state, full_qpos, full_qvel, floor_contact, floor_force)
            traces.actions.append(action.detach().clone())
            traces.controls.append(ctrl.detach().clone())
            ref_indices.append(ref_idx + int(ref_start))
            last_action = action

    return traces.result(
        ref_indices=torch.stack(ref_indices, dim=0),
        final_last_action=last_action.detach().clone(),
        final_history_state=obs_builder.history_state_dict(),
    )


def command_from_full_qpos_trajectory(
    template_motion: InteractionMotion,
    full_qpos_trajectory: torch.Tensor,
    config: InteractionRolloutConfig,
    *,
    full_qvel_trajectory: torch.Tensor | None = None,
    preserve_template_first: bool = False,
) -> G1CommandBatch:
    device = torch.device(config.device)
    template_motion = template_motion.to(device)
    if template_motion.layout is None:
        raise ValueError("InteractionMotion is missing layout.")
    layout = template_motion.layout
    full_qpos_trajectory = full_qpos_trajectory.to(device, dtype=torch.float32)
    if full_qpos_trajectory.ndim == 2:
        full_qpos_trajectory = full_qpos_trajectory[:, None, :]
    if full_qpos_trajectory.ndim != 3 or full_qpos_trajectory.shape[-1] != layout.nq:
        raise ValueError(
            f"Expected full qpos shape (T, N, {layout.nq}) or (T, {layout.nq}), "
            f"got {full_qpos_trajectory.shape}."
        )
    full_qpos_trajectory = full_qpos_trajectory.clone()
    num_envs = int(full_qpos_trajectory.shape[1])
    if full_qvel_trajectory is None:
        full_qvel_trajectory = qvel_from_full_qpos(full_qpos_trajectory, layout, dt=POLICY_DT)
    else:
        full_qvel_trajectory = full_qvel_trajectory.to(device, dtype=torch.float32)
        if full_qvel_trajectory.ndim == 2:
            full_qvel_trajectory = full_qvel_trajectory[:, None, :]
        if full_qvel_trajectory.shape != full_qpos_trajectory.shape[:-1] + (layout.nv,):
            raise ValueError(
                "Expected full qvel shape "
                f"{full_qpos_trajectory.shape[:-1] + (layout.nv,)}, "
                f"got {full_qvel_trajectory.shape}."
            )

    kin_config = replace(
        config,
        num_envs=num_envs,
        max_steps=None,
        use_cuda_graph=False,
    )
    env = G1WbcInteractionMujocoWarpEnv(kin_config)
    body_pos = []
    body_quat = []
    body_lin_vel = []
    body_ang_vel = []
    with torch.inference_mode():
        for frame_idx in range(full_qpos_trajectory.shape[0]):
            env.reset(full_qpos_trajectory[frame_idx], full_qvel_trajectory[frame_idx])
            state = env.robot_state()
            body_pos.append(state.body_pos_w.detach().clone())
            body_quat.append(state.body_quat_w.detach().clone())
            body_lin_vel.append(state.body_lin_vel_w.detach().clone())
            body_ang_vel.append(state.body_ang_vel_w.detach().clone())

    robot_qpos = layout.robot_qpos(full_qpos_trajectory)
    robot_qvel = layout.robot_qvel(full_qvel_trajectory)
    joint_pos = robot_qpos[..., 7:].contiguous()
    joint_vel = robot_qvel[..., 6:].contiguous()
    body_pos_w = torch.stack(body_pos, dim=0)
    body_quat_w = torch.stack(body_quat, dim=0)
    body_lin_vel_w = torch.stack(body_lin_vel, dim=0)
    body_ang_vel_w = torch.stack(body_ang_vel, dim=0)
    if preserve_template_first:
        frame_count = full_qpos_trajectory.shape[0]
        joint_pos[:, 0] = template_motion.joint_pos[:frame_count]
        joint_vel[:, 0] = template_motion.joint_vel[:frame_count]
        body_pos_w[:, 0] = template_motion.body_pos_w[:frame_count]
        body_quat_w[:, 0] = template_motion.body_quat_w[:frame_count]
        body_lin_vel_w[:, 0] = template_motion.body_lin_vel_w[:frame_count]
        body_ang_vel_w[:, 0] = template_motion.body_ang_vel_w[:frame_count]
        full_qpos_trajectory[:, 0] = template_motion.full_state_qpos()[:frame_count]
        full_qvel_trajectory[:, 0] = template_motion.full_state_qvel()[:frame_count]

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
        qpos_trajectory=robot_qpos.contiguous(),
        qvel_trajectory=robot_qvel.contiguous(),
    )


class _TraceBuffers:
    def __init__(self) -> None:
        self.qpos: list[torch.Tensor] = []
        self.qvel: list[torch.Tensor] = []
        self.body_pos_w: list[torch.Tensor] = []
        self.body_quat_w: list[torch.Tensor] = []
        self.body_lin_vel_w: list[torch.Tensor] = []
        self.body_ang_vel_w: list[torch.Tensor] = []
        self.contact_indicator: list[torch.Tensor] = []
        self.contact_force: list[torch.Tensor] = []
        self.floor_contact_indicator: list[torch.Tensor] = []
        self.floor_contact_force: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.controls: list[torch.Tensor] = []

    def append(
        self,
        state: RobotState,
        full_qpos: torch.Tensor,
        full_qvel: torch.Tensor,
        floor_contact: torch.Tensor,
        floor_force: torch.Tensor,
    ) -> None:
        self.qpos.append(full_qpos.detach().clone())
        self.qvel.append(full_qvel.detach().clone())
        self.body_pos_w.append(state.body_pos_w.detach().clone())
        self.body_quat_w.append(state.body_quat_w.detach().clone())
        self.body_lin_vel_w.append(state.body_lin_vel_w.detach().clone())
        self.body_ang_vel_w.append(state.body_ang_vel_w.detach().clone())
        self.contact_indicator.append(floor_contact[:, :2].detach().clone())
        self.contact_force.append(floor_force[:, :2].detach().clone())
        self.floor_contact_indicator.append(floor_contact.detach().clone())
        self.floor_contact_force.append(floor_force.detach().clone())

    def result(
        self,
        *,
        ref_indices: torch.Tensor,
        final_last_action: torch.Tensor | None = None,
        final_history_state: dict | None = None,
    ) -> RolloutResult:
        return RolloutResult(
            qpos=torch.stack(self.qpos, dim=0),
            qvel=torch.stack(self.qvel, dim=0),
            body_pos_w=torch.stack(self.body_pos_w, dim=0),
            body_quat_w=torch.stack(self.body_quat_w, dim=0),
            body_lin_vel_w=torch.stack(self.body_lin_vel_w, dim=0),
            body_ang_vel_w=torch.stack(self.body_ang_vel_w, dim=0),
            actions=torch.stack(self.actions, dim=0),
            controls=torch.stack(self.controls, dim=0),
            contact_indicator=torch.stack(self.contact_indicator, dim=0),
            contact_force=torch.stack(self.contact_force, dim=0),
            floor_contact_indicator=torch.stack(self.floor_contact_indicator, dim=0),
            floor_contact_force=torch.stack(self.floor_contact_force, dim=0),
            ref_indices=ref_indices,
            final_last_action=final_last_action,
            final_history_state=final_history_state,
        )


def _resolve_name_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id >= 0:
        return int(obj_id)
    prefixed = f"robot/{name}"
    obj_id = mujoco.mj_name2id(model, objtype, prefixed)
    return int(obj_id)


__all__ = [
    "G1WbcInteractionMujocoWarpEnv",
    "InteractionRolloutConfig",
    "command_from_full_qpos_trajectory",
    "load_interaction_model_and_layout",
    "run_interaction_command_rollout",
]
