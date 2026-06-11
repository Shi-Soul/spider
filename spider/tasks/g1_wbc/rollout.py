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
    WXY_G1_MODEL_PATH,
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
    nconmax_per_env: int = 512
    njmax_per_env: int = 2048
    sync_after_step: bool = True
    forward_after_step: bool = True
    use_cuda_graph: bool = True


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


def _resolve_name_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id >= 0:
        return int(obj_id)
    prefixed = f"robot/{name}"
    obj_id = mujoco.mj_name2id(model, objtype, prefixed)
    return int(obj_id)


def _wxy_actuator_joint_names() -> tuple[str, ...]:
    names: list[str] = []
    for patterns, *_ in ACTUATOR_GROUPS:
        for joint_name in MUJOCO_JOINT_NAMES:
            if joint_name in names:
                continue
            if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
                names.append(joint_name)
    if set(names) != set(MUJOCO_JOINT_NAMES):
        missing = sorted(set(MUJOCO_JOINT_NAMES) - set(names))
        extra = sorted(set(names) - set(MUJOCO_JOINT_NAMES))
        raise ValueError(f"Invalid WXY actuator groups; missing={missing}, extra={extra}")
    return tuple(names)


def _configure_wxy_collision_spec(spec: mujoco.MjSpec) -> None:
    foot_pattern = re.compile(r"^(?:robot/)?(left|right)_foot[1-7]_collision$")
    for geom in spec.geoms:
        name = geom.name or ""
        if name in ("terrain", "floor"):
            geom.contype = 1
            geom.conaffinity = 1
            geom.condim = 3
            continue
        if not re.fullmatch(r".*_collision", name):
            geom.contype = 0
            geom.conaffinity = 0
            continue
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = 1
        geom.priority = 0
        if foot_pattern.fullmatch(name):
            geom.condim = 3
            geom.priority = 1
            geom.friction[0] = 0.6


def _add_wxy_terrain_spec(spec: mujoco.MjSpec) -> None:
    if any((geom.name or "") == "terrain" for geom in spec.geoms):
        return
    terrain_body = spec.worldbody.add_body(name="terrain")
    terrain_body.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.01),
    )


def _add_wxy_self_collision_sensors(spec: mujoco.MjSpec) -> None:
    """Add tracking_bfm's self-collision sensors to keep model layout identical."""

    existing = {sensor.name for sensor in spec.sensors}
    sensor_specs = (
        ("self_collision_pelvis_found", 1 << 0),
        ("self_collision_pelvis_force", 1 << 1),
    )
    for name, data_bits in sensor_specs:
        if name in existing:
            continue
        spec.add_sensor(
            name=name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname="robot/pelvis",
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname="robot/pelvis",
            intprm=(data_bits, 0, 1),
        )


def _add_wxy_actuators_to_spec(spec: mujoco.MjSpec) -> None:
    existing = {actuator.name for actuator in spec.actuators}
    for joint_name in _wxy_actuator_joint_names():
        prefixed = f"robot/{joint_name}"
        if prefixed in existing or joint_name in existing:
            continue
        kp, kd, effort, armature = _match_actuator_group(joint_name)
        joint = spec.joint(prefixed)
        joint.armature = float(armature)
        joint.damping[0] = 0.0
        joint.frictionloss = 0.0
        actuator = spec.add_actuator(name=prefixed, target=prefixed)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange[0] = -float(effort)
        actuator.forcerange[1] = float(effort)
        joint_range = joint.range
        delta = float(effort) / float(kp)
        actuator.ctrlrange[0] = float(joint_range[0]) - delta
        actuator.ctrlrange[1] = float(joint_range[1]) + delta
        actuator.gainprm[0] = float(kp)
        actuator.biasprm[1] = -float(kp)
        actuator.biasprm[2] = -float(kd)


def _add_wxy_init_keyframe(spec: mujoco.MjSpec) -> None:
    """Add tracking_bfm's merged scene init keyframe."""

    if any((key.name or "") == "init_state" for key in spec.keys):
        return
    qpos = [0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]
    qpos.extend(float(KNEES_BENT_JOINT_POS.get(name, 0.0)) for name in MUJOCO_JOINT_NAMES)
    ctrl = []
    for actuator in spec.actuators:
        target = actuator.target or ""
        joint_name = target.removeprefix("robot/")
        ctrl.append(float(KNEES_BENT_JOINT_POS.get(joint_name, 0.0)))
    spec.add_key(name="init_state", qpos=qpos, ctrl=ctrl)


def _build_wxy_model() -> mujoco.MjModel:
    """Build a G1 WBC model matching tracking_bfm body/joint/actuator layout."""

    wxy_path = WXY_G1_MODEL_PATH.expanduser().resolve()
    mesh_dir = str(wxy_path.parent / "meshes")

    xml_text = wxy_path.read_text()
    xml_text = xml_text.replace('meshdir="meshes"', f'meshdir="{mesh_dir}"')
    robot_spec = mujoco.MjSpec.from_string(xml_text)

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.compiler.meshdir = mesh_dir

    # terrain first, then robot (matches tracking_bfm body ordering)
    _add_wxy_terrain_spec(spec)
    spec.worldbody.add_site(
        name="env_origin_0",
        pos=(0.0, 0.0, 0.0),
        size=(0.3, 0.3, 0.3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=(0.2, 0.6, 0.2, 0.3),
        group=4,
    )
    spec.attach(robot_spec, prefix="robot/",
                frame=spec.worldbody.add_frame(name="robot_frame"))

    _configure_wxy_collision_spec(spec)
    _add_wxy_actuators_to_spec(spec)
    _add_wxy_init_keyframe(spec)
    _add_wxy_self_collision_sensors(spec)
    model = spec.compile()
    configure_wbc_model(model)
    return model


def load_wbc_model(model_path: str | Path) -> mujoco.MjModel:
    """Load a G1 WBC model with tracking_bfm-compatible physics semantics."""

    path = Path(model_path).expanduser()
    if path.name == "tbfm_model.pkl":
        import pickle as _pickle
        with open(path, "rb") as _f:
            return _pickle.load(_f)
    if path.name == WXY_G1_MODEL_PATH.name:
        return _build_wxy_model()
    model = mujoco.MjModel.from_xml_path(str(path))
    configure_wbc_model(model)
    return model


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



def _find_actuator_by_joint(model: mujoco.MjModel, joint_name: str) -> int:
    """Find actuator index by matching target joint name (with or without robot/ prefix)."""
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if not (0 <= joint_id < model.njnt):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if jname in (joint_name, f"robot/{joint_name}"):
            return int(act_id)
    return -1

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
    model.opt.disableflags = 0
    model.opt.enableflags = 0

    for joint_name in MUJOCO_JOINT_NAMES:
        kp, kd, effort, armature = _match_actuator_group(joint_name)
        joint_id = _resolve_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _find_actuator_by_joint(model, joint_name)
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
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)



def _resolve_actuator_ids_by_joint(model: mujoco.MjModel) -> list[int]:
    """Return actuator indices matching MUJOCO_JOINT_NAMES, by target joint."""
    joint_name_to_actuator: dict[str, int] = {}
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if 0 <= joint_id < model.njnt:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if jname:
                joint_name_to_actuator[jname] = int(act_id)
    ids: list[int] = []
    for joint_name in MUJOCO_JOINT_NAMES:
        # Try robot/ prefix first (tracking_bfm layout), then bare name.
        act_id = joint_name_to_actuator.get(f"robot/{joint_name}", -1)
        if act_id < 0:
            act_id = joint_name_to_actuator.get(joint_name, -1)
        if act_id < 0:
            raise ValueError(f"G1 model is missing actuator for joint {joint_name}")
        ids.append(act_id)
    return ids

class G1WbcMujocoWarpEnv:
    """Minimal standalone batched G1 simulator for WBC policy rollout."""

    def __init__(self, config: WbcRolloutConfig):
        self.config = config
        self.device = str(config.device)
        self.torch_device = torch.device(config.device)
        self.num_envs = int(config.num_envs)

        self.model_cpu = load_wbc_model(config.model_path)
        self.data_cpu = mujoco.MjData(self.model_cpu)
        mujoco.mj_forward(self.model_cpu, self.data_cpu)

        self.body_ids = [
            _resolve_name_id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in MUJOCO_BODY_NAMES
        ]
        if any(body_id < 0 for body_id in self.body_ids):
            missing = [
                name for name, body_id in zip(MUJOCO_BODY_NAMES, self.body_ids) if body_id < 0
            ]
            raise ValueError(f"G1 model is missing bodies: {missing}")
        self.root_body_id = self.body_ids[0]
        self.foot_geom_ids = self._resolve_foot_geoms()
        self.floor_geom_id = self._resolve_ground_geom()
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
            foot_ids.append(
                torch.tensor(geom_ids, dtype=torch.long, device=self.torch_device)
            )
        return foot_ids[0], foot_ids[1]

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

    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> None:
        """Set all worlds to the provided state and recompute derived quantities."""

        qpos = self._batch_state(qpos, QPOS_DIM)
        if qvel is None:
            qvel = torch.zeros(self.num_envs, QVEL_DIM, device=self.torch_device)
        else:
            qvel = self._batch_state(qvel, QVEL_DIM)
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
        base_ang_vel_b = None
        if self.imu_ang_vel_slice is not None:
            base_ang_vel_b = wp.to_torch(self.data_wp.sensordata)[
                :, self.imu_ang_vel_slice
            ].clone()
        return RobotState(
            qpos=qpos,
            qvel=qvel,
            body_pos_w=xpos,
            body_quat_w=xquat,
            body_lin_vel_w=lin_vel_w,
            body_ang_vel_w=ang_vel_w,
            base_ang_vel_b=base_ang_vel_b,
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
