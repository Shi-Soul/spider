"""Model layout discovery for G1 WBC interaction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from spider.tasks.g1_wbc.constants import MUJOCO_BODY_NAMES, MUJOCO_JOINT_NAMES
from spider.tasks.g1_wbc.rollout import configure_wbc_model


@dataclass(frozen=True)
class ObjectJointSpec:
    """Joint metadata for object-owned DOFs."""

    joint_id: int
    joint_type: int
    qpos_adr: int
    qvel_adr: int


@dataclass(frozen=True)
class SceneObjectSpec:
    """A non-robot movable subtree discovered in the loaded MuJoCo model."""

    name: str
    root_body_id: int
    body_ids: tuple[int, ...]
    joints: tuple[ObjectJointSpec, ...]
    freejoint_id: int | None
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    pose_qpos_adr: int | None = None
    pose_qvel_adr: int | None = None

    @property
    def joint_ids(self) -> tuple[int, ...]:
        return tuple(joint.joint_id for joint in self.joints)

    @property
    def has_freejoint_pose(self) -> bool:
        return self.pose_qpos_adr is not None and self.pose_qvel_adr is not None

    @property
    def qpos_slice(self) -> slice:
        if not self.has_freejoint_pose:
            raise ValueError(f"Object {self.name!r} does not have a freejoint pose slice.")
        assert self.pose_qpos_adr is not None
        return slice(self.pose_qpos_adr, self.pose_qpos_adr + 7)

    @property
    def qvel_slice(self) -> slice:
        if not self.has_freejoint_pose:
            raise ValueError(f"Object {self.name!r} does not have a freejoint velocity slice.")
        assert self.pose_qvel_adr is not None
        return slice(self.pose_qvel_adr, self.pose_qvel_adr + 6)


@dataclass(frozen=True)
class InteractionModelLayout:
    """Robot and object index maps for a complete interaction model."""

    model_path: Path
    nq: int
    nv: int
    robot_qpos_indices: tuple[int, ...]
    robot_qvel_indices: tuple[int, ...]
    root_qpos_adr: int
    root_qvel_adr: int
    robot_joint_qpos_indices: tuple[int, ...]
    robot_joint_qvel_indices: tuple[int, ...]
    robot_body_ids: tuple[int, ...]
    root_body_id: int
    root_joint_id: int
    objects: tuple[SceneObjectSpec, ...]

    @property
    def has_objects(self) -> bool:
        return bool(self.objects)

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects)

    def robot_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(self.robot_qpos_indices, device=qpos.device, dtype=torch.long)
        return qpos.index_select(-1, idx)

    def robot_qvel(self, qvel: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(self.robot_qvel_indices, device=qvel.device, dtype=torch.long)
        return qvel.index_select(-1, idx)

    def assign_robot_qpos(self, full_qpos: torch.Tensor, robot_qpos: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(
            self.robot_qpos_indices, device=full_qpos.device, dtype=torch.long
        )
        out = full_qpos.clone()
        out.index_copy_(-1, idx, robot_qpos)
        return out

    def assign_robot_qvel(self, full_qvel: torch.Tensor, robot_qvel: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(
            self.robot_qvel_indices, device=full_qvel.device, dtype=torch.long
        )
        out = full_qvel.clone()
        out.index_copy_(-1, idx, robot_qvel)
        return out


def load_interaction_model(model_path: str | Path) -> mujoco.MjModel:
    """Load a complete WXY G1 interaction model from one XML/spec."""

    path = Path(model_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    configure_wbc_model(model)
    return model


def discover_interaction_layout(
    model: mujoco.MjModel,
    model_path: str | Path,
) -> InteractionModelLayout:
    """Discover robot slices and all non-robot movable subtrees."""

    root_joint_id = _find_joint(model, "floating_base_joint")
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
    root_qvel_adr = int(model.jnt_dofadr[root_joint_id])
    robot_qpos_indices = list(range(root_qpos_adr, root_qpos_adr + 7))
    robot_qvel_indices = list(range(root_qvel_adr, root_qvel_adr + 6))

    joint_qpos_indices: list[int] = []
    joint_qvel_indices: list[int] = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = _find_joint(model, joint_name)
        joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        joint_qvel_indices.append(int(model.jnt_dofadr[joint_id]))
    robot_qpos_indices.extend(joint_qpos_indices)
    robot_qvel_indices.extend(joint_qvel_indices)

    robot_body_ids = tuple(_find_body(model, name) for name in MUJOCO_BODY_NAMES)
    root_body_id = robot_body_ids[0]
    robot_body_set = set(robot_body_ids)
    robot_joint_set = {root_joint_id, *(_find_joint(model, name) for name in MUJOCO_JOINT_NAMES)}
    objects = _discover_movable_subtrees(model, robot_body_set, robot_joint_set)

    return InteractionModelLayout(
        model_path=Path(model_path).expanduser().resolve(),
        nq=int(model.nq),
        nv=int(model.nv),
        robot_qpos_indices=tuple(robot_qpos_indices),
        robot_qvel_indices=tuple(robot_qvel_indices),
        root_qpos_adr=root_qpos_adr,
        root_qvel_adr=root_qvel_adr,
        robot_joint_qpos_indices=tuple(joint_qpos_indices),
        robot_joint_qvel_indices=tuple(joint_qvel_indices),
        robot_body_ids=robot_body_ids,
        root_body_id=root_body_id,
        root_joint_id=root_joint_id,
        objects=tuple(objects),
    )


def _discover_movable_subtrees(
    model: mujoco.MjModel,
    robot_body_ids: set[int],
    robot_joint_ids: set[int],
) -> list[SceneObjectSpec]:
    object_roots = _object_root_bodies(model, robot_body_ids, robot_joint_ids)
    objects: list[SceneObjectSpec] = []
    for root_body_id in object_roots:
        body_ids = tuple(_body_subtree(model, root_body_id))
        body_set = set(body_ids)
        joint_ids = tuple(
            joint_id
            for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) in body_set and joint_id not in robot_joint_ids
        )
        if not joint_ids:
            continue
        joints = tuple(
            ObjectJointSpec(
                joint_id=int(joint_id),
                joint_type=int(model.jnt_type[joint_id]),
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                qvel_adr=int(model.jnt_dofadr[joint_id]),
            )
            for joint_id in joint_ids
        )
        freejoint_ids = tuple(
            joint_id
            for joint_id in joint_ids
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        freejoint_id = freejoint_ids[0] if freejoint_ids else None
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        for joint_id in joint_ids:
            qpos_indices.extend(_joint_qpos_indices(model, joint_id))
            qvel_indices.extend(_joint_qvel_indices(model, joint_id))
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)
        pose_qpos_adr = None
        pose_qvel_adr = None
        if freejoint_id is not None:
            pose_qpos_adr = int(model.jnt_qposadr[freejoint_id])
            pose_qvel_adr = int(model.jnt_dofadr[freejoint_id])
        objects.append(
            SceneObjectSpec(
                name=name or f"object_body_{root_body_id}",
                root_body_id=int(root_body_id),
                body_ids=body_ids,
                joints=joints,
                freejoint_id=freejoint_id,
                qpos_indices=tuple(qpos_indices),
                qvel_indices=tuple(qvel_indices),
                pose_qpos_adr=pose_qpos_adr,
                pose_qvel_adr=pose_qvel_adr,
            )
        )
    return objects


def _object_root_bodies(
    model: mujoco.MjModel,
    robot_body_ids: set[int],
    robot_joint_ids: set[int],
) -> list[int]:
    joint_body_ids = {
        int(model.jnt_bodyid[joint_id])
        for joint_id in range(model.njnt)
        if joint_id not in robot_joint_ids
    }
    roots: list[int] = []
    for body_id in range(1, model.nbody):
        if body_id in robot_body_ids:
            continue
        if body_id not in joint_body_ids:
            continue
        if not _has_movable_nonrobot_ancestor(
            model,
            int(body_id),
            robot_body_ids,
            joint_body_ids,
        ):
            roots.append(int(body_id))
    return roots


def _has_movable_nonrobot_ancestor(
    model: mujoco.MjModel,
    body_id: int,
    robot_body_ids: set[int],
    joint_body_ids: set[int],
) -> bool:
    parent_id = int(model.body_parentid[body_id])
    while parent_id > 0 and parent_id not in robot_body_ids:
        if parent_id in joint_body_ids:
            return True
        parent_id = int(model.body_parentid[parent_id])
    return False


def _body_subtree(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    out: list[int] = []
    stack = [int(root_body_id)]
    while stack:
        body_id = stack.pop()
        out.append(body_id)
        children = [
            child_id
            for child_id in range(1, model.nbody)
            if int(model.body_parentid[child_id]) == body_id
        ]
        stack.extend(reversed(children))
    return out


def _joint_qpos_indices(model: mujoco.MjModel, joint_id: int) -> list[int]:
    adr = int(model.jnt_qposadr[joint_id])
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return list(range(adr, adr + 7))
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return list(range(adr, adr + 4))
    return [adr]


def _joint_qvel_indices(model: mujoco.MjModel, joint_id: int) -> list[int]:
    adr = int(model.jnt_dofadr[joint_id])
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return list(range(adr, adr + 6))
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return list(range(adr, adr + 3))
    return [adr]


def validate_full_state_shapes(
    layout: InteractionModelLayout,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
) -> None:
    if qpos.ndim != 2 or qpos.shape[-1] != layout.nq:
        raise ValueError(f"Expected qpos shape (T, {layout.nq}), got {qpos.shape}.")
    if qvel is not None and (qvel.ndim != 2 or qvel.shape[-1] != layout.nv):
        raise ValueError(f"Expected qvel shape (T, {layout.nv}), got {qvel.shape}.")


def _find_joint(model: mujoco.MjModel, name: str) -> int:
    for candidate in (name, f"robot/{name}"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if joint_id >= 0:
            return int(joint_id)
    raise ValueError(f"Interaction model is missing joint {name!r}.")


def _find_body(model: mujoco.MjModel, name: str) -> int:
    for candidate in (name, f"robot/{name}"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if body_id >= 0:
            return int(body_id)
    raise ValueError(f"Interaction model is missing body {name!r}.")


__all__ = [
    "InteractionModelLayout",
    "ObjectJointSpec",
    "SceneObjectSpec",
    "discover_interaction_layout",
    "load_interaction_model",
    "validate_full_state_shapes",
]
