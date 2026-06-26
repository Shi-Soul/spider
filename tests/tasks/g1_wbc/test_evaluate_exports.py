from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spider.tasks.g1_wbc import evaluate


def _command(num_envs: int = 1) -> SimpleNamespace:
    frames = 3
    return SimpleNamespace(
        fps=50.0,
        num_envs=num_envs,
        joint_pos=torch.arange(frames * num_envs * 29, dtype=torch.float32).reshape(
            frames, num_envs, 29
        ),
        joint_vel=torch.full((frames, num_envs, 29), 2.0),
        body_pos_w=torch.full((frames, num_envs, 30, 3), 3.0),
        body_quat_w=torch.full((frames, num_envs, 30, 4), 4.0),
        body_lin_vel_w=torch.full((frames, num_envs, 30, 3), 5.0),
        body_ang_vel_w=torch.full((frames, num_envs, 30, 3), 6.0),
        qpos_trajectory=torch.full((frames, num_envs, 36), 7.0),
        qvel_trajectory=torch.full((frames, num_envs, 35), 8.0),
    )


def test_save_mpc_result_includes_command_body_velocities(tmp_path: Path) -> None:
    """MPC debug exports include command body velocity tensors."""

    command = _command()
    result = SimpleNamespace(
        refined_qpos=torch.full((3, 36), 9.0),
        scores=torch.tensor([1.0, 2.0]),
        command=command,
    )

    path = tmp_path / "mpc_command.npz"
    evaluate._save_mpc_result(path, result)

    with np.load(path) as data:
        assert "command_body_lin_vel_w" in data.files
        assert "command_body_ang_vel_w" in data.files
        assert data["command_body_lin_vel_w"].shape == (3, 1, 30, 3)
        assert data["command_body_ang_vel_w"].shape == (3, 1, 30, 3)
        np.testing.assert_allclose(data["command_body_lin_vel_w"], 5.0)
        np.testing.assert_allclose(data["command_body_ang_vel_w"], 6.0)


def test_save_tracking_bfm_motion_writes_unbatched_motion_schema(
    tmp_path: Path,
) -> None:
    """Tracking-BFM motion exports use the unbatched standard motion schema."""

    command = _command()

    path = tmp_path / "mpc_motion.npz"
    evaluate._save_tracking_bfm_motion(path, command)

    with np.load(path) as data:
        assert set(data.files) == {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "motion_type",
        }
        assert data["fps"].shape == ()
        assert data["fps"].dtype == np.float32
        assert float(data["fps"].item()) == pytest.approx(50.0)
        assert data["joint_pos"].shape == (3, 29)
        assert data["joint_vel"].shape == (3, 29)
        assert data["body_pos_w"].shape == (3, 30, 3)
        assert data["body_quat_w"].shape == (3, 30, 4)
        assert data["body_lin_vel_w"].shape == (3, 30, 3)
        assert data["body_ang_vel_w"].shape == (3, 30, 3)
        assert str(data["motion_type"].item()) == "mujoco"
        np.testing.assert_allclose(data["joint_vel"], 2.0)
        np.testing.assert_allclose(data["body_lin_vel_w"], 5.0)
        np.testing.assert_allclose(data["body_ang_vel_w"], 6.0)


def test_save_tracking_bfm_motion_rejects_multi_env_commands(
    tmp_path: Path,
) -> None:
    """Tracking-BFM motion exports reject ambiguous multi-env command batches."""

    command = _command(num_envs=2)

    with pytest.raises(ValueError, match="single-env"):
        evaluate._save_tracking_bfm_motion(tmp_path / "mpc_motion.npz", command)
