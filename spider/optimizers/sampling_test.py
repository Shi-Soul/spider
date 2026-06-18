from __future__ import annotations

import torch

from spider.config import build_sampling_mpc_config
from spider.interp import interp
from spider.optimizers.sampling import _interpolate_knot_samples, _sample_ctrls_impl


def test_build_sampling_mpc_config_uses_explicit_endpoint_knot_count() -> None:
    config = build_sampling_mpc_config(
        robot_type="g1",
        embodiment_type="humanoid",
        simulator="dummy",
        device="cpu",
        sim_dt=0.02,
        horizon_steps=160,
        ctrl_steps=40,
        knot_count=4,
        num_samples=8,
        max_num_iterations=2,
        temperature=0.7,
        nq=35,
        nv=34,
        nu=34,
    )

    assert config.horizon_steps == 160
    assert config.ctrl_steps == 40
    assert config.noise_scale.shape == (8, 4, 34)


def test_interpolate_knot_samples_expands_endpoint_knots_to_dense_horizon() -> None:
    knots = torch.tensor([[[0.0], [3.0], [6.0], [9.0]]])

    dense = _interpolate_knot_samples(knots, target_steps=10)

    assert dense.shape == (1, 10, 1)
    assert dense[0, 0, 0].item() == 0.0
    assert dense[0, -1, 0].item() == 9.0
    assert torch.allclose(dense[0, :, 0], torch.arange(10, dtype=torch.float32))


def test_sample_ctrls_legacy_path_matches_old_interp_behavior() -> None:
    config = build_sampling_mpc_config(
        robot_type="g1",
        embodiment_type="bimanual",
        simulator="dummy",
        device="cpu",
        sim_dt=0.02,
        horizon_steps=12,
        ctrl_steps=4,
        knot_count=4,
        num_samples=2,
        max_num_iterations=2,
        temperature=0.7,
        nq=35,
        nv=34,
        nu=6,
    )
    config.num_knot_points = 0
    config.knot_steps = 3
    ctrls = torch.zeros(12, 6)

    torch.manual_seed(123)
    actual = _sample_ctrls_impl(config, ctrls)
    torch.manual_seed(123)
    expected = ctrls + interp(
        torch.randn_like(config.noise_scale) * config.noise_scale,
        config.knot_steps,
    )

    assert torch.allclose(actual, expected)
