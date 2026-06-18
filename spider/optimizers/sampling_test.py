from __future__ import annotations

import torch

from spider.config import Config, compute_noise_schedule
from spider.interp import interp
from spider.optimizers.sampling import _interpolate_knot_samples, _sample_ctrls_impl


def test_sample_ctrls_explicit_endpoint_path_matches_endpoint_interpolation() -> None:
    config = Config(
        robot_type="g1",
        embodiment_type="humanoid",
        simulator="dummy",
        device="cpu",
        num_samples=2,
    )
    config.nu = 1
    config.num_knot_points = 4
    config.noise_scale = torch.ones(2, 4, 1)
    ctrls = torch.zeros(10, 1)

    torch.manual_seed(123)
    actual = _sample_ctrls_impl(config, ctrls)
    torch.manual_seed(123)
    expected = ctrls + _interpolate_knot_samples(
        torch.randn_like(config.noise_scale) * config.noise_scale,
        ctrls.shape[0],
    )

    assert torch.allclose(actual, expected)


def test_interpolate_knot_samples_expands_endpoint_knots_to_dense_horizon() -> None:
    knots = torch.tensor([[[0.0], [3.0], [6.0], [9.0]]])

    dense = _interpolate_knot_samples(knots, target_steps=10)

    assert dense.shape == (1, 10, 1)
    assert dense[0, 0, 0].item() == 0.0
    assert dense[0, -1, 0].item() == 9.0
    assert torch.allclose(dense[0, :, 0], torch.arange(10, dtype=torch.float32))


def test_sample_ctrls_legacy_path_matches_old_interp_behavior() -> None:
    config = Config(
        robot_type="g1",
        embodiment_type="bimanual",
        simulator="dummy",
        device="cpu",
        sim_dt=0.02,
        ref_dt=0.02,
        render_dt=0.02,
        horizon=0.24,
        ctrl_dt=0.08,
        knot_dt=0.06,
        num_samples=2,
        max_num_iterations=2,
        temperature=0.7,
    )
    config.horizon_steps = 12
    config.ctrl_steps = 4
    config.knot_steps = 3
    config.ref_steps = 1
    config.nq = 35
    config.nv = 34
    config.nu = 6
    config = compute_noise_schedule(config)
    ctrls = torch.zeros(12, 6)

    torch.manual_seed(123)
    actual = _sample_ctrls_impl(config, ctrls)
    torch.manual_seed(123)
    expected = ctrls + interp(
        torch.randn_like(config.noise_scale) * config.noise_scale,
        config.knot_steps,
    )

    assert torch.allclose(actual, expected)
