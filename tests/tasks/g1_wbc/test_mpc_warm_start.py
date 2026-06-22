from __future__ import annotations

# ruff: noqa: D101,D102
import sys
import unittest
from unittest.mock import patch

import torch

from spider.tasks.g1_wbc import evaluate, mpc


class G1WbcMpcWarmStartTest(unittest.TestCase):
    def test_config_defaults_preserve_disabled_warm_start(self) -> None:
        config = mpc.G1WbcMpcConfig()

        self.assertFalse(config.use_warm_start)
        self.assertEqual(config.warm_start_source, "best")
        self.assertEqual(config.warm_start_decay, 1.0)

    def test_parse_args_builds_warm_start_config(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint",
            "--mpc-warm-start",
            "--mpc-warm-start-source",
            "mean",
            "--mpc-warm-start-decay",
            "0.25",
        ]

        with patch.object(sys, "argv", argv):
            args = evaluate._parse_args()
        config = evaluate._build_mpc_config(args)

        self.assertTrue(config.use_warm_start)
        self.assertEqual(config.warm_start_source, "mean")
        self.assertEqual(config.warm_start_decay, 0.25)

    def test_warm_start_mean_delta_shifts_decays_and_zero_pads_full(self) -> None:
        config = mpc.G1WbcMpcConfig(
            sampling_mode="full",
            freeze_first_frame=False,
            warm_start_decay=0.5,
        )
        dim = mpc.QPOS_DIM - 1
        previous_delta = torch.arange(6 * dim, dtype=torch.float32).reshape(6, dim)

        warm_start = mpc._warm_start_mean_delta(
            previous_delta,
            execute_steps=2,
            horizon=5,
            config=config,
        )

        assert warm_start is not None
        expected = torch.cat(
            [
                previous_delta[2:] * 0.5,
                torch.zeros(1, dim, dtype=torch.float32),
            ],
            dim=0,
        )
        torch.testing.assert_close(warm_start, expected)

    def test_warm_start_mean_delta_converts_shifted_horizon_to_knots(self) -> None:
        config = mpc.G1WbcMpcConfig(
            sampling_mode="knot",
            knot_count=3,
            freeze_first_frame=False,
        )
        dim = mpc.QPOS_DIM - 1
        previous_delta = torch.arange(8, dtype=torch.float32)[:, None].expand(8, dim)

        warm_start = mpc._warm_start_mean_delta(
            previous_delta,
            execute_steps=2,
            horizon=5,
            config=config,
        )

        assert warm_start is not None
        expected = torch.tensor([2.0, 4.0, 6.0])[:, None].expand(3, dim)
        torch.testing.assert_close(warm_start, expected)


if __name__ == "__main__":
    unittest.main()
