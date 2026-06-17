# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Different optimizers for the simulation results. 
Optimizers will take in initial guess of the control sequence, and output the optimized control sequence. 

Backends:
- sampling / sampling_fast: generic SPIDER sampled-MPC optimizers.
- receding: generic optimize-execute-shift MPC loop helper.
"""

from spider.optimizers.receding import (
    RecedingHorizonResult,
    SamplingMpcTask,
    run_receding_horizon,
    run_sampling_receding_mpc,
    sampling_mpc_metadata,
)

__all__ = [
    "RecedingHorizonResult",
    "SamplingMpcTask",
    "run_receding_horizon",
    "run_sampling_receding_mpc",
    "sampling_mpc_metadata",
]
