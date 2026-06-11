"""WBC policy loader for the G1 tracking checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from spider.tasks.g1_wbc.constants import ACTION_DIM, DEFAULT_CKPT_DIRS, OBS_DIM


class WbcActor(nn.Module):
    """MLP actor compatible with the saved WXY checkpoints."""

    def __init__(
        self,
        input_dim: int = OBS_DIM,
        hidden_dims: tuple[int, ...] = (2048, 2048, 1024, 1024, 512, 256, 128),
        output_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.ELU())
        self.mlp = nn.Sequential(*modules)
        self.register_buffer("obs_mean", torch.zeros(1, input_dim))
        self.register_buffer("obs_std", torch.ones(1, input_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - self.obs_mean) / (self.obs_std + 1.0e-2)
        return self.mlp(obs)


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if checkpoint in DEFAULT_CKPT_DIRS:
        directory = DEFAULT_CKPT_DIRS[str(checkpoint)]
        candidates = sorted(directory.glob("model_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt checkpoint found under {directory}")
        return candidates[-1].resolve()
    if path.is_dir():
        candidates = sorted(path.glob("model_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt checkpoint found under {path}")
        return candidates[-1].resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_wbc_actor(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda:0",
) -> WbcActor:
    ckpt_path = resolve_checkpoint_path(checkpoint)
    checkpoint_data = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint_data.get("actor_state_dict", checkpoint_data)
    actor = WbcActor()

    actor.obs_mean.copy_(state_dict["obs_normalizer._mean"])
    actor.obs_std.copy_(state_dict["obs_normalizer._std"])

    mlp_state = {}
    actor_state = actor.state_dict()
    for key, value in state_dict.items():
        if key.startswith("mlp."):
            mlp_state[key] = value
    missing = [key for key in actor_state if key.startswith("mlp.") and key not in mlp_state]
    if missing:
        raise ValueError(f"Checkpoint {ckpt_path} missing actor weights: {missing[:4]}")
    actor.load_state_dict({**actor.state_dict(), **mlp_state}, strict=True)
    actor.to(device)
    actor.eval()
    return actor
