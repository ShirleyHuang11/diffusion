"""Transformer denoiser over trajectory windows.

Each window timestep is a token (linear projection of the state vector plus a
sinusoidal position embedding); the diffusion step and an optional condition
vector (e.g. a behavioral policy embedding for classifier-free guidance) are
embedded and added to every token. The head predicts the noise per timestep.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=positions.device) / half
    )
    angles = positions.float().unsqueeze(-1) * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class TrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        state_dim: int,
        window: int,
        cond_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.window = window
        self.cond_dim = cond_dim
        self.in_proj = nn.Linear(state_dim, d_model)
        self.step_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, d_model)
            # learned embedding standing in for "no condition" during CFG
            self.null_cond = nn.Parameter(torch.zeros(d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, state_dim)
        self.register_buffer(
            "positions", torch.arange(window, dtype=torch.float32), persistent=False
        )
        self.d_model = d_model

    def forward(
        self,
        noisy: torch.Tensor,  # (B, W, D)
        steps: torch.Tensor,  # (B,)
        cond: torch.Tensor | None = None,  # (B, cond_dim) or None for null
    ) -> torch.Tensor:
        tokens = self.in_proj(noisy)
        tokens = tokens + sinusoidal_embedding(self.positions, self.d_model)
        step_emb = self.step_mlp(sinusoidal_embedding(steps, self.d_model))
        tokens = tokens + step_emb.unsqueeze(1)
        if self.cond_dim > 0:
            if cond is None:
                cond_emb = self.null_cond.expand(noisy.shape[0], -1)
            else:
                cond_emb = self.cond_proj(cond)
            tokens = tokens + cond_emb.unsqueeze(1)
        return self.out_proj(self.encoder(tokens))
