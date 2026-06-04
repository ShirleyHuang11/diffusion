"""Fixed-length trajectory windows sliced from a TrajectoryBuffer.

The teacher models windows of ``window`` consecutive joint states. Episodes
shorter than the window are skipped; longer episodes contribute every stride
offset. Per-dimension normalization statistics are computed once over the
training windows and applied to model inputs (and inverted on samples).
"""

from __future__ import annotations

import numpy as np
import torch

from reap.data.buffer import TrajectoryBuffer


class TrajectoryWindowDataset:
    def __init__(self, buffer: TrajectoryBuffer, window: int = 32, stride: int = 8):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.window = window
        self.state_dim = buffer.state_dim
        windows = []
        for ep in buffer.episodes:
            for start in range(0, len(ep) - window + 1, stride):
                windows.append(ep[start : start + window])
        if not windows:
            raise ValueError(
                f"no windows of length {window} available; longest episode is "
                f"{max((len(ep) for ep in buffer.episodes), default=0)} states"
            )
        self.windows = np.stack(windows).astype(np.float32)  # (N, W, D)
        self.mean = self.windows.reshape(-1, self.state_dim).mean(axis=0)
        self.std = self.windows.reshape(-1, self.state_dim).std(axis=0) + 1e-6

    def __len__(self) -> int:
        return len(self.windows)

    def normalize(self, states: np.ndarray | torch.Tensor):
        if isinstance(states, torch.Tensor):
            mean = torch.as_tensor(self.mean, device=states.device)
            std = torch.as_tensor(self.std, device=states.device)
            return (states - mean) / std
        return (states - self.mean) / self.std

    def denormalize(self, states: np.ndarray | torch.Tensor):
        if isinstance(states, torch.Tensor):
            mean = torch.as_tensor(self.mean, device=states.device)
            std = torch.as_tensor(self.std, device=states.device)
            return states * std + mean
        return states * self.std + self.mean

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        idx = rng.integers(0, len(self.windows), size=batch_size)
        batch = self.normalize(self.windows[idx])
        return torch.as_tensor(batch, dtype=torch.float32)
