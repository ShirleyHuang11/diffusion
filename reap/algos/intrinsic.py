"""Intrinsic exploration bonuses computed on the centralized joint state.

Bonuses are auxiliary training-time signals only: they are added to the reward
used for advantage estimation but are logged in the ``intrinsic`` metrics
channel and never touch the ``extrinsic`` channel. Every module is stateful
(predictor weights, normalizers, visit counts) and therefore serializes its
full state into checkpoints.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn


class RunningMeanStd:
    """Welford-style running mean/variance for bonus normalization."""

    def __init__(self):
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        batch_mean = values.mean()
        batch_var = values.var()
        batch_count = values.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state: dict) -> None:
        self.count = state["count"]
        self.mean = state["mean"]
        self.var = state["var"]


class NoBonus:
    """Disabled intrinsic bonus (vanilla MAPPO)."""

    name = "none"

    def compute(self, joint_states: np.ndarray) -> np.ndarray:
        return np.zeros(len(joint_states), dtype=np.float32)

    def update(self, joint_states: np.ndarray) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class RndBonus:
    """Random Network Distillation (Burda et al. 2018) on joint states.

    Bonus is the predictor's error against a frozen randomly-initialized
    target network, normalized by a running standard deviation.
    """

    name = "rnd"

    def __init__(self, state_dim: int, embed_dim: int = 64, hidden: int = 128, lr: float = 1e-4):
        def mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim)
            )

        self.target = mlp()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = mlp()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.normalizer = RunningMeanStd()

    def _errors(self, joint_states: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(joint_states), dtype=torch.float32)
        with torch.no_grad():
            target = self.target(x)
        pred = self.predictor(x)
        return ((pred - target) ** 2).mean(dim=-1)

    def compute(self, joint_states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            errors = self._errors(joint_states).numpy()
        return (errors / self.normalizer.std).astype(np.float32)

    def update(self, joint_states: np.ndarray) -> dict:
        errors = self._errors(joint_states)
        loss = errors.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.normalizer.update(errors.detach().numpy())
        return {"rnd_loss": float(loss.item())}

    def state_dict(self) -> dict:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])


class CountBonus:
    """Count-based exploration bonus 1/sqrt(N(s)) over hashed joint states."""

    name = "count"

    def __init__(self, precision: int = 1):
        self.precision = precision
        self.counts: dict[str, int] = {}

    def _key(self, joint_state: np.ndarray) -> str:
        rounded = np.round(np.asarray(joint_state, dtype=np.float64), self.precision)
        return hashlib.sha1(rounded.tobytes()).hexdigest()

    def compute(self, joint_states: np.ndarray) -> np.ndarray:
        bonuses = np.empty(len(joint_states), dtype=np.float32)
        for i, s in enumerate(joint_states):
            n = self.counts.get(self._key(s), 0)
            bonuses[i] = 1.0 / np.sqrt(n + 1.0)
        return bonuses

    def update(self, joint_states: np.ndarray) -> dict:
        for s in joint_states:
            key = self._key(s)
            self.counts[key] = self.counts.get(key, 0) + 1
        return {"count_table_size": float(len(self.counts))}

    def state_dict(self) -> dict:
        return {"precision": self.precision, "counts": dict(self.counts)}

    def load_state_dict(self, state: dict) -> None:
        self.precision = state["precision"]
        self.counts = dict(state["counts"])


def make_bonus(kind: str, state_dim: int, **kwargs):
    """Construct an intrinsic bonus module by name (none / rnd / count)."""
    if kind == "none":
        return NoBonus()
    if kind == "rnd":
        return RndBonus(state_dim, **kwargs)
    if kind == "count":
        return CountBonus(**kwargs)
    raise ValueError(f"unknown intrinsic bonus kind {kind!r}")
