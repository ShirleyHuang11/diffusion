"""Distilled predictors replacing direct teacher queries in the RL loop.

Direct propensity/feasibility queries cost many denoising passes per state;
training only needs two tiny MLP forward passes per transition. Each
predictor is distilled from a table of (state, direct-query value) pairs and
must pass a held-out fidelity check against the direct queries before it may
replace them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DistilledPredictor:
    """Small MLP regressor onto [0, 1] (sigmoid head)."""

    def __init__(self, state_dim: int, hidden: int = 64, lr: float = 1e-3, seed: int = 0):
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        seed: int = 0,
    ) -> list[float]:
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if len(states) != len(targets):
            raise ValueError("states and targets must have equal length")
        if np.any((targets < 0) | (targets > 1)):
            raise ValueError("targets must lie in [0, 1]")
        x = torch.as_tensor(states)
        y = torch.as_tensor(targets).unsqueeze(-1)
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            idx = rng.permutation(len(x))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(x), batch_size):
                mb = idx[start : start + batch_size]
                pred = torch.sigmoid(self.net(x[mb]))
                loss = ((pred - y[mb]) ** 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1
            history.append(epoch_loss / batches)
        return history

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        with torch.no_grad():
            values = torch.sigmoid(self.net(torch.as_tensor(states))).squeeze(-1).numpy()
        return values.astype(np.float64)

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
        }

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])


def distillation_fidelity_report(
    predictor: DistilledPredictor,
    holdout_states: np.ndarray,
    direct_values: np.ndarray,
    mae_max: float = 0.10,
    report_path: str | Path | None = None,
) -> dict:
    """Compare the predictor against direct queries on held-out states.

    The predictor may replace direct queries only when ``passed`` is true;
    a failed check is a logged result that keeps direct-query mode active.
    """
    direct_values = np.asarray(direct_values, dtype=np.float64)
    predicted = predictor.predict(holdout_states)
    errors = np.abs(predicted - direct_values)
    report = {
        "holdout_states": int(len(holdout_states)),
        "mae": float(errors.mean()),
        "max_abs_error": float(errors.max()),
        "mae_max": mae_max,
        "passed": bool(errors.mean() <= mae_max),
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
