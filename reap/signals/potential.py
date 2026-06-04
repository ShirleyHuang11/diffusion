"""The REAP potential: feasibility-gated propensity.

    Phi(s) = propensity(s) * 1[ feasibility(s) >= tau_gate ]

Propensity is the value being climbed; feasibility enters ONLY as a gate.
Values are looked up from precomputed per-state tables (direct-query mode
computes them on a state subsample; distilled predictors replace the tables
later). States missing from the table contribute zero potential — shaping is
silent where the signal has not been evaluated, which is safe under PBRS.
"""

from __future__ import annotations

import numpy as np

from reap.shaping.pbrs import Potential


def state_key(state: np.ndarray, precision: int = 4) -> bytes:
    return np.round(np.asarray(state, dtype=np.float64), precision).tobytes()


class ReapPotential(Potential):
    def __init__(self, tau_gate: float = 0.5, precision: int = 4):
        if not 0 <= tau_gate <= 1:
            raise ValueError(f"tau_gate must be in [0, 1], got {tau_gate}")
        self.tau_gate = tau_gate
        self.precision = precision
        self._propensity: dict[bytes, float] = {}
        self._feasibility: dict[bytes, float] = {}

    def update_tables(
        self, states: np.ndarray, propensity: np.ndarray, feasibility: np.ndarray
    ) -> None:
        if not (len(states) == len(propensity) == len(feasibility)):
            raise ValueError("states/propensity/feasibility must have equal length")
        for s, p, f in zip(states, propensity, feasibility):
            if not 0 <= p <= 1:
                raise ValueError(f"propensity {p} outside [0, 1]")
            key = state_key(s, self.precision)
            self._propensity[key] = float(p)
            self._feasibility[key] = float(f)

    @property
    def coverage(self) -> int:
        return len(self._propensity)

    def value(self, state: np.ndarray, steps_remaining: int) -> float:
        key = state_key(state, self.precision)
        if key not in self._propensity:
            return 0.0  # unevaluated states get zero potential (silent shaping)
        gate = self._feasibility[key] >= self.tau_gate
        return self._propensity[key] if gate else 0.0

    def state_dict(self) -> dict:
        return {
            "tau_gate": self.tau_gate,
            "precision": self.precision,
            "propensity": dict(self._propensity),
            "feasibility": dict(self._feasibility),
        }

    def load_state_dict(self, state: dict) -> None:
        self.tau_gate = state["tau_gate"]
        self.precision = state["precision"]
        self._propensity = dict(state["propensity"])
        self._feasibility = dict(state["feasibility"])
