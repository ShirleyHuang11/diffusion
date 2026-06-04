"""Generation-quality measurement and gating for the trajectory teacher.

The report measures sampled windows against a state validator (invalid-state
rate after projection to the valid manifold) and a success predicate
(endpoint-in-goal rate), plus an optional externally computed bridge
transition-consistency rate. Gates are configurable defaults; violating a
gate sets ``shaping_enabled = False`` — the violation is a first-class
logged result, never a silent failure.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path

import numpy as np


class StateValidator(abc.ABC):
    """Projects generated states onto the valid manifold and tests validity."""

    @abc.abstractmethod
    def project(self, states: np.ndarray) -> np.ndarray:
        """Map raw generated states to the nearest valid-manifold encoding."""

    @abc.abstractmethod
    def is_valid(self, states: np.ndarray) -> np.ndarray:
        """Boolean validity per state row."""


DEFAULT_GATES = {
    "invalid_state_rate_max": 0.10,
    "bridge_consistency_min": 0.80,
}


def generation_quality_report(
    sampled_windows: np.ndarray,  # (N, W, D) denormalized
    validator: StateValidator,
    success_fn=None,  # state -> bool membership in the goal set
    bridge_consistency_rate: float | None = None,
    gates: dict | None = None,
    report_path: str | Path | None = None,
) -> dict:
    gates = {**DEFAULT_GATES, **(gates or {})}
    n, w, d = sampled_windows.shape
    flat = sampled_windows.reshape(-1, d)
    projected = validator.project(flat)
    valid = validator.is_valid(projected)
    invalid_rate = float(1.0 - valid.mean()) if len(valid) else 1.0

    endpoint_success_rate = None
    if success_fn is not None:
        endpoints = validator.project(sampled_windows[:, -1])
        endpoint_success_rate = float(
            np.mean([bool(success_fn(s)) for s in endpoints])
        )

    violations = []
    if invalid_rate > gates["invalid_state_rate_max"]:
        violations.append(
            f"invalid_state_rate {invalid_rate:.3f} > {gates['invalid_state_rate_max']}"
        )
    if (
        bridge_consistency_rate is not None
        and bridge_consistency_rate < gates["bridge_consistency_min"]
    ):
        violations.append(
            f"bridge_consistency_rate {bridge_consistency_rate:.3f} < "
            f"{gates['bridge_consistency_min']}"
        )

    report = {
        "samples": int(n),
        "window": int(w),
        "invalid_state_rate": invalid_rate,
        "endpoint_success_rate": endpoint_success_rate,
        "bridge_consistency_rate": bridge_consistency_rate,
        "gates": gates,
        "gate_violations": violations,
        "shaping_enabled": not violations,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
