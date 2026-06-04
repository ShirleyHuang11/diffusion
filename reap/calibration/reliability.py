"""Reliability measurement (bins, ECE, Brier), isotonic recalibration, ladder.

Predicted propensities are compared against realized success outcomes on a
held-out set that must be disjoint from the teacher's training data. On a
miscalibration the response ladder escalates: isotonic recalibration first;
if still miscalibrated, shrink the shaping weight; at the floor, disable
shaping entirely (beta = 0) and raise the alert flag. Every rung is a logged,
first-class event.
"""

from __future__ import annotations

import numpy as np


def _validate(predicted: np.ndarray, realized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if predicted.shape != realized.shape or predicted.ndim != 1:
        raise ValueError("predicted and realized must be equal-length 1-D arrays")
    if len(predicted) == 0:
        raise ValueError("calibration check needs at least one sample")
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    if np.any((realized < 0) | (realized > 1)):
        raise ValueError("realized outcomes must lie in [0, 1]")
    return predicted, realized


def reliability_bins(predicted, realized, n_bins: int = 10) -> list[dict]:
    predicted, realized = _validate(predicted, realized)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (predicted >= lo) & (predicted < hi if hi < 1.0 else predicted <= hi)
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": int(mask.sum()),
                "predicted_mean": float(predicted[mask].mean()) if mask.any() else None,
                "realized_mean": float(realized[mask].mean()) if mask.any() else None,
            }
        )
    return bins


def expected_calibration_error(predicted, realized, n_bins: int = 10) -> float:
    predicted, realized = _validate(predicted, realized)
    total = len(predicted)
    ece = 0.0
    for b in reliability_bins(predicted, realized, n_bins):
        if b["count"]:
            ece += (b["count"] / total) * abs(b["predicted_mean"] - b["realized_mean"])
    return float(ece)


def brier_score(predicted, realized) -> float:
    predicted, realized = _validate(predicted, realized)
    return float(np.mean((predicted - realized) ** 2))


def isotonic_fit(predicted, realized):
    """Pool-adjacent-violators isotonic regression of realized on predicted.

    Returns a monotone recalibration function mapping raw predictions to
    recalibrated probabilities (piecewise-constant with interpolation).
    """
    predicted, realized = _validate(predicted, realized)
    order = np.argsort(predicted, kind="stable")
    x = predicted[order]
    y = realized[order].copy()
    weights = np.ones_like(y)

    # pool adjacent violators
    values = list(y)
    wts = list(weights)
    starts = list(range(len(y)))
    i = 0
    while i < len(values) - 1:
        if values[i] > values[i + 1] + 1e-12:
            merged = (values[i] * wts[i] + values[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
            values[i] = merged
            wts[i] += wts[i + 1]
            del values[i + 1], wts[i + 1], starts[i + 1]
            while i > 0 and values[i - 1] > values[i] + 1e-12:
                merged = (values[i - 1] * wts[i - 1] + values[i] * wts[i]) / (
                    wts[i - 1] + wts[i]
                )
                values[i - 1] = merged
                wts[i - 1] += wts[i]
                del values[i], wts[i], starts[i]
                i -= 1
        else:
            i += 1

    block_x = np.array([x[s] for s in starts], dtype=np.float64)
    block_y = np.clip(np.array(values, dtype=np.float64), 0.0, 1.0)

    def recalibrate(p):
        p = np.asarray(p, dtype=np.float64)
        return np.interp(p, block_x, block_y, left=block_y[0], right=block_y[-1])

    return recalibrate


def ensure_disjoint(calibration_keys, training_keys) -> None:
    """Raise when the calibration holdout overlaps the teacher training data."""
    overlap = set(calibration_keys) & set(training_keys)
    if overlap:
        raise ValueError(
            f"calibration holdout overlaps teacher training data on "
            f"{len(overlap)} state keys; the check would be optimistically biased"
        )


class CalibrationLadder:
    """Automated escalation: ok -> isotonic -> shrink beta -> disable + alert."""

    def __init__(
        self,
        beta: float,
        ece_gate: float = 0.15,
        shrink_factor: float = 0.5,
        beta_floor: float = 1e-3,
        n_bins: int = 10,
        resolution_floor: float = 0.02,
    ):
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta = float(beta)
        self.initial_beta = float(beta)
        self.ece_gate = ece_gate
        self.shrink_factor = shrink_factor
        self.beta_floor = beta_floor
        self.n_bins = n_bins
        # a recalibrated signal must retain discrimination: isotonic collapsing
        # to a near-constant is marginally calibrated but useless as a potential
        self.resolution_floor = resolution_floor
        self.recalibrator = None
        self.alert = False
        self.history: list[dict] = []

    def check(self, predicted, realized) -> dict:
        """Run one calibration check; escalate on failure. Returns the event."""
        predicted, realized = _validate(predicted, realized)
        raw_ece = expected_calibration_error(predicted, realized, self.n_bins)
        event = {
            "raw_ece": raw_ece,
            "brier": brier_score(predicted, realized),
            "ece_gate": self.ece_gate,
            "bins": reliability_bins(predicted, realized, self.n_bins),
            "beta_before": self.beta,
        }

        if raw_ece <= self.ece_gate:
            event["action"] = "ok"
            self.recalibrator = None
        else:
            # fit on one half, evaluate on the other, so the fit cannot grade
            # its own homework; require retained discrimination on top of ECE
            fit_half, eval_half = slice(0, None, 2), slice(1, None, 2)
            recalibrator = isotonic_fit(predicted[fit_half], realized[fit_half])
            recal_eval = recalibrator(predicted[eval_half])
            post_ece = expected_calibration_error(
                recal_eval, realized[eval_half], self.n_bins
            )
            resolution = float(np.std(recal_eval))
            event["post_isotonic_ece"] = post_ece
            event["post_isotonic_resolution"] = resolution
            if post_ece <= self.ece_gate and resolution >= self.resolution_floor:
                event["action"] = "recalibrated"
                self.recalibrator = recalibrator
            else:
                self.recalibrator = None
                self.beta *= self.shrink_factor
                if self.beta < self.beta_floor:
                    self.beta = 0.0
                    self.alert = True
                    event["action"] = "disabled"
                else:
                    event["action"] = "shrink_beta"

        event["beta_after"] = self.beta
        event["alert"] = self.alert
        self.history.append(event)
        return event

    def apply(self, predicted):
        """Recalibrate predictions with the current isotonic fit (if any)."""
        if self.recalibrator is None:
            return np.asarray(predicted, dtype=np.float64)
        return self.recalibrator(predicted)

    def state_dict(self) -> dict:
        # the isotonic fit is refit at every check; only scalars persist
        return {
            "beta": self.beta,
            "initial_beta": self.initial_beta,
            "ece_gate": self.ece_gate,
            "shrink_factor": self.shrink_factor,
            "beta_floor": self.beta_floor,
            "n_bins": self.n_bins,
            "resolution_floor": self.resolution_floor,
            "alert": self.alert,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict) -> None:
        self.beta = state["beta"]
        self.initial_beta = state["initial_beta"]
        self.ece_gate = state["ece_gate"]
        self.shrink_factor = state["shrink_factor"]
        self.beta_floor = state["beta_floor"]
        self.n_bins = state["n_bins"]
        self.resolution_floor = state["resolution_floor"]
        self.alert = state["alert"]
        self.history = list(state["history"])
        self.recalibrator = None
