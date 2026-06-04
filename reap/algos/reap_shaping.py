"""Integrated REAP shaping: snapshot-pinned potentials with gated enablement.

The controller owns the REAP potential (predictor-backed or table-backed),
the calibration ladder, and the refresh schedule. One immutable snapshot is
pinned per rollout batch: both endpoints of every transition in a batch see
the same potential and the same beta; refreshing while a batch is pinned is
a structural error. Enablement is read from the committed teacher-quality
artifact — when gates failed there, every shaping term is exactly zero and
the gate reason is logged as a first-class event.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from reap.calibration import CalibrationLadder
from reap.signals.distill import DistilledPredictor
from reap.signals.potential import ReapPotential


class PredictorPotential:
    """Phi(s) = p_hat(s) * 1[f_hat(s) >= tau] from distilled predictors."""

    def __init__(self, p_hat: DistilledPredictor, f_hat: DistilledPredictor, tau_gate: float):
        self.p_hat = p_hat
        self.f_hat = f_hat
        self.tau_gate = tau_gate

    def value(self, state: np.ndarray, steps_remaining: int) -> float:
        state = np.asarray(state, dtype=np.float32)[None, :]
        if float(self.f_hat.predict(state)[0]) < self.tau_gate:
            return 0.0
        return float(self.p_hat.predict(state)[0])

    def state_dict(self) -> dict:
        return {
            "p_hat": self.p_hat.state_dict(),
            "f_hat": self.f_hat.state_dict(),
            "tau_gate": self.tau_gate,
        }

    def load_state_dict(self, state: dict) -> None:
        self.p_hat.load_state_dict(state["p_hat"])
        self.f_hat.load_state_dict(state["f_hat"])
        self.tau_gate = state["tau_gate"]

    def update_tables(self, states, propensity, feasibility) -> None:
        # predictor-backed potentials are refreshed by refitting p_hat/f_hat
        # (done by the refresher itself); record the refresh size only
        self.last_refresh_states = int(len(states))


class ZeroPotential:
    """Null potential for scopes whose quality gate disables shaping."""

    def value(self, state, steps_remaining) -> float:
        return 0.0

    def update_tables(self, states, propensity, feasibility) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class PotentialSnapshot:
    """Immutable view of the potential + beta for exactly one rollout batch."""

    def __init__(self, snapshot_id: int, potential, beta: float, enabled: bool):
        self.snapshot_id = snapshot_id
        self._potential = potential
        self.beta = beta if enabled else 0.0
        self.enabled = enabled

    def value(self, env, joint_state, steps_remaining) -> float:
        if not self.enabled:
            return 0.0
        return float(self._potential.value(np.asarray(joint_state), int(steps_remaining)))


class ReapShapingController:
    """Owns potential, ladder, refresh schedule, and the event log."""

    def __init__(
        self,
        potential,
        ladder: CalibrationLadder,
        quality_report: dict,
        refresh_every_updates: int = 50,
        refresher=None,  # callable() -> (states, propensity, feasibility) or None
        event_sink=None,  # callable(dict) invoked as each event occurs
    ):
        if refresh_every_updates < 1:
            raise ValueError("refresh_every_updates must be >= 1")
        self.potential = potential
        self.ladder = ladder
        self.refresh_every = refresh_every_updates
        self.refresher = refresher
        self.event_sink = event_sink
        self.enabled = bool(quality_report.get("shaping_enabled", False))
        violations = quality_report.get("gate_violations") or []
        self.disabled_reason = (
            None if self.enabled else ("; ".join(violations) or "quality gates not passed")
        )
        self.snapshot_id = 0
        self.refresh_count = 0
        self._pinned = False
        self.events: list[dict] = []
        self._emit(
            {
                "type": "gate",
                "enabled": self.enabled,
                "reason": self.disabled_reason,
            }
        )

    def _emit(self, event: dict) -> dict:
        event = {"time": time.time(), **event}
        self.events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)
        return event

    @contextmanager
    def pin(self):
        """Pin one snapshot for one rollout batch; refresh is forbidden inside."""
        if self._pinned:
            raise RuntimeError("a potential snapshot is already pinned")
        self._pinned = True
        try:
            yield PotentialSnapshot(
                self.snapshot_id, self.potential, self.ladder.beta, self.enabled
            )
        finally:
            self._pinned = False

    def maybe_refresh(
        self,
        update_index: int,
        calibration_predicted: np.ndarray | None = None,
        calibration_realized: np.ndarray | None = None,
        calibration_fn=None,
    ) -> dict | None:
        """Refresh the propensity tables every K updates (between batches only).

        ``calibration_fn() -> (predicted, realized)`` is evaluated AFTER the
        refresher runs, so calibration always governs the refreshed signal
        that will shape the next batch (not the stale pre-refresh one).
        """
        if update_index == 0 or update_index % self.refresh_every != 0:
            return None
        if self._pinned:
            raise RuntimeError(
                "refresh attempted while a rollout batch is pinned; snapshots "
                "must never change mid-batch"
            )
        event: dict = {
            "type": "refresh",
            "update_index": int(update_index),
            "snapshot_id_before": self.snapshot_id,
        }
        if self.refresher is not None:
            states, propensity, feasibility = self.refresher()
            self.potential.update_tables(states, propensity, feasibility)
            event["refreshed_states"] = int(len(states))
        else:
            event["refreshed_states"] = 0
            event["note"] = "no refresher configured"
        if calibration_fn is not None:
            calibration_predicted, calibration_realized = calibration_fn()
        if calibration_predicted is not None and calibration_realized is not None:
            cal = self.ladder.check(calibration_predicted, calibration_realized)
            event["calibration"] = {
                k: cal[k] for k in ("raw_ece", "brier", "action", "beta_after", "alert")
            }
            if cal["action"] == "disabled":
                self.enabled = False
                self.disabled_reason = "calibration ladder disabled shaping (beta=0)"
        self.snapshot_id += 1
        self.refresh_count += 1
        event["snapshot_id_after"] = self.snapshot_id
        return self._emit(event)

    # checkpoint payload: ALL stateful pieces travel with the trainer
    def state_dict(self) -> dict:
        return {
            "potential": self.potential.state_dict(),
            "ladder": self.ladder.state_dict(),
            "snapshot_id": self.snapshot_id,
            "refresh_count": self.refresh_count,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "refresh_every": self.refresh_every,
            "events": list(self.events),
        }

    def load_state_dict(self, state: dict) -> None:
        self.potential.load_state_dict(state["potential"])
        self.ladder.load_state_dict(state["ladder"])
        self.snapshot_id = state["snapshot_id"]
        self.refresh_count = state["refresh_count"]
        self.enabled = state["enabled"]
        self.disabled_reason = state["disabled_reason"]
        self.refresh_every = state["refresh_every"]
        self.events = list(state["events"])
