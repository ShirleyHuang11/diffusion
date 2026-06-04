"""Calibration metric and response-ladder tests, including fault injection."""

import numpy as np
import pytest

from reap.calibration import (
    CalibrationLadder,
    brier_score,
    ensure_disjoint,
    expected_calibration_error,
    isotonic_fit,
    reliability_bins,
)


def test_perfectly_calibrated_has_zero_ece():
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 1, size=20_000)
    realized = (rng.uniform(0, 1, size=20_000) < predicted).astype(float)
    assert expected_calibration_error(predicted, realized) < 0.02
    assert brier_score(predicted, realized) < 0.30


def test_overconfident_predictions_show_high_ece():
    predicted = np.full(1000, 0.9)
    realized = np.zeros(1000)  # never succeeds: maximal overconfidence
    assert expected_calibration_error(predicted, realized) == pytest.approx(0.9)
    assert brier_score(predicted, realized) == pytest.approx(0.81)


def test_reliability_bins_structure():
    bins = reliability_bins(np.array([0.05, 0.95, 0.96]), np.array([0.0, 1.0, 1.0]), n_bins=10)
    assert len(bins) == 10
    assert bins[0]["count"] == 1
    assert bins[-1]["count"] == 2
    assert bins[-1]["realized_mean"] == 1.0
    assert sum(b["count"] for b in bins) == 3


def test_input_validation():
    with pytest.raises(ValueError, match="equal-length"):
        expected_calibration_error(np.array([0.5]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        expected_calibration_error(np.array([1.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="at least one"):
        expected_calibration_error(np.array([]), np.array([]))


def test_isotonic_fit_is_monotone_and_reduces_ece():
    rng = np.random.default_rng(1)
    predicted = rng.uniform(0, 1, size=5000)
    true_prob = np.clip(predicted * 0.4, 0, 1)  # systematic overconfidence
    realized = (rng.uniform(0, 1, size=5000) < true_prob).astype(float)

    raw_ece = expected_calibration_error(predicted, realized)
    recalibrate = isotonic_fit(predicted, realized)
    post_ece = expected_calibration_error(recalibrate(predicted), realized)
    assert post_ece < raw_ece * 0.5

    grid = np.linspace(0, 1, 101)
    mapped = recalibrate(grid)
    assert np.all(np.diff(mapped) >= -1e-9)  # monotone
    assert np.all((mapped >= 0) & (mapped <= 1))


def test_ladder_ok_when_calibrated():
    rng = np.random.default_rng(2)
    predicted = rng.uniform(0, 1, size=5000)
    realized = (rng.uniform(0, 1, size=5000) < predicted).astype(float)
    ladder = CalibrationLadder(beta=1.0)
    event = ladder.check(predicted, realized)
    assert event["action"] == "ok"
    assert ladder.beta == 1.0
    assert ladder.alert is False


def test_ladder_recalibrates_fixable_miscalibration():
    rng = np.random.default_rng(3)
    predicted = rng.uniform(0, 1, size=5000)
    true_prob = np.clip(predicted * 0.3, 0, 1)
    realized = (rng.uniform(0, 1, size=5000) < true_prob).astype(float)
    ladder = CalibrationLadder(beta=1.0)
    event = ladder.check(predicted, realized)
    assert event["action"] == "recalibrated"
    assert ladder.beta == 1.0  # beta untouched when isotonic suffices
    # applying the fit moves predictions toward realized frequencies
    assert expected_calibration_error(ladder.apply(predicted), realized) <= ladder.ece_gate


def test_ladder_escalates_to_shrink_and_disable_on_unfixable_fault():
    """Fault injection: anti-correlated predictions cannot be monotonically
    recalibrated; the ladder shrinks beta each refresh and ends disabled."""
    rng = np.random.default_rng(4)
    predicted = rng.uniform(0, 1, size=4000)
    realized = (rng.uniform(0, 1, size=4000) < (1.0 - predicted)).astype(float)
    ladder = CalibrationLadder(beta=1.0, shrink_factor=0.5, beta_floor=0.3)

    first = ladder.check(predicted, realized)
    assert first["action"] == "shrink_beta"
    assert ladder.beta == pytest.approx(0.5)

    second = ladder.check(predicted, realized)
    assert second["action"] == "disabled"  # 0.25 < beta_floor 0.3
    assert ladder.beta == 0.0
    assert ladder.alert is True
    assert [e["action"] for e in ladder.history] == ["shrink_beta", "disabled"]


def test_ladder_state_roundtrip():
    ladder = CalibrationLadder(beta=0.8)
    ladder.check(np.full(100, 0.9), np.zeros(100))
    clone = CalibrationLadder(beta=1.0)
    clone.load_state_dict(ladder.state_dict())
    assert clone.beta == ladder.beta
    assert clone.history == ladder.history


def test_disjointness_check():
    ensure_disjoint({b"a", b"b"}, {b"c"})  # disjoint: fine
    with pytest.raises(ValueError, match="overlaps"):
        ensure_disjoint({b"a", b"b"}, {b"b", b"c"})