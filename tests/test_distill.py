"""Distilled predictor tests: fit quality, fidelity gate, serialization."""

import numpy as np
import pytest

from reap.signals.distill import DistilledPredictor, distillation_fidelity_report


def synthetic_data(n=600, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, dim)).astype(np.float32)
    targets = 1.0 / (1.0 + np.exp(-states[:, 0] + 0.5 * states[:, 1]))  # smooth in-state map
    return states, targets.astype(np.float32)


def test_fit_learns_and_outputs_in_range():
    states, targets = synthetic_data()
    predictor = DistilledPredictor(state_dim=states.shape[1], seed=0)
    history = predictor.fit(states, targets, epochs=150)
    assert history[-1] < history[0] * 0.3
    predictions = predictor.predict(states)
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert np.abs(predictions - targets).mean() < 0.08


def test_fidelity_report_passes_after_training(tmp_path):
    states, targets = synthetic_data()
    holdout_states, holdout_targets = synthetic_data(n=200, seed=1)
    predictor = DistilledPredictor(state_dim=states.shape[1], seed=0)
    predictor.fit(states, targets, epochs=150)
    report = distillation_fidelity_report(
        predictor, holdout_states, holdout_targets,
        mae_max=0.10, report_path=tmp_path / "fidelity.json",
    )
    assert report["passed"] is True
    assert report["mae"] <= 0.10
    assert (tmp_path / "fidelity.json").is_file()


def test_fidelity_report_fails_for_untrained_predictor():
    holdout_states, holdout_targets = synthetic_data(n=200, seed=2)
    # an untrained net is near-constant 0.5; targets vary across [0, 1]
    predictor = DistilledPredictor(state_dim=holdout_states.shape[1], seed=3)
    report = distillation_fidelity_report(
        predictor, holdout_states, holdout_targets, mae_max=0.05
    )
    assert report["passed"] is False


def test_serialization_roundtrip():
    states, targets = synthetic_data(n=100)
    predictor = DistilledPredictor(state_dim=states.shape[1], seed=0)
    predictor.fit(states, targets, epochs=20)
    clone = DistilledPredictor(state_dim=states.shape[1], seed=9)
    clone.load_state_dict(predictor.state_dict())
    assert np.allclose(predictor.predict(states), clone.predict(states), atol=1e-6)


def test_input_normalization_handles_badly_scaled_features():
    """Without input normalization a feature at offset 1000 wrecks training;
    the predictor must normalize internally and serialize the statistics."""
    rng = np.random.default_rng(5)
    states = rng.normal(size=(600, 4)).astype(np.float32)
    states[:, 2] = states[:, 2] * 500.0 + 1000.0  # badly scaled feature
    targets = (1.0 / (1.0 + np.exp(-states[:, 0]))).astype(np.float32)
    predictor = DistilledPredictor(state_dim=4, seed=0)
    predictor.fit(states, targets, epochs=150)
    assert np.abs(predictor.predict(states) - targets).mean() < 0.08
    clone = DistilledPredictor(state_dim=4, seed=9)
    clone.load_state_dict(predictor.state_dict())
    assert np.allclose(clone.input_mean, predictor.input_mean)
    assert np.allclose(predictor.predict(states), clone.predict(states), atol=1e-6)


def test_fidelity_report_carries_provenance(tmp_path):
    states, targets = synthetic_data(n=100)
    predictor = DistilledPredictor(state_dim=states.shape[1], seed=0)
    predictor.fit(states, targets, epochs=20)
    report = distillation_fidelity_report(
        predictor, states[:20], targets[:20],
        provenance={"layout": "cramped_room", "seed": 0},
    )
    assert report["provenance"]["layout"] == "cramped_room"
    assert report["predictor_hidden"] == 64


def test_target_validation():
    predictor = DistilledPredictor(state_dim=3)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        predictor.fit(np.zeros((4, 3), dtype=np.float32), np.array([0.5, 1.2, 0.3, 0.1]))
    with pytest.raises(ValueError, match="equal length"):
        predictor.fit(np.zeros((4, 3), dtype=np.float32), np.array([0.5]))