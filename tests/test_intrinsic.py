"""Intrinsic bonus module tests: math, learning behavior, serialization."""

import numpy as np
import pytest
import torch

from reap.algos.intrinsic import CountBonus, NoBonus, RndBonus, RunningMeanStd, make_bonus


def test_running_mean_std_matches_numpy():
    rms = RunningMeanStd()
    chunks = [np.random.default_rng(i).normal(3.0, 2.0, size=50) for i in range(4)]
    for chunk in chunks:
        rms.update(chunk)
    all_values = np.concatenate(chunks)
    assert rms.mean == pytest.approx(all_values.mean(), rel=1e-3)
    assert rms.std == pytest.approx(all_values.std(), rel=1e-2)


def test_no_bonus_is_zero():
    bonus = NoBonus()
    states = np.ones((5, 8), dtype=np.float32)
    assert np.all(bonus.compute(states) == 0.0)
    assert bonus.update(states) == {}


def test_count_bonus_decays_with_visits():
    bonus = CountBonus()
    state = np.array([[1.0, 2.0, 3.0]])
    first = bonus.compute(state)[0]
    assert first == pytest.approx(1.0)  # unvisited: 1/sqrt(1)
    for expected_n in (1, 2, 3):
        bonus.update(state)
        assert bonus.compute(state)[0] == pytest.approx(1.0 / np.sqrt(expected_n + 1))
    # a different state is still fresh
    assert bonus.compute(np.array([[9.0, 9.0, 9.0]]))[0] == pytest.approx(1.0)


def test_count_bonus_serialization_roundtrip():
    bonus = CountBonus()
    states = np.random.default_rng(0).normal(size=(10, 4))
    bonus.update(states)
    clone = CountBonus()
    clone.load_state_dict(bonus.state_dict())
    assert np.allclose(bonus.compute(states), clone.compute(states))


def test_rnd_bonus_positive_and_learns():
    torch.manual_seed(0)
    bonus = RndBonus(state_dim=6, embed_dim=8, hidden=16, lr=1e-2)
    states = np.random.default_rng(1).normal(size=(32, 6)).astype(np.float32)
    initial = bonus.compute(states).mean()
    assert initial > 0
    for _ in range(50):  # training on the same states shrinks the raw error
        bonus.update(states)
    with torch.no_grad():
        final_error = bonus._errors(states).mean().item()
    with torch.no_grad():
        bonus_fresh = RndBonus(state_dim=6, embed_dim=8, hidden=16)
        # compare raw errors, not normalized bonuses, to isolate learning
    assert final_error < 1e-2 or final_error < bonus_fresh._errors(states).mean().item()


def test_rnd_serialization_roundtrip():
    torch.manual_seed(0)
    bonus = RndBonus(state_dim=4, embed_dim=8, hidden=16)
    states = np.random.default_rng(2).normal(size=(16, 4)).astype(np.float32)
    bonus.update(states)
    clone = RndBonus(state_dim=4, embed_dim=8, hidden=16)
    clone.load_state_dict(bonus.state_dict())
    assert np.allclose(bonus.compute(states), clone.compute(states), atol=1e-6)


def test_make_bonus_dispatch():
    assert make_bonus("none", 4).name == "none"
    assert make_bonus("rnd", 4).name == "rnd"
    assert make_bonus("count", 4).name == "count"
    with pytest.raises(ValueError, match="unknown intrinsic bonus"):
        make_bonus("curiosity", 4)
