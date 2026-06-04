"""REAP signal tests on the chain fixture: embedding, propensity, feasibility, potential."""

import numpy as np
import pytest
import torch

from reap.algos.mappo import MappoTrainer
from reap.data import collect_warmup
from reap.diffusion import GaussianDiffusion, TrajectoryDenoiser, TrajectoryWindowDataset
from reap.diffusion.ddpm import train_teacher
from reap.signals import (
    BehavioralPolicyEmbedding,
    ReapPotential,
    TransitionChecker,
    collect_probe_observations,
    estimate_feasibility,
    estimate_propensity,
)
from tests.chain_env import ChainEnv
from tests.test_diffusion import OneHotValidator

LENGTH, WINDOW = 4, 4


class ChainChecker(TransitionChecker):
    def realizable(self, state, next_state):
        return int(np.argmax(next_state)) - int(np.argmax(state)) in (0, 1)


class NothingChecker(TransitionChecker):
    def realizable(self, state, next_state):
        return False


def success_fn(state, start_state=None):
    return bool(np.argmax(state) == LENGTH - 1)


@pytest.fixture(scope="module")
def teacher():
    env = ChainEnv(length=LENGTH, horizon=8)
    buffer, _ = collect_warmup(
        env, ladder=[("solver", lambda lo, js: [1, 1])],
        min_successes=30, max_env_steps=2000,
    )
    dataset = TrajectoryWindowDataset(buffer, window=WINDOW, stride=1)
    torch.manual_seed(0)
    model = TrajectoryDenoiser(
        state_dim=LENGTH, window=WINDOW, d_model=32, nhead=2, num_layers=1
    )
    diffusion = GaussianDiffusion(num_steps=25)
    train_teacher(model, diffusion, dataset, steps=300, batch_size=32, seed=0)
    return dataset, model, diffusion


def test_probe_embedding_is_deterministic_and_policy_sensitive():
    env = ChainEnv(length=LENGTH, horizon=8)
    rng = np.random.default_rng(0)
    probes = collect_probe_observations(
        env, lambda lo, js: rng.integers(0, 2, size=2).tolist(), n_probes=6, rng=rng
    )
    embedding = BehavioralPolicyEmbedding(probes)
    trainer_a = MappoTrainer(ChainEnv(LENGTH, 8), {"hidden_size": 16, "rollout_length": 8}, seed=0)
    trainer_b = MappoTrainer(ChainEnv(LENGTH, 8), {"hidden_size": 16, "rollout_length": 8}, seed=7)
    e_a1 = embedding.embed(trainer_a.nets)
    e_a2 = embedding.embed(trainer_a.nets)
    e_b = embedding.embed(trainer_b.nets)
    assert e_a1.shape == (6 * 2 * 2,)  # probes * agents * actions
    assert np.allclose(e_a1, e_a2)  # deterministic for fixed nets
    assert not np.allclose(e_a1, e_b)  # different policies embed differently
    # rows are probability vectors
    assert np.allclose(e_a1.reshape(-1, 2).sum(axis=1), 1.0, atol=1e-5)


def test_probe_shape_validation():
    with pytest.raises(ValueError, match="probe observations"):
        BehavioralPolicyEmbedding(np.zeros((3, 4)))


def test_propensity_range_and_signal(teacher):
    dataset, model, diffusion = teacher
    states = np.eye(LENGTH, dtype=np.float32)  # every chain cell
    propensity = estimate_propensity(
        diffusion, model, dataset, states,
        policy_embedding=None, success_fn=success_fn, validator=OneHotValidator(),
        samples_per_state=12, generator=torch.Generator().manual_seed(0),
    )
    assert propensity.shape == (LENGTH,)
    assert np.all((propensity >= 0) & (propensity <= 1))
    # the teacher was trained on solver trajectories: the goal-adjacent state
    # should look at least as promising as the start state
    assert propensity[LENGTH - 2] >= propensity[0] - 0.25


def test_propensity_conditional_plumbing(teacher):
    dataset, _, diffusion = teacher
    torch.manual_seed(1)
    cond_model = TrajectoryDenoiser(
        state_dim=LENGTH, window=WINDOW, cond_dim=5, d_model=32, nhead=2, num_layers=1
    )
    propensity = estimate_propensity(
        diffusion, cond_model, dataset, np.eye(LENGTH, dtype=np.float32)[:2],
        policy_embedding=np.ones(5, dtype=np.float32), success_fn=success_fn,
        validator=OneHotValidator(), samples_per_state=4,
        guidance_scale=1.5, generator=torch.Generator().manual_seed(2),
    )
    assert propensity.shape == (2,)
    assert np.all((propensity >= 0) & (propensity <= 1))


def test_feasibility_consistency_filter(teacher):
    dataset, model, diffusion = teacher
    states = np.eye(LENGTH, dtype=np.float32)[:2]
    goals = np.eye(LENGTH, dtype=np.float32)[LENGTH - 1 :]
    feasible = estimate_feasibility(
        diffusion, model, dataset, states, goals,
        validator=OneHotValidator(), checker=ChainChecker(),
        samples_per_state=8, generator=torch.Generator().manual_seed(3),
    )
    assert feasible.shape == (2,)
    assert np.all((feasible >= 0) & (feasible <= 1))

    blocked = estimate_feasibility(
        diffusion, model, dataset, states, goals,
        validator=OneHotValidator(), checker=NothingChecker(),
        samples_per_state=8, generator=torch.Generator().manual_seed(3),
    )
    assert np.allclose(blocked, 0.0)  # nothing realizable -> feasibility zero


def test_feasibility_requires_goals(teacher):
    dataset, model, diffusion = teacher
    with pytest.raises(ValueError, match="goal-set example"):
        estimate_feasibility(
            diffusion, model, dataset, np.eye(LENGTH, dtype=np.float32)[:1],
            np.empty((0, LENGTH)), validator=OneHotValidator(), checker=ChainChecker(),
        )


def test_reap_potential_gating_and_tables():
    potential = ReapPotential(tau_gate=0.5)
    states = np.eye(3, dtype=np.float32)
    potential.update_tables(
        states,
        propensity=np.array([0.9, 0.7, 0.4]),
        feasibility=np.array([0.8, 0.2, 0.6]),  # middle state fails the gate
    )
    assert potential.coverage == 3
    assert potential.value(states[0], 5) == pytest.approx(0.9)
    assert potential.value(states[1], 5) == 0.0  # gated out
    assert potential.value(states[2], 5) == pytest.approx(0.4)
    assert potential.value(np.array([9.0, 9.0, 9.0], dtype=np.float32), 5) == 0.0  # unseen

    clone = ReapPotential()
    clone.load_state_dict(potential.state_dict())
    assert clone.value(states[0], 5) == pytest.approx(0.9)


def test_reap_potential_validation():
    with pytest.raises(ValueError, match="tau_gate"):
        ReapPotential(tau_gate=1.5)
    potential = ReapPotential()
    with pytest.raises(ValueError, match="propensity"):
        potential.update_tables(
            np.eye(2, dtype=np.float32), np.array([1.2, 0.5]), np.array([0.5, 0.5])
        )
    with pytest.raises(ValueError, match="feasibility"):
        potential.update_tables(
            np.eye(2, dtype=np.float32), np.array([0.5, 0.5]), np.array([0.5, -0.1])
        )