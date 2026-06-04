"""Hybrid teacher tests: paired collection, projector, policy-conditioned refresh."""

import numpy as np
import pytest

pytest.importorskip("overcooked_ai_py")

from reap.hybrid_teacher import (
    PairedOvercooked,
    PolicyConditionedRefresher,
    SimulatorRolloutProjector,
)
from reap.teacher_pipeline import DELIVERY_SCALE

pytestmark = pytest.mark.overcooked


@pytest.fixture(scope="module")
def paired():
    return PairedOvercooked(layout="cramped_room", horizon=60)


def rollout_pair(paired, steps=12, seed=0):
    rng = np.random.default_rng(seed)
    local, lj, fj = paired.reset()
    l_states, f_states = [lj], [fj]
    for _ in range(steps):
        result, lj, fj = paired.step(rng.integers(0, paired.num_actions, size=2).tolist())
        l_states.append(lj)
        f_states.append(fj)
        if result.terminated or result.truncated:
            break
    return np.stack(l_states), np.stack(f_states)


def test_paired_collection_synchronized(paired):
    l_states, f_states = rollout_pair(paired)
    assert len(l_states) == len(f_states)
    assert l_states.shape[1] == paired.lossless_dim
    assert f_states.shape[1] == paired.feature_dim
    # delivery feature identical across the two encodings at every step
    assert np.allclose(l_states[:, -1], f_states[:, -1])


def test_projector_reproduces_real_window(paired):
    """Projecting the TRUE feature window from its true start recovers the
    rollout: near-zero projection error and the exact delivery count."""
    l_states, f_states = rollout_pair(paired, steps=8, seed=1)
    projector = SimulatorRolloutProjector(paired)
    out = projector.project(l_states[0], f_states)
    assert out is not None
    assert out["projection_errors"].max() < 1e-4  # true successors are exact
    true_deliveries = round((l_states[-1][-1] - l_states[0][-1]) / DELIVERY_SCALE)
    assert out["deliveries"] == true_deliveries
    assert np.allclose(out["lossless_window"][0], l_states[0])


def test_projector_output_always_exactly_valid(paired):
    """Even from a NOISY feature window, the projected lossless window is
    valid and dynamics-consistent by construction."""
    from reap.teacher_pipeline import (
        OvercookedExactValidator,
        OvercookedSimulatorChecker,
    )

    l_states, f_states = rollout_pair(paired, steps=6, seed=2)
    noisy = f_states + np.random.default_rng(3).normal(0, 2.0, size=f_states.shape)
    noisy = noisy.astype(np.float32)
    projector = SimulatorRolloutProjector(paired)
    out = projector.project(l_states[0], noisy)
    assert out is not None
    window = out["lossless_window"]

    validator = OvercookedExactValidator(
        np.concatenate([l_states, window]), grid_shape=paired.base.lossless_shape,
        mdp=paired.base._mdp,
    )
    assert validator.is_valid(window).all()
    checker = OvercookedSimulatorChecker(paired.base._mdp, paired.base.lossless_shape)
    for t in range(len(window) - 1):
        assert checker.realizable(window[t], window[t + 1])


def test_projector_rejects_invalid_start(paired):
    l_states, f_states = rollout_pair(paired, steps=4, seed=4)
    corrupt = l_states[0].copy()
    corrupt[0] = 7.0  # impossible channel value: start no longer decodable
    projector = SimulatorRolloutProjector(paired)
    assert projector.project(corrupt, f_states) is None


def test_policy_conditioned_refresher_uses_current_policy():
    """The refresher embeds the CURRENT nets, queries the (stub) teacher with
    that embedding, refits p-hat on the returned targets, and the embedding
    changes when the policy changes."""
    from reap.algos.mappo import MappoTrainer
    from reap.signals import BehavioralPolicyEmbedding
    from reap.signals.distill import DistilledPredictor
    from tests.chain_env import ChainEnv

    rng = np.random.default_rng(0)
    probes = rng.normal(size=(5, 2, 4)).astype(np.float32)
    embedding = BehavioralPolicyEmbedding(probes)
    trainer_a = MappoTrainer(ChainEnv(4, 8), {"hidden_size": 16, "rollout_length": 8}, seed=0)
    trainer_b = MappoTrainer(ChainEnv(4, 8), {"hidden_size": 16, "rollout_length": 8}, seed=9)
    current = {"nets": trainer_a.nets}

    n_anchors, window, fdim = 6, 4, 5
    lossless_anchors = rng.normal(size=(n_anchors, 4)).astype(np.float32)
    feature_anchors = rng.normal(size=(n_anchors, fdim)).astype(np.float32)
    seen_embeddings = []

    def stub_sampler(embedding_vec, anchors_f):
        seen_embeddings.append(embedding_vec.copy())
        # teacher stub: delivery delta keyed on the embedding sum so different
        # policies yield different targets
        m = 4
        windows = np.zeros((len(anchors_f), m, window, fdim), dtype=np.float32)
        bump = 1.0 if embedding_vec.sum() > seen_embeddings[0].sum() - 1e-9 else 0.0
        windows[:, : int(2 + bump), -1, -1] = DELIVERY_SCALE  # deliveries in some samples
        return windows

    p_hat = DistilledPredictor(4, hidden=16, seed=0)
    f_hat = DistilledPredictor(4, hidden=16, seed=1)
    refresher = PolicyConditionedRefresher(
        nets_provider=lambda: current["nets"],
        embedding=embedding,
        sampler=stub_sampler,
        lossless_anchors=lossless_anchors,
        feature_anchors=feature_anchors,
        feasibility=np.ones(n_anchors),
        p_hat=p_hat,
        f_hat=f_hat,
        refit_epochs=30,
    )

    states, prop, feas = refresher()
    assert states.shape == (n_anchors, 4)
    assert np.all((prop >= 0) & (prop <= 1)) and np.any(prop > 0)
    assert np.allclose(feas, 1.0)
    emb_first = refresher.last_embedding.copy()
    # p-hat actually refit toward the returned targets
    assert np.abs(p_hat.predict(lossless_anchors) - prop).mean() < 0.25

    current["nets"] = trainer_b.nets  # policy changed -> embedding must change
    refresher()
    assert not np.allclose(refresher.last_embedding, emb_first)
    assert len(seen_embeddings) == 2