"""PBRS shaping tests: exact semantics on the deterministic chain fixture."""

import numpy as np
import pytest

from reap.shaping import Potential, PotentialShaper, TabularPotential
from tests.chain_env import ChainEnv

BETA, GAMMA = 0.5, 0.9


class PositionPotential(Potential):
    """Phi = chain position (argmax of the one-hot state)."""

    def value(self, state, steps_remaining):
        return float(np.argmax(state))


def make_shaper(beta=BETA, gamma=GAMMA, learner_gamma=GAMMA):
    return PotentialShaper(beta=beta, gamma=gamma, learner_gamma=learner_gamma)


def one_hot(pos, length=5):
    v = np.zeros(length, dtype=np.float32)
    v[pos] = 1.0
    return v


def test_mid_episode_shaping_matches_hand_computation():
    shaper = make_shaper()
    f = shaper.shape(
        PositionPotential(),
        states=np.array([one_hot(1)]),
        next_states=np.array([one_hot(2)]),
        steps_remaining=np.array([7]),
        episode_ends=np.array([False]),
    )
    assert f[0] == pytest.approx(BETA * (GAMMA * 2.0 - 1.0))


def test_success_terminal_zeroes_next_potential():
    shaper = make_shaper()
    f = shaper.shape(
        PositionPotential(),
        states=np.array([one_hot(3)]),
        next_states=np.array([one_hot(4)]),  # reaching the goal cell
        steps_remaining=np.array([2]),
        episode_ends=np.array([True]),
    )
    assert f[0] == pytest.approx(BETA * (GAMMA * 0.0 - 3.0))  # Phi(terminal)=0


def test_timeout_truncation_zeroes_next_potential():
    shaper = make_shaper()
    f = shaper.shape(
        PositionPotential(),
        states=np.array([one_hot(2)]),
        next_states=np.array([one_hot(2)]),  # stayed; horizon expired
        steps_remaining=np.array([1]),
        episode_ends=np.array([True]),
    )
    assert f[0] == pytest.approx(-BETA * 2.0)


def test_full_episode_telescopes_to_initial_potential():
    """sum_t gamma^t F_t == -beta * Phi(s_0): shaping cancels in the return."""
    env = ChainEnv(length=5, horizon=12)
    potential = PositionPotential()
    shaper = make_shaper()
    env.reset()
    transitions = []
    state = one_hot(0)
    while True:
        remaining = env.steps_remaining
        result = env.step([1, 1])
        end = result.terminated or result.truncated
        transitions.append((state, result.joint_state, remaining, end))
        state = result.joint_state
        if end:
            break
    f = shaper.shape(
        potential,
        states=np.array([t[0] for t in transitions]),
        next_states=np.array([t[1] for t in transitions]),
        steps_remaining=np.array([t[2] for t in transitions]),
        episode_ends=np.array([t[3] for t in transitions]),
    )
    discounted = sum(GAMMA**t * f[t] for t in range(len(f)))
    assert discounted == pytest.approx(-BETA * potential.value(one_hot(0), 12))


def test_discount_mismatch_rejected():
    with pytest.raises(ValueError, match="must equal the learner's discount"):
        make_shaper(gamma=0.9, learner_gamma=0.99)


@pytest.mark.parametrize("beta", [-0.1, -5.0])
def test_negative_beta_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        make_shaper(beta=beta)


@pytest.mark.parametrize("gamma", [0.0, 1.5])
def test_invalid_gamma_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        make_shaper(gamma=gamma, learner_gamma=gamma)


def test_batch_length_mismatch_rejected():
    shaper = make_shaper()
    with pytest.raises(ValueError, match="equal length"):
        shaper.shape(
            PositionPotential(),
            states=np.array([one_hot(0), one_hot(1)]),
            next_states=np.array([one_hot(1)]),
            steps_remaining=np.array([3, 2]),
            episode_ends=np.array([False, False]),
        )


def test_nonpositive_steps_remaining_rejected():
    shaper = make_shaper()
    with pytest.raises(ValueError, match="steps_remaining"):
        shaper.shape(
            PositionPotential(),
            states=np.array([one_hot(0)]),
            next_states=np.array([one_hot(1)]),
            steps_remaining=np.array([0]),
            episode_ends=np.array([False]),
        )


def test_tabular_potential_lookup_and_default():
    pot = TabularPotential(key_fn=lambda s: int(np.argmax(s)), table={0: 0.5, 2: 1.5})
    assert pot.value(one_hot(0), 5) == 0.5
    assert pot.value(one_hot(2), 5) == 1.5
    assert pot.value(one_hot(3), 5) == 0.0  # missing key defaults to zero


def test_zero_beta_disables_shaping():
    shaper = make_shaper(beta=0.0)
    f = shaper.shape(
        PositionPotential(),
        states=np.array([one_hot(1)]),
        next_states=np.array([one_hot(2)]),
        steps_remaining=np.array([4]),
        episode_ends=np.array([False]),
    )
    assert f[0] == 0.0
