"""Overcooked sparse-wrapper tests: shapes, actions, reward sparsity, success."""

import numpy as np
import pytest

pytest.importorskip("overcooked_ai_py")

from reap.envs import make_env
from reap.envs.overcooked_env import LAYOUTS, OvercookedSparseEnv

ALL_LAYOUTS = sorted(LAYOUTS)

pytestmark = pytest.mark.overcooked


def stay_index():
    from overcooked_ai_py.mdp.actions import Action

    return Action.ALL_ACTIONS.index(Action.STAY)


@pytest.fixture(scope="module", params=ALL_LAYOUTS)
def layout_env(request):
    return OvercookedSparseEnv(layout=request.param, horizon=30, encoding="lossless")


def test_observation_and_state_shapes(layout_env):
    local, joint = layout_env.reset()
    assert len(local) == layout_env.num_agents == 2
    for obs in local:
        assert obs.shape == (layout_env.local_obs_dim,)
        assert obs.dtype == np.float32
    assert joint.shape == (layout_env.joint_state_dim,)


def test_action_space_and_mapping(layout_env):
    from overcooked_ai_py.mdp.actions import Action

    assert layout_env.num_actions == len(Action.ALL_ACTIONS) == 6
    layout_env.reset()
    result = layout_env.step([stay_index()] * 2)  # stay actions always legal
    assert isinstance(result.extrinsic_reward, float)


def test_wrong_action_count_rejected(layout_env):
    layout_env.reset()
    with pytest.raises(ValueError, match="expected 2 actions"):
        layout_env.step([0])


def test_termination_at_horizon_without_success(layout_env):
    layout_env.reset()
    result = None
    for _ in range(layout_env.horizon):
        result = layout_env.step([stay_index()] * 2)
    assert result.truncated  # horizon end is truncation
    assert not result.terminated
    # negative test: timeout without delivery must NOT register success
    assert layout_env.is_success() is False
    assert result.info["success"] is False
    assert layout_env.steps_remaining == 0


def test_stay_rollout_gives_zero_extrinsic_reward(layout_env):
    layout_env.reset()
    total = 0.0
    for _ in range(layout_env.horizon):
        total += layout_env.step([stay_index()] * 2).extrinsic_reward
    assert total == 0.0


def test_unknown_layout_rejected():
    with pytest.raises(ValueError, match="unknown layout"):
        OvercookedSparseEnv(layout="secret_kitchen")


def test_unknown_encoding_rejected():
    with pytest.raises(ValueError, match="unknown encoding"):
        OvercookedSparseEnv(layout="cramped_room", encoding="hieroglyphs")


def test_random_rollout_extrinsic_only_at_deliveries():
    """Extrinsic reward equals the native sparse (delivery) channel exactly,
    and the native dense/shaped channel never leaks into it."""
    env = OvercookedSparseEnv(layout="cramped_room", horizon=400, encoding="lossless")
    rng = np.random.default_rng(7)
    env.reset()
    shaped_events = 0
    for _ in range(1200):
        result = env.step(rng.integers(0, env.num_actions, size=2).tolist())
        info = result.info
        assert result.extrinsic_reward == info["debug_native_sparse_reward"]
        if info["debug_native_shaped_reward"] != 0.0:
            shaped_events += 1
            # dense reward fired but extrinsic only carries the sparse channel
            assert result.extrinsic_reward == info["debug_native_sparse_reward"]
        if result.extrinsic_reward != 0.0:
            assert info["deliveries"] >= 1  # nonzero extrinsic only at delivery
        if result.terminated or result.truncated:
            env.reset()
    # random play on cramped_room reliably triggers dense events (onion pickups),
    # which proves the leak test is exercising a real difference
    assert shaped_events > 0


def test_features_encoding_cramped_room():
    """Compact feature encoding: per-agent vectors, joint = concatenation."""
    env = OvercookedSparseEnv(layout="cramped_room", horizon=20, encoding="features")
    local, joint = env.reset()
    assert len(local) == 2
    assert local[0].shape == local[1].shape
    assert joint.shape == (2 * env.local_obs_dim,)
    result = env.step([stay_index()] * 2)
    assert result.joint_state.shape == (env.joint_state_dim,)


def test_make_env_dispatch():
    env = make_env("overcooked", layout="cramped_room", horizon=10, encoding="lossless")
    assert isinstance(env, OvercookedSparseEnv)
    with pytest.raises(ValueError, match="unknown env id"):
        make_env("chess")
