"""MPE Spread harness tests: shapes, reward contract, termination, resume."""

import numpy as np
import pytest

from reap.envs import make_env
from reap.envs.mpe_spread import AGENT_SIZE, MpeSpreadEnv


def make(num_agents=3, horizon=25, **kw):
    return MpeSpreadEnv(num_agents=num_agents, horizon=horizon, **kw)


def test_factory_constructs_mpe_spread():
    env = make_env("mpe_spread", num_agents=3, horizon=25)
    assert isinstance(env, MpeSpreadEnv)


def test_reset_and_step_shapes():
    env = make()
    local, joint = env.reset(seed=0)
    assert len(local) == 3
    assert all(o.shape == (18,) for o in local)  # reference 18-dim observation
    assert joint.shape == (54,)
    result = env.step([0, 1, 4])
    assert len(result.local_obs) == 3
    assert result.joint_state.shape == (54,)
    assert isinstance(result.extrinsic_reward, float)


def test_observation_layout_matches_reference():
    env = make()
    env.reset(seed=3)
    local, _ = env._observe()
    # [vel(2), pos(2), landmark rel(6), other rel(4), comm zeros(4)]
    obs0 = local[0]
    assert np.allclose(obs0[0:2], env._vel[0])
    assert np.allclose(obs0[2:4], env._pos[0])
    assert np.allclose(obs0[4:6], env._landmarks[0] - env._pos[0])
    assert np.allclose(obs0[10:12], env._pos[1] - env._pos[0])
    assert np.allclose(obs0[14:18], 0.0)


def test_reward_is_summed_per_agent_with_self_collision():
    env = make()
    env.reset(seed=1)
    result = env.step([0, 0, 0])
    # team reward = sum over agents of (-sum min dists - collisions incl. self)
    sum_min = result.info["sum_min_dists"]
    pair = np.sqrt(np.sum(np.square(env._pos[:, None] - env._pos[None, :]), axis=-1))
    collision_count = int((pair < 2 * AGENT_SIZE).sum())  # diagonal = self terms
    assert collision_count >= 3  # each agent always collides with itself
    expected = -3 * sum_min - collision_count
    assert result.extrinsic_reward == pytest.approx(expected)


def test_horizon_truncates_and_never_terminates():
    env = make(horizon=5)
    env.reset(seed=0)
    for t in range(5):
        result = env.step([0, 0, 0])
        assert result.terminated is False
        assert result.truncated is (t == 4)


def test_seeded_resets_are_reproducible():
    a = make()
    b = make()
    la, ja = a.reset(seed=42)
    lb, jb = b.reset(seed=42)
    assert np.array_equal(ja, jb)
    ra = a.step([1, 2, 3])
    rb = b.step([1, 2, 3])
    assert np.array_equal(ra.joint_state, rb.joint_state)
    assert ra.extrinsic_reward == rb.extrinsic_reward


def test_different_seeds_differ():
    env = make()
    _, j0 = env.reset(seed=0)
    _, j1 = env.reset(seed=1)
    assert not np.array_equal(j0, j1)


def test_state_snapshot_roundtrip_mid_episode():
    env = make()
    env.reset(seed=7)
    for _ in range(3):
        env.step([1, 1, 1])
    snap = env.get_state()
    r_a = env.step([2, 3, 4])

    env2 = make()
    env2.reset(seed=999)  # unrelated state; snapshot must fully overwrite it
    env2.set_state(snap)
    r_b = env2.step([2, 3, 4])
    assert np.array_equal(r_a.joint_state, r_b.joint_state)
    assert r_a.extrinsic_reward == pytest.approx(r_b.extrinsic_reward)
    assert env2.steps_elapsed == env.steps_elapsed


def test_snapshot_is_a_deep_copy():
    env = make()
    env.reset(seed=0)
    snap = env.get_state()
    before = snap["pos"].copy()
    env.step([1, 1, 1])
    assert np.array_equal(snap["pos"], before)  # stepping must not mutate it


def test_action_validation():
    env = make()
    env.reset(seed=0)
    for bad in (-1, 5, 1.5, "up", None, True):
        with pytest.raises(ValueError):
            env.step([bad, 0, 0])
    with pytest.raises(ValueError):
        env.step([0, 0])  # wrong number of actions


def test_control_force_moves_agent():
    env = make()
    env.reset(seed=0)
    x0 = env._pos[0, 0]
    far = env._pos[0] + np.array([10.0, 10.0])  # isolate from contact forces
    env._pos[1] = far
    env._pos[2] = far + 1.0
    env.step([1, 0, 0])  # +x force on agent 0
    assert env._pos[0, 0] > x0
    # reference integration: vel = 0*(1-damping) + 5.0*0.1 = 0.5, pos += 0.05
    assert env._pos[0, 0] == pytest.approx(x0 + 0.05)
    assert env._vel[0, 0] == pytest.approx(0.5)


def test_success_requires_all_landmarks_covered():
    env = make()
    env.reset(seed=0)
    assert env.is_success() is False
    env._pos = env._landmarks.copy()  # park every agent on a landmark
    result = env.step([0, 0, 0])
    assert result.info["occupied_landmarks"] >= 2  # one step of drift allowed
    env._pos = env._landmarks.copy()
    env._vel[:] = 0.0
    env.step([0, 0, 0])
    assert env.is_success() is True


def test_num_agents_scales_dimensions():
    env = make(num_agents=4)
    local, joint = env.reset(seed=0)
    assert len(local) == 4
    assert env.local_obs_dim == 4 + 8 + 12  # vel+pos, 4 landmarks, 3 others
    assert joint.shape == (4 * env.local_obs_dim,)


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        MpeSpreadEnv(num_agents=1)
    with pytest.raises(ValueError):
        MpeSpreadEnv(horizon=0)
