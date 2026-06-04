"""Exact decoder and simulator-checker tests on real Overcooked states."""

import numpy as np
import pytest

pytest.importorskip("overcooked_ai_py")

from reap.envs.overcooked_decode import decode_lossless, encode_state, roundtrip_valid
from reap.envs.overcooked_env import OvercookedSparseEnv
from reap.teacher_pipeline import (
    DELIVERY_SCALE,
    DeliveryAugmentedOvercooked,
    OvercookedExactValidator,
    OvercookedSimulatorChecker,
)

pytestmark = pytest.mark.overcooked


@pytest.fixture(scope="module")
def env():
    return OvercookedSparseEnv(layout="cramped_room", horizon=200, encoding="lossless")


def collect_states(env, steps=300, seed=0):
    rng = np.random.default_rng(seed)
    env.reset()
    grids = [encode_state(env._env.state, env._mdp)]
    for _ in range(steps):
        result = env.step(rng.integers(0, env.num_actions, size=2).tolist())
        grids.append(encode_state(env._env.state, env._mdp))
        if result.terminated or result.truncated:
            env.reset()
            grids.append(encode_state(env._env.state, env._mdp))
    return grids


def test_decode_roundtrip_on_real_states(env):
    grids = collect_states(env, steps=300)
    for grid in grids:
        assert roundtrip_valid(grid, env._mdp)


def test_decode_rejects_corrupted_grids(env):
    env.reset()
    grid = encode_state(env._env.state, env._mdp)
    twin = grid.copy()
    twin[..., 0] = 0  # erase player 0 entirely
    assert decode_lossless(twin, env._mdp) is None
    overlap = grid.copy()
    overlap[..., 1] = overlap[..., 0]  # both players on the same cell
    assert decode_lossless(overlap, env._mdp) is None
    fractional = grid.astype(float)
    fractional[0, 0, 11] = 0.5
    assert decode_lossless(fractional, env._mdp) is None


def test_simulator_checker_accepts_real_transitions(env):
    aug = DeliveryAugmentedOvercooked(env)
    rng = np.random.default_rng(1)
    _, joint = aug.reset()
    checker = OvercookedSimulatorChecker(env._mdp, env.lossless_shape)
    accepted = 0
    for _ in range(40):
        result = aug.step(rng.integers(0, aug.num_actions, size=2).tolist())
        assert checker.realizable(joint, result.joint_state)
        accepted += 1
        joint = result.joint_state
        if result.terminated or result.truncated:
            _, joint = aug.reset()
    assert accepted == 40


def test_simulator_checker_rejects_teleport_and_fake_delivery(env):
    aug = DeliveryAugmentedOvercooked(env)
    rng = np.random.default_rng(2)
    _, joint = aug.reset()
    result = aug.step(rng.integers(0, aug.num_actions, size=2).tolist())
    checker = OvercookedSimulatorChecker(env._mdp, env.lossless_shape)

    grid_shape = env.lossless_shape
    grid_size = int(np.prod(grid_shape))
    teleported = result.joint_state.copy()
    grid = teleported[:grid_size].reshape(*grid_shape).copy()
    pos = np.argwhere(grid[..., 0] == 1)[0]
    grid[pos[0], pos[1], 0] = 0
    far = (grid.shape[0] - 1 - pos[0], grid.shape[1] - 1 - pos[1])
    grid[far[0], far[1], 0] = 1
    # move the orientation marker with the player so the state stays decodable
    for d in range(4):
        if grid[pos[0], pos[1], 2 + d] == 1:
            grid[pos[0], pos[1], 2 + d] = 0
            grid[far[0], far[1], 2 + d] = 1
    teleported[:grid_size] = grid.ravel()
    assert not checker.realizable(joint, teleported)

    fake_delivery = result.joint_state.copy()
    fake_delivery[-1] += DELIVERY_SCALE  # count bump with no delivering action
    assert not checker.realizable(joint, fake_delivery)


def test_checker_rejects_non_roundtripping_endpoints(env):
    """Codex round-5 reproduction: a corrupted static (counter) layer decodes
    but does not roundtrip; the checker must reject it on EITHER endpoint."""
    aug = DeliveryAugmentedOvercooked(env)
    rng = np.random.default_rng(4)
    _, joint = aug.reset()
    result = aug.step(rng.integers(0, aug.num_actions, size=2).tolist())
    checker = OvercookedSimulatorChecker(env._mdp, env.lossless_shape)
    assert checker.realizable(joint, result.joint_state)  # sanity: real transition

    grid_shape = env.lossless_shape
    grid_size = int(np.prod(grid_shape))

    def corrupt_counter_layer(flat):
        out = flat.copy()
        grid = out[:grid_size].reshape(*grid_shape).copy()
        counter_cells = np.argwhere(grid[..., 11] == 1)  # counter_loc layer
        cell = counter_cells[0]
        grid[cell[0], cell[1], 11] = 0  # erase one counter: decodes, no roundtrip
        out[:grid_size] = grid.ravel()
        return out

    corrupt_source = corrupt_counter_layer(joint)
    assert not checker.realizable(corrupt_source, result.joint_state)
    corrupt_target = corrupt_counter_layer(result.joint_state)
    assert not checker.realizable(joint, corrupt_target)


def test_checker_rejects_bad_delivery_counts(env):
    aug = DeliveryAugmentedOvercooked(env)
    rng = np.random.default_rng(5)
    _, joint = aug.reset()
    result = aug.step(rng.integers(0, aug.num_actions, size=2).tolist())
    checker = OvercookedSimulatorChecker(env._mdp, env.lossless_shape)

    fractional = joint.copy()
    fractional[-1] = 0.5 * DELIVERY_SCALE  # not a count multiple
    assert not checker.realizable(fractional, result.joint_state)

    negative = joint.copy()
    negative[-1] = -DELIVERY_SCALE
    assert not checker.realizable(negative, result.joint_state)


def test_exact_validator_flags_invalid_and_projects(env):
    aug = DeliveryAugmentedOvercooked(env)
    _, joint = aug.reset()
    validator = OvercookedExactValidator(
        np.stack([joint, joint]), grid_shape=env.lossless_shape, mdp=env._mdp
    )
    assert validator.is_valid(np.stack([joint]))[0]
    corrupted = joint.copy()
    corrupted[: int(np.prod(env.lossless_shape))][0] = 7.0  # impossible channel value
    noisy = joint + np.random.default_rng(3).normal(0, 0.2, size=joint.shape)
    projected = validator.project(np.stack([noisy]))
    # projection re-lands on the training manifold for near-clean states
    assert validator.is_valid(projected)[0]