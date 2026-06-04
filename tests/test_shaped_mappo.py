"""Shaped-MAPPO integration tests: shaping flows into training only."""

import numpy as np
import pytest

from reap.algos.mappo import MappoTrainer
from reap.metrics import deterministic_view, read_jsonl
from tests.chain_env import ChainEnv

SHAPED_PARAMS = {
    "rollout_length": 32,
    "hidden_size": 16,
    "update_epochs": 2,
    "num_minibatches": 2,
    "shaping_potential": "joint_argmax_position",
    "shaping_beta": 0.5,
}


def test_shaping_terms_computed_on_chain():
    trainer = MappoTrainer(ChainEnv(length=5, horizon=8), SHAPED_PARAMS, seed=0)
    rollout = trainer.collect_rollout()
    assert rollout["shaping"].shape == (32,)
    assert np.any(rollout["shaping"] != 0.0)  # position changes produce terms
    # extrinsic rewards remain pure task rewards: only 0 or the success reward
    assert set(np.unique(rollout["extrinsic"])) <= {0.0, 1.0}
    diags = trainer.update(rollout)
    assert np.isfinite(diags["policy_loss"])


def test_shaping_zeroes_next_potential_at_episode_end():
    trainer = MappoTrainer(ChainEnv(length=3, horizon=4), {
        **SHAPED_PARAMS, "rollout_length": 16,
    }, seed=1)
    rollout = trainer.collect_rollout()
    gamma, beta = trainer.gamma, 0.5
    ends = np.where(rollout["dones"] == 1.0)[0]
    assert len(ends) > 0
    for t in ends:
        phi_s = float(np.argmax(rollout["joint_states"][t]))
        # at episode end the next-potential term is zero by construction
        assert rollout["shaping"][t] == pytest.approx(beta * (gamma * 0.0 - phi_s))


def test_shaping_requires_potential_name():
    with pytest.raises(ValueError, match="no shaping_potential"):
        MappoTrainer(ChainEnv(), {"shaping_beta": 1.0}, seed=0)


def test_unknown_potential_rejected():
    with pytest.raises(ValueError, match="unknown shaping potential"):
        MappoTrainer(ChainEnv(), {"shaping_potential": "magic", "shaping_beta": 1.0}, seed=0)


def test_zero_beta_means_zero_terms():
    trainer = MappoTrainer(ChainEnv(length=5, horizon=8), {
        **SHAPED_PARAMS, "shaping_beta": 0.0,
    }, seed=0)
    rollout = trainer.collect_rollout()
    assert np.all(rollout["shaping"] == 0.0)


@pytest.mark.overcooked
def test_overcooked_progress_potential_bounded():
    pytest.importorskip("overcooked_ai_py")
    from reap.envs.overcooked_env import OvercookedSparseEnv

    env = OvercookedSparseEnv(layout="cramped_room", horizon=200, encoding="lossless")
    env.reset()
    assert env.progress_potential() == 0.0  # nothing in progress at the start
    rng = np.random.default_rng(0)
    values = []
    for _ in range(300):
        result = env.step(rng.integers(0, env.num_actions, size=2).tolist())
        values.append(env.progress_potential())
        if result.terminated or result.truncated:
            env.reset()
    values = np.array(values)
    assert np.all((values >= 0.0) & (values <= 1.6))
    assert np.any(values > 0.0)  # random play picks things up occasionally


@pytest.mark.overcooked
def test_shaped_overcooked_run_keeps_extrinsic_pure(tmp_path):
    pytest.importorskip("overcooked_ai_py")
    import dataclasses

    from reap.config import AlgoConfig, CheckpointConfig, Config, EnvConfig, LoggingConfig, RunConfig
    from reap.train import run_from_config

    cfg = Config(
        run=RunConfig(name="sh", seed=0, mode="smoke", out_dir=str(tmp_path / "runs"),
                      max_wall_clock_minutes=20.0, device="cpu"),
        env=EnvConfig(id="overcooked", layout="cramped_room", horizon=50, encoding="lossless"),
        algo=AlgoConfig(name="mappo", total_env_steps=192, params={
            "rollout_length": 64, "hidden_size": 32, "update_epochs": 2,
            "num_minibatches": 2, "shaping_potential": "overcooked_progress",
            "shaping_beta": 5.0,
        }),
        logging=LoggingConfig(interval_env_steps=64),
        checkpoint=CheckpointConfig(interval_env_steps=192, keep_last=1),
    )
    run_from_config(cfg)
    records = read_jsonl(tmp_path / "runs" / "sh" / "seed0" / "metrics.jsonl")
    assert records
    for rec in records:
        # shaped channel carries the shaping terms; extrinsic stays task-only
        assert "term_mean" in rec["shaped"]
        assert rec["extrinsic"]["episode_return_mean"] == 0.0  # untrained, 50-step horizon