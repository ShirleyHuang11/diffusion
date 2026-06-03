"""MAPPO trunk tests: smoke training, determinism, resume fidelity, channel separation."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("overcooked_ai_py")

from reap.config import (
    AlgoConfig,
    CheckpointConfig,
    Config,
    EnvConfig,
    LoggingConfig,
    RunConfig,
)
from reap.metrics import deterministic_view, read_jsonl
from reap.train import run_from_config

pytestmark = pytest.mark.overcooked

TINY_PARAMS = {
    "rollout_length": 64,
    "hidden_size": 32,
    "update_epochs": 2,
    "num_minibatches": 2,
}


def make_cfg(tmp_path, name="m", seed=0, total_steps=192, params=None, log_every=64,
             ckpt_every=192):
    return Config(
        run=RunConfig(
            name=name, seed=seed, mode="smoke", out_dir=str(tmp_path / "runs"),
            max_wall_clock_minutes=30.0, device="cpu",
        ),
        env=EnvConfig(id="overcooked", layout="cramped_room", horizon=50, encoding="lossless"),
        algo=AlgoConfig(name="mappo", total_env_steps=total_steps,
                        params={**TINY_PARAMS, **(params or {})}),
        logging=LoggingConfig(interval_env_steps=log_every),
        checkpoint=CheckpointConfig(interval_env_steps=ckpt_every, keep_last=2),
    )


def run_dir(cfg):
    from pathlib import Path

    return Path(cfg.run.out_dir) / cfg.run.name / f"seed{cfg.run.seed}"


def test_mappo_smoke_trains_without_nans(tmp_path):
    cfg = make_cfg(tmp_path)
    summary = run_from_config(cfg)
    assert summary["env_step"] == 192
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    assert records, "no metrics logged"
    for rec in records:
        for channel in ("extrinsic", "intrinsic", "diag"):
            for key, value in rec[channel].items():
                assert np.isfinite(value), f"{channel}/{key} is not finite"
        assert {"episode_return_mean", "episodes", "success_rate"} <= set(rec["extrinsic"])


def test_mappo_same_seed_determinism(tmp_path):
    cfg_a = make_cfg(tmp_path, name="a")
    cfg_b = make_cfg(tmp_path, name="b")
    run_from_config(cfg_a)
    run_from_config(cfg_b)
    rec_a = deterministic_view(read_jsonl(run_dir(cfg_a) / "metrics.jsonl"))
    rec_b = deterministic_view(read_jsonl(run_dir(cfg_b) / "metrics.jsonl"))
    assert rec_a == rec_b


def test_mappo_resume_is_trajectory_faithful(tmp_path):
    """Uninterrupted vs interrupted+resumed runs match exactly. The checkpoint
    interval (100) is not aligned to rollout (64) or horizon (50) boundaries,
    so the save lands mid-episode at the first update boundary past 100."""
    cfg_full = make_cfg(tmp_path, name="full", total_steps=256, ckpt_every=100)
    run_from_config(cfg_full)

    cfg_part = make_cfg(tmp_path, name="part", total_steps=128, ckpt_every=100)
    run_from_config(cfg_part)
    cfg_resumed = dataclasses.replace(
        cfg_part,
        algo=dataclasses.replace(cfg_part.algo, total_env_steps=256),
    )
    run_from_config(cfg_resumed, resume=True)

    rec_full = deterministic_view(read_jsonl(run_dir(cfg_full) / "metrics.jsonl"))
    rec_part = deterministic_view(read_jsonl(run_dir(cfg_part) / "metrics.jsonl"))
    assert rec_part == rec_full


def test_mappo_intrinsic_never_contaminates_extrinsic(tmp_path):
    """With a large RND bonus active, the logged extrinsic channel still
    reports only task reward (zero for an untrained policy at this scale),
    while the intrinsic channel shows the bonus flowing."""
    cfg = make_cfg(
        tmp_path,
        params={"intrinsic": "rnd", "intrinsic_coef": 10.0},
    )
    run_from_config(cfg)
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    saw_bonus = False
    for rec in records:
        # sparse task reward: returns are multiples of the delivery reward (20);
        # an untrained policy on a 50-step horizon delivers nothing
        assert rec["extrinsic"]["episode_return_mean"] == 0.0
        if rec["intrinsic"]["bonus_mean"] > 0:
            saw_bonus = True
    assert saw_bonus, "intrinsic bonus never appeared in its own channel"


def test_mappo_count_bonus_runs(tmp_path):
    cfg = make_cfg(tmp_path, params={"intrinsic": "count", "intrinsic_coef": 0.1})
    summary = run_from_config(cfg)
    assert summary["env_step"] == 192
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    assert any(r["intrinsic"].get("count_table_size", 0) > 0 for r in records)


def test_mappo_unknown_param_rejected(tmp_path):
    from reap.algos.mappo import MappoTrainer
    from reap.envs import make_env

    env = make_env("overcooked", layout="cramped_room", horizon=20, encoding="lossless")
    with pytest.raises(ValueError, match="unknown mappo params"):
        MappoTrainer(env, {"learning_rate_typo": 1e-3}, seed=0)
