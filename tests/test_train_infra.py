"""End-to-end infrastructure tests: determinism, resume, wall-clock cap."""

import dataclasses

import pytest

pytest.importorskip("overcooked_ai_py")

from reap.checkpoint import CheckpointError, latest_checkpoint
from reap.config import (
    AlgoConfig,
    CheckpointConfig,
    Config,
    EnvConfig,
    LoggingConfig,
    RunConfig,
)
from reap.metrics import deterministic_view, read_jsonl
from reap.train import WallClockExceeded, run_from_config

pytestmark = pytest.mark.overcooked


def make_cfg(tmp_path, name="t", seed=0, total_steps=300, minutes=10.0):
    return Config(
        run=RunConfig(
            name=name,
            seed=seed,
            mode="smoke",
            out_dir=str(tmp_path / "runs"),
            max_wall_clock_minutes=minutes,
            device="cpu",
        ),
        env=EnvConfig(id="overcooked", layout="cramped_room", horizon=50, encoding="lossless"),
        algo=AlgoConfig(name="random", total_env_steps=total_steps),
        logging=LoggingConfig(interval_env_steps=100),
        checkpoint=CheckpointConfig(interval_env_steps=100, keep_last=2),
    )


def run_dir(cfg):
    from pathlib import Path

    return Path(cfg.run.out_dir) / cfg.run.name / f"seed{cfg.run.seed}"


def test_same_seed_runs_produce_identical_metrics(tmp_path):
    cfg_a = make_cfg(tmp_path, name="a")
    cfg_b = make_cfg(tmp_path, name="b")
    run_from_config(cfg_a)
    run_from_config(cfg_b)
    rec_a = deterministic_view(read_jsonl(run_dir(cfg_a) / "metrics.jsonl"))
    rec_b = deterministic_view(read_jsonl(run_dir(cfg_b) / "metrics.jsonl"))
    assert rec_a == rec_b
    assert len(rec_a) == 3  # 300 steps / 100-step interval


def test_different_seed_runs_differ(tmp_path):
    cfg_a = make_cfg(tmp_path, name="a", seed=0)
    cfg_b = make_cfg(tmp_path, name="b", seed=1)
    run_from_config(cfg_a)
    run_from_config(cfg_b)
    rec_a = deterministic_view(read_jsonl(run_dir(cfg_a) / "metrics.jsonl"))
    rec_b = deterministic_view(read_jsonl(run_dir(cfg_b) / "metrics.jsonl"))
    # success stats can coincide at tiny scale; rng state in checkpoints must not
    from reap.checkpoint import load_checkpoint

    state_a = load_checkpoint(latest_checkpoint(run_dir(cfg_a) / "checkpoints"))
    state_b = load_checkpoint(latest_checkpoint(run_dir(cfg_b) / "checkpoints"))
    assert state_a["rng_state"] != state_b["rng_state"]
    assert len(rec_a) == len(rec_b)


def test_resume_continues_run(tmp_path):
    cfg = make_cfg(tmp_path, total_steps=200)
    run_from_config(cfg)
    assert latest_checkpoint(run_dir(cfg) / "checkpoints") is not None

    cfg_more = dataclasses.replace(cfg, algo=AlgoConfig(name="random", total_env_steps=400))
    summary = run_from_config(cfg_more, resume=True)
    assert summary["env_step"] == 400
    records = read_jsonl(run_dir(cfg) / "metrics.jsonl")
    steps = [r["env_step"] for r in records]
    assert steps == sorted(steps)
    assert steps[-1] == 400
    # schema consistent across the resume boundary
    keys = {tuple(sorted(r)) for r in records}
    assert len(keys) == 1


def test_resume_is_trajectory_faithful_mid_episode(tmp_path):
    """Interrupted+resumed runs must reproduce the uninterrupted metric stream
    exactly, even when the checkpoint lands mid-episode (intervals deliberately
    not aligned to the episode horizon)."""
    horizon = 30  # episodes end at 30, 60, 90, 120, 150
    base = dict(total_steps=150, minutes=10.0)

    cfg_full = make_cfg(tmp_path, name="full", **base)
    cfg_full = dataclasses.replace(
        cfg_full,
        env=dataclasses.replace(cfg_full.env, horizon=horizon),
        logging=LoggingConfig(interval_env_steps=50),
        checkpoint=CheckpointConfig(interval_env_steps=75, keep_last=2),  # mid-episode
    )
    run_from_config(cfg_full)

    cfg_part = dataclasses.replace(
        cfg_full,
        run=dataclasses.replace(cfg_full.run, name="part"),
        algo=AlgoConfig(name="random", total_env_steps=75),
    )
    run_from_config(cfg_part)  # stops at step 75: episode 3 in progress
    cfg_resumed = dataclasses.replace(
        cfg_part, algo=AlgoConfig(name="random", total_env_steps=150)
    )
    run_from_config(cfg_resumed, resume=True)

    rec_full = deterministic_view(read_jsonl(run_dir(cfg_full) / "metrics.jsonl"))
    rec_part = deterministic_view(read_jsonl(run_dir(cfg_part) / "metrics.jsonl"))
    assert rec_part == rec_full  # identical streams incl. episode counters

    from reap.checkpoint import load_checkpoint

    final_full = load_checkpoint(latest_checkpoint(run_dir(cfg_full) / "checkpoints"))
    final_part = load_checkpoint(latest_checkpoint(run_dir(cfg_part) / "checkpoints"))
    assert final_full["episodes"] == final_part["episodes"] == 5
    assert final_full["successes"] == final_part["successes"]
    assert final_full["return_sum"] == final_part["return_sum"]
    assert final_full["rng_state"] == final_part["rng_state"]


def test_resume_without_checkpoint_fails(tmp_path):
    cfg = make_cfg(tmp_path)
    from reap.config import ConfigError

    with pytest.raises(ConfigError, match="no checkpoint found"):
        run_from_config(cfg, resume=True)


def test_resume_from_corrupted_checkpoint_fails_loudly(tmp_path):
    cfg = make_cfg(tmp_path, total_steps=200)
    run_from_config(cfg)
    ckpt = latest_checkpoint(run_dir(cfg) / "checkpoints")
    raw = bytearray(ckpt.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ckpt.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError, match="integrity failure"):
        run_from_config(cfg, resume=True)


def test_wall_clock_cap_enforced(tmp_path):
    cfg = make_cfg(tmp_path, total_steps=1_000_000, minutes=1e-9)
    with pytest.raises(WallClockExceeded):
        run_from_config(cfg)
