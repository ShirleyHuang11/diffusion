"""Warmup buffer and ladder-collection tests on the chain fixture."""

import numpy as np
import pytest

from reap.data import TrajectoryBuffer, WarmupGateError, collect_warmup
from tests.chain_env import ChainEnv


def always_right(local_obs, joint_state):
    return [1, 1]  # solves the chain deterministically


def always_stay(local_obs, joint_state):
    return [0, 0]  # never succeeds


def test_buffer_add_and_report():
    buffer = TrajectoryBuffer(state_dim=5)
    states = np.zeros((4, 5), dtype=np.float32)
    buffer.add_episode(states, ret=1.0, success=True, source="test")
    buffer.add_episode(states, ret=0.0, success=False, source="test")
    report = buffer.report()
    assert report["episodes"] == 2
    assert report["success_count"] == 1
    assert report["total_env_steps"] == 6
    assert len(report["success_state_examples"]) == 1
    assert report["per_source"]["test"] == {"episodes": 2, "successes": 1}


def test_buffer_rejects_bad_shapes():
    buffer = TrajectoryBuffer(state_dim=5)
    with pytest.raises(ValueError, match="shape"):
        buffer.add_episode(np.zeros((4, 3), dtype=np.float32), 0.0, False)
    with pytest.raises(ValueError, match="at least one transition"):
        buffer.add_episode(np.zeros((1, 5), dtype=np.float32), 0.0, False)


def test_buffer_save_load_roundtrip(tmp_path):
    buffer = TrajectoryBuffer(state_dim=3)
    rng = np.random.default_rng(0)
    for i in range(4):
        buffer.add_episode(
            rng.normal(size=(5 + i, 3)).astype(np.float32),
            ret=float(i), success=i % 2 == 0, source=f"rung{i % 2}",
        )
    path = buffer.save(tmp_path / "buf.npz")
    loaded = TrajectoryBuffer.load(path)
    assert len(loaded) == 4
    assert loaded.success_count == buffer.success_count
    assert loaded.sources == buffer.sources
    for a, b in zip(buffer.episodes, loaded.episodes):
        assert np.allclose(a, b)


def test_warmup_gate_met_on_first_rung(tmp_path):
    env = ChainEnv(length=4, horizon=10)
    buffer, report = collect_warmup(
        env,
        ladder=[("solver", always_right), ("fallback", always_stay)],
        min_successes=5,
        max_env_steps=1000,
        report_path=tmp_path / "warmup.json",
    )
    assert report["gate"]["met"] is True
    assert buffer.success_count >= 5
    # the gate was met on rung one; the fallback never collected
    assert "fallback" not in report["per_source"]
    assert (tmp_path / "warmup.json").is_file()


def test_warmup_ladder_falls_through_to_second_rung():
    env = ChainEnv(length=4, horizon=6)
    buffer, report = collect_warmup(
        env,
        ladder=[("stayer", always_stay), ("solver", always_right)],
        min_successes=3,
        max_env_steps=120,  # 60 steps per rung: stayer burns its share, solver delivers
    )
    assert report["gate"]["met"] is True
    assert report["per_source"]["stayer"]["successes"] == 0
    assert report["per_source"]["solver"]["successes"] >= 3


def test_warmup_zero_success_halts_with_diagnostic(tmp_path):
    env = ChainEnv(length=4, horizon=6)
    report_path = tmp_path / "warmup.json"
    with pytest.raises(WarmupGateError, match="must not proceed") as excinfo:
        collect_warmup(
            env,
            ladder=[("stayer", always_stay)],
            min_successes=1,
            max_env_steps=60,
            report_path=report_path,
        )
    # the diagnostic report exists both on the exception and on disk
    assert excinfo.value.report["success_count"] == 0
    assert excinfo.value.report["gate"]["met"] is False
    assert report_path.is_file()


def test_warmup_cap_is_exact_even_mid_episode(tmp_path):
    """Budget cap smaller than one episode horizon: collection stops at the
    cap exactly and records the truncation (regression for cap overshoot)."""
    env = ChainEnv(length=4, horizon=6)
    with pytest.raises(WarmupGateError) as excinfo:
        collect_warmup(
            env,
            ladder=[("stayer", always_stay)],
            min_successes=1,
            max_env_steps=1,  # far below horizon 6
            report_path=tmp_path / "warmup.json",
        )
    report = excinfo.value.report
    assert report["total_env_steps"] == 1  # exact stop, no overshoot
    assert report["gate"]["collection_truncated_at_cap"] is True


def test_warmup_cap_not_marked_truncated_when_gate_met():
    env = ChainEnv(length=4, horizon=10)
    _, report = collect_warmup(
        env, ladder=[("solver", always_right)], min_successes=2, max_env_steps=1000
    )
    assert report["gate"]["collection_truncated_at_cap"] is False


def test_warmup_fallback_runs_after_first_rung_exhausts_share(tmp_path):
    """Codex round-3 regression: a first rung whose episodes outlast its share
    must NOT consume the global cap; the fallback rung runs and meets the gate."""
    env = ChainEnv(length=4, horizon=10)
    buffer, report = collect_warmup(
        env,
        ladder=[("stayer", always_stay), ("solver", always_right)],
        min_successes=1,
        max_env_steps=10,  # 5-step share per rung; stayer episodes last 10
        report_path=tmp_path / "warmup.json",
    )
    assert report["gate"]["met"] is True
    assert report["per_source"]["solver"]["successes"] >= 1
    assert report["gate"]["rung_truncations"]["stayer"] is True  # share cut it off
    assert report["gate"]["steps_per_rung"] == 5
    assert buffer.total_env_steps <= 10


def test_warmup_rejects_empty_ladder():
    env = ChainEnv()
    with pytest.raises(ValueError, match="ladder"):
        collect_warmup(env, ladder=[], min_successes=1, max_env_steps=10)
