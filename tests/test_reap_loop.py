"""Integrated REAP loop tests: gating, snapshot pinning, refresh, resume."""

import numpy as np
import pytest

from reap.algos.mappo import MappoTrainer
from reap.algos.reap_shaping import (
    PotentialSnapshot,
    PredictorPotential,
    ReapShapingController,
)
from reap.calibration import CalibrationLadder
from reap.signals.distill import DistilledPredictor
from reap.signals.potential import ReapPotential
from tests.chain_env import ChainEnv

PASSING_QUALITY = {"shaping_enabled": True, "gate_violations": []}
FAILING_QUALITY = {
    "shaping_enabled": False,
    "gate_violations": ["invalid_state_rate 0.97 > 0.1", "bridge_consistency 0.0 < 0.8"],
}

TINY = {"rollout_length": 16, "hidden_size": 16, "update_epochs": 1, "num_minibatches": 2}


def table_potential(value=0.8):
    potential = ReapPotential(tau_gate=0.5)
    states = np.eye(5, dtype=np.float32)
    potential.update_tables(
        states, propensity=np.full(5, value), feasibility=np.ones(5)
    )
    return potential


def make_controller(quality, refresher=None, refresh_every=2, beta=1.0):
    return ReapShapingController(
        potential=table_potential(),
        ladder=CalibrationLadder(beta=beta),
        quality_report=quality,
        refresh_every_updates=refresh_every,
        refresher=refresher,
    )


def test_disabled_shaping_produces_zero_terms_and_logs_reason():
    controller = make_controller(FAILING_QUALITY)
    gate_events = [e for e in controller.events if e["type"] == "gate"]
    assert gate_events and gate_events[0]["enabled"] is False
    assert "invalid_state_rate" in gate_events[0]["reason"]

    trainer = MappoTrainer(ChainEnv(5, 8), TINY, seed=0)
    trainer.shaping_provider = controller
    rollout = trainer.collect_rollout()
    assert np.all(rollout["shaping"] == 0.0)  # gate forces exact zeros
    assert rollout["shaping_snapshot"] == 0


def test_enabled_shaping_produces_terms_from_the_table():
    controller = make_controller(PASSING_QUALITY)
    trainer = MappoTrainer(ChainEnv(5, 8), TINY, seed=0)
    trainer.shaping_provider = controller
    rollout = trainer.collect_rollout()
    assert np.any(rollout["shaping"] != 0.0)


def test_refresh_forbidden_while_batch_pinned():
    controller = make_controller(PASSING_QUALITY)
    with controller.pin():
        with pytest.raises(RuntimeError, match="pinned"):
            controller.maybe_refresh(controller.refresh_every)
    # after release the same refresh succeeds
    event = controller.maybe_refresh(controller.refresh_every)
    assert event["snapshot_id_after"] == 1


def test_double_pin_rejected():
    controller = make_controller(PASSING_QUALITY)
    with controller.pin():
        with pytest.raises(RuntimeError, match="already pinned"):
            controller.pin().__enter__()


def test_refresh_schedule_and_metadata():
    calls = []

    def refresher():
        calls.append(1)
        states = np.eye(5, dtype=np.float32)
        return states, np.full(5, 0.6), np.ones(5)

    controller = make_controller(PASSING_QUALITY, refresher=refresher, refresh_every=2)
    assert controller.maybe_refresh(1) is None  # not on the schedule
    event = controller.maybe_refresh(2)
    assert event["refreshed_states"] == 5 and len(calls) == 1
    assert event["snapshot_id_before"] == 0 and event["snapshot_id_after"] == 1
    assert "time" in event
    # the refreshed table is visible to the next snapshot
    with controller.pin() as snap:
        assert snap.snapshot_id == 1
        assert snap.value(None, np.eye(5, dtype=np.float32)[0], 4) == pytest.approx(0.6)


def test_calibration_ladder_beta_propagates_to_next_snapshot():
    controller = make_controller(PASSING_QUALITY, refresh_every=1, beta=1.0)
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0, 1, 2000)
    realized = (rng.uniform(0, 1, 2000) < (1.0 - predicted)).astype(float)  # unfixable
    event = controller.maybe_refresh(1, predicted, realized)
    assert event["calibration"]["action"] == "shrink_beta"
    with controller.pin() as snap:
        assert snap.beta == pytest.approx(0.5)  # shrunk beta pins into the batch


def test_controller_state_roundtrip_restores_snapshot_and_ladder():
    controller = make_controller(PASSING_QUALITY, refresh_every=1, beta=1.0)
    rng = np.random.default_rng(1)
    predicted = rng.uniform(0, 1, 1000)
    realized = (rng.uniform(0, 1, 1000) < (1.0 - predicted)).astype(float)
    controller.maybe_refresh(1, predicted, realized)
    state = controller.state_dict()

    clone = make_controller(PASSING_QUALITY)
    clone.load_state_dict(state)
    assert clone.snapshot_id == controller.snapshot_id == 1
    assert clone.refresh_count == 1
    assert clone.ladder.beta == controller.ladder.beta
    assert clone.events == controller.events
    with clone.pin() as snap:
        assert snap.snapshot_id == 1


def test_predictor_potential_gates_on_f_hat():
    rng = np.random.default_rng(2)
    states = rng.normal(size=(400, 4)).astype(np.float32)
    p_hat = DistilledPredictor(4, seed=0)
    p_hat.fit(states, np.full(400, 0.9, dtype=np.float32), epochs=80)
    f_low = DistilledPredictor(4, seed=1)
    f_low.fit(states, np.full(400, 0.05, dtype=np.float32), epochs=80)
    f_high = DistilledPredictor(4, seed=2)
    f_high.fit(states, np.full(400, 0.95, dtype=np.float32), epochs=80)

    gated = PredictorPotential(p_hat, f_low, tau_gate=0.5)
    assert gated.value(states[0], 5) == 0.0  # feasibility below the gate
    open_potential = PredictorPotential(p_hat, f_high, tau_gate=0.5)
    assert open_potential.value(states[0], 5) == pytest.approx(0.9, abs=0.1)


def test_build_calibration_payload_disjoint_split(tmp_path):
    """The payload builder reproduces an episode-level disjoint split."""
    from reap.data import TrajectoryBuffer
    from reap.teacher_pipeline import build_calibration_payload

    rng = np.random.default_rng(0)
    buffer = TrajectoryBuffer(state_dim=6)
    for i in range(20):
        states = rng.normal(size=(12, 6)).astype(np.float32)
        states[:, -1] = (np.arange(12) > 8).astype(np.float32) * 0.1  # late delivery
        buffer.add_episode(states, ret=float(i % 2), success=i % 2 == 0, source="t")
    buffer.save(tmp_path / "warmup_buffer.npz")

    info = build_calibration_payload(tmp_path, window=4, seed=0)
    assert info["refresh_anchors"] > 0
    assert info["calibration_anchors"] > 0
    train_set = set(info["train_episode_indices"])
    holdout_set = set(info["holdout_episode_indices"])
    assert train_set and holdout_set and not (train_set & holdout_set)
    payload = np.load(tmp_path / "calibration_holdout.npz", allow_pickle=True)
    assert {"refresh_anchors", "calibration_anchors", "calibration_realized",
            "provenance", "disjointness"} <= set(payload.files)


def _reap_cfg(tmp_path, name, total_steps, log_every=64, ckpt_every=128,
              refresh_every=1):
    from reap.config import (
        AlgoConfig, CheckpointConfig, Config, EnvConfig, LoggingConfig, RunConfig,
    )

    return Config(
        run=RunConfig(name=name, seed=0, mode="smoke",
                      out_dir=str(tmp_path / "runs"), max_wall_clock_minutes=20.0,
                      device="cpu"),
        env=EnvConfig(id="overcooked", layout="cramped_room", horizon=50,
                      encoding="lossless"),
        algo=AlgoConfig(name="reap_mappo", total_env_steps=total_steps, params={
            "rollout_length": 64, "hidden_size": 32, "update_epochs": 1,
            "num_minibatches": 2, "refresh_every_k_updates": refresh_every,
        }),
        logging=LoggingConfig(interval_env_steps=log_every),
        checkpoint=CheckpointConfig(interval_env_steps=ckpt_every, keep_last=1),
    )


def _skip_without_artifacts():
    # disabled-scope runner tests need only the committed quality artifact
    pytest.importorskip("overcooked_ai_py")


def _enabled_quality(tmp_path):
    import json

    path = tmp_path / "quality_pass.json"
    path.write_text(json.dumps({"shaping_enabled": True, "gate_violations": []}))
    return str(path)


def _skip_without_hybrid_artifacts():
    from pathlib import Path

    pytest.importorskip("overcooked_ai_py")
    for needed in ("runs/teacher_hybrid_cramped/predictors.pt",
                   "runs/teacher_hybrid_cramped/calibration_holdout.npz",
                   "runs/teacher_hybrid_cramped/probe_observations.npy",
                   "runs/teacher_hybrid_cramped/teacher.pt"):
        if not Path(needed).is_file():
            pytest.skip(f"{needed} not present in this checkout")


def _hybrid_params(tmp_path, refresh_mode="fixed_predictors"):
    return {
        "quality_report": _enabled_quality(tmp_path),
        "predictors": "runs/teacher_hybrid_cramped/predictors.pt",
        "calibration_payload": "runs/teacher_hybrid_cramped/calibration_holdout.npz",
        "probe_observations": "runs/teacher_hybrid_cramped/probe_observations.npy",
        "teacher": "runs/teacher_hybrid_cramped/teacher.pt",
        "refresh_mode": refresh_mode,
    }


@pytest.mark.overcooked
def test_reap_mappo_refreshes_are_real_and_calibrated(tmp_path):
    """Enabled scope: every scheduled refresh refreshes real states AND
    carries calibration computed AFTER the refresh."""
    import dataclasses
    import json

    _skip_without_hybrid_artifacts()
    from reap.train import run_from_config

    cfg = _reap_cfg(tmp_path, "reapcal", total_steps=192, refresh_every=1)
    cfg = dataclasses.replace(
        cfg, algo=dataclasses.replace(
            cfg.algo, params={**cfg.algo.params, **_hybrid_params(tmp_path)})
    )
    run_from_config(cfg)
    events = [json.loads(line) for line in
              (tmp_path / "runs" / "reapcal" / "seed0" / "shaping_events.jsonl")
              .read_text().splitlines()]
    refreshes = [e for e in events if e["type"] == "refresh"]
    assert refreshes, "no refresh events recorded"
    for event in refreshes:
        assert event["refreshed_states"] > 0
        assert "note" not in event  # never "no refresher configured"
        cal = event["calibration"]
        assert {"raw_ece", "brier", "action", "beta_after", "alert"} <= set(cal)


def test_post_refresh_calibration_reflects_refit_predictor():
    """Codex round-10 gap 1: when the refresher changes p-hat, the
    calibration event must describe the POST-refresh predictions."""
    from reap.signals.distill import DistilledPredictor

    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(60, 4)).astype(np.float32)
    cal_anchors = rng.normal(size=(40, 4)).astype(np.float32)
    realized = np.zeros(40)  # nothing ever succeeds

    p_hat = DistilledPredictor(4, hidden=32, seed=0)
    p_hat.fit(anchors, np.full(60, 0.95, dtype=np.float32), epochs=120)  # overconfident
    pre_refresh_pred = np.clip(p_hat.predict(cal_anchors), 0, 1)
    assert pre_refresh_pred.mean() > 0.7  # stale predictor is overconfident

    potential = table_potential()

    def refresher():
        # the refresh refits p_hat toward honesty (low propensity)
        p_hat.fit(anchors, np.full(60, 0.02, dtype=np.float32), epochs=120)
        return anchors, np.full(60, 0.02), np.ones(60)

    controller = ReapShapingController(
        potential=potential, ladder=CalibrationLadder(beta=1.0),
        quality_report=PASSING_QUALITY, refresh_every_updates=1,
        refresher=refresher,
    )
    event = controller.maybe_refresh(
        1, calibration_fn=lambda: (np.clip(p_hat.predict(cal_anchors), 0, 1), realized)
    )
    # post-refresh predictions are near zero -> nearly calibrated vs zeros;
    # had the stale predictor been used, ECE would be ~0.7+
    assert event["calibration"]["raw_ece"] < 0.2
    post = np.clip(p_hat.predict(cal_anchors), 0, 1)
    assert post.mean() < 0.2


@pytest.mark.overcooked
def test_reap_mappo_final_record_at_exact_budget(tmp_path):
    """Non-divisible budget/log-interval still ends with a record at the budget."""
    _skip_without_artifacts()
    from reap.metrics import read_jsonl
    from reap.train import run_from_config

    cfg = _reap_cfg(tmp_path, "reapfinal", total_steps=70, log_every=64,
                    refresh_every=50)
    summary = run_from_config(cfg)
    assert summary["env_step"] == 70
    records = read_jsonl(tmp_path / "runs" / "reapfinal" / "seed0" / "metrics.jsonl")
    assert records[-1]["env_step"] == 70


@pytest.mark.overcooked
def test_reap_mappo_resume_no_duplicate_gate_event(tmp_path):
    import dataclasses
    import json

    _skip_without_artifacts()
    from reap.train import run_from_config

    cfg = _reap_cfg(tmp_path, "reapresume", total_steps=64, ckpt_every=64,
                    refresh_every=50)
    run_from_config(cfg)
    cfg_more = dataclasses.replace(
        cfg, algo=dataclasses.replace(cfg.algo, total_env_steps=128)
    )
    run_from_config(cfg_more, resume=True)
    events = [json.loads(line) for line in
              (tmp_path / "runs" / "reapresume" / "seed0" / "shaping_events.jsonl")
              .read_text().splitlines()]
    gate_events = [e for e in events if e["type"] == "gate"]
    assert len(gate_events) == 1  # resume must not append a duplicate


@pytest.mark.overcooked
def test_reap_mappo_refuses_missing_calibration_payload_when_enabled(tmp_path):
    pytest.importorskip("overcooked_ai_py")
    import dataclasses
    from pathlib import Path

    if not Path("runs/teacher_hybrid_cramped/predictors.pt").is_file():
        pytest.skip("teacher predictors not present in this checkout")
    from reap.config import ConfigError
    from reap.train import run_from_config

    cfg = _reap_cfg(tmp_path, "reapnopayload", total_steps=64)
    cfg = dataclasses.replace(
        cfg,
        algo=dataclasses.replace(
            cfg.algo,
            params={**cfg.algo.params,
                    "quality_report": _enabled_quality(tmp_path),
                    "predictors": "runs/teacher_hybrid_cramped/predictors.pt",
                    "calibration_payload": str(tmp_path / "absent.npz")},
        ),
    )
    with pytest.raises(ConfigError, match="calibration payload"):
        run_from_config(cfg)


@pytest.mark.overcooked
def test_reap_mappo_policy_conditioned_runner_and_resume(tmp_path, monkeypatch):
    """Runner-boundary policy_conditioned test with a stub sampler: refreshes
    track the live policy, calibration is post-refit, predictor state rides
    in checkpoints and restores on resume."""
    import dataclasses
    import json

    _skip_without_hybrid_artifacts()
    import reap.hybrid_teacher as hybrid

    calls = {"n": 0}

    def stub_factory(teacher_path, samples_per_state=8, device="auto", seed=0):
        payload = np.load("runs/teacher_hybrid_cramped/calibration_holdout.npz",
                          allow_pickle=True)
        fdim = payload["refresh_anchors_features"].shape[1]
        window = 4

        def sampler(embedding_vec, feature_anchors):
            calls["n"] += 1
            m = 4
            windows = np.zeros((len(feature_anchors), m, window, fdim), dtype=np.float32)
            windows[:, :2, -1, -1] = 0.1  # half the samples deliver
            return windows

        return sampler

    monkeypatch.setattr(hybrid, "make_teacher_sampler", stub_factory)

    cfg = _reap_cfg(tmp_path, "reappc", total_steps=128, ckpt_every=128,
                    refresh_every=1)
    cfg = dataclasses.replace(
        cfg, algo=dataclasses.replace(
            cfg.algo,
            params={**cfg.algo.params,
                    **_hybrid_params(tmp_path, refresh_mode="policy_conditioned")})
    )
    from reap.train import run_from_config

    summary = run_from_config(cfg)
    assert summary["shaping_enabled"] is True
    assert calls["n"] >= 1  # the teacher stub was queried with live embeddings

    run_dir = tmp_path / "runs" / "reappc" / "seed0"
    events = [json.loads(line) for line in
              (run_dir / "shaping_events.jsonl").read_text().splitlines()]
    refreshes = [e for e in events if e["type"] == "refresh"]
    assert refreshes and all("calibration" in e for e in refreshes)

    # predictor weights mutated by the refit must ride in the checkpoint:
    # resume and verify the controller restores them (predictions match)
    from reap.checkpoint import latest_checkpoint, load_checkpoint

    payload = load_checkpoint(latest_checkpoint(run_dir / "checkpoints"))
    assert "p_hat" in payload["controller"]["potential"]

    from reap.signals.distill import DistilledPredictor

    saved = payload["controller"]["potential"]["p_hat"]
    probe = np.load("runs/teacher_hybrid_cramped/calibration_holdout.npz",
                    allow_pickle=True)["calibration_anchors"].astype(np.float32)[:8]
    restored = DistilledPredictor(saved["state_dim"], hidden=saved["net"]["0.weight"].shape[0])
    restored.load_state_dict(saved)
    expected = restored.predict(probe)

    # resume with refreshes effectively off: the resumed segment must carry
    # the REFIT weights forward unchanged (restored from the checkpoint, not
    # rebuilt from the original artifact)
    cfg_more = dataclasses.replace(
        cfg, algo=dataclasses.replace(
            cfg.algo, total_env_steps=192,
            params={**cfg.algo.params, "refresh_every_k_updates": 10_000}))
    run_from_config(cfg_more, resume=True)
    payload2 = load_checkpoint(latest_checkpoint(run_dir / "checkpoints"))
    saved2 = payload2["controller"]["potential"]["p_hat"]
    restored2 = DistilledPredictor(saved2["state_dim"], hidden=saved2["net"]["0.weight"].shape[0])
    restored2.load_state_dict(saved2)
    assert np.allclose(restored2.predict(probe), expected, atol=1e-6)


@pytest.mark.overcooked
def test_reap_mappo_runner_disabled_path_smoke(tmp_path):
    """End-to-end reap_mappo smoke on the real committed artifacts: shaping is
    disabled by the quality gate, terms are zero, events and controller
    checkpoints exist."""
    import json
    from pathlib import Path

    pytest.importorskip("overcooked_ai_py")
    if not Path("runs/teacher_cramped/predictors.pt").is_file():
        pytest.skip("teacher predictors not present in this checkout")

    from reap.config import (
        AlgoConfig, CheckpointConfig, Config, EnvConfig, LoggingConfig, RunConfig,
    )
    from reap.metrics import read_jsonl
    from reap.train import run_from_config

    cfg = Config(
        run=RunConfig(name="reapsmoke", seed=0, mode="smoke",
                      out_dir=str(tmp_path / "runs"), max_wall_clock_minutes=20.0,
                      device="cpu"),
        env=EnvConfig(id="overcooked", layout="cramped_room", horizon=50,
                      encoding="lossless"),
        algo=AlgoConfig(name="reap_mappo", total_env_steps=128, params={
            "rollout_length": 64, "hidden_size": 32, "update_epochs": 1,
            "num_minibatches": 2, "refresh_every_k_updates": 1,
        }),
        logging=LoggingConfig(interval_env_steps=64),
        checkpoint=CheckpointConfig(interval_env_steps=128, keep_last=1),
    )
    summary = run_from_config(cfg)
    assert summary["env_step"] == 128
    assert summary["shaping_enabled"] is False  # committed artifact gates it off

    run_dir = tmp_path / "runs" / "reapsmoke" / "seed0"
    records = read_jsonl(run_dir / "metrics.jsonl")
    for rec in records:
        assert rec["shaped"]["term_mean"] == 0.0
        assert rec["shaped"]["enabled"] == 0.0
        assert "gpu_mem_mb" in rec["diag"]
        assert "refresh_count" in rec["diag"]
    events = [json.loads(line) for line in
              (run_dir / "shaping_events.jsonl").read_text().splitlines()]
    assert events[0]["type"] == "gate" and events[0]["enabled"] is False
    # gate-disabled scopes never refresh: the audit trail is the gate event
    assert not any(e["type"] == "refresh" for e in events)

    from reap.checkpoint import latest_checkpoint, load_checkpoint

    payload = load_checkpoint(latest_checkpoint(run_dir / "checkpoints"))
    assert "controller" in payload
    assert payload["controller"]["enabled"] is False


def test_calibration_disable_flips_controller_enabled():
    controller = make_controller(
        PASSING_QUALITY, refresh_every=1, beta=1.0
    )
    controller.ladder.beta_floor = 0.9  # one shrink crosses the floor
    rng = np.random.default_rng(3)
    predicted = rng.uniform(0, 1, 1000)
    realized = (rng.uniform(0, 1, 1000) < (1.0 - predicted)).astype(float)
    event = controller.maybe_refresh(1, predicted, realized)
    assert event["calibration"]["action"] == "disabled"
    assert controller.enabled is False
    with controller.pin() as snap:
        assert snap.value(None, np.eye(5, dtype=np.float32)[0], 4) == 0.0