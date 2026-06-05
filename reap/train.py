"""Config-driven training entrypoint.

Usage:
    python -m reap.train --config configs/smoke_random_cramped.yaml [--resume]

Dispatches on ``algo.name``. The ``random`` runner exercises the full
infrastructure (seeding, rollouts, metrics, checkpoints, wall-clock caps) with
uniform-random actions; learning algorithms register here as they land.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from reap.checkpoint import (
    checkpoint_name,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from reap.config import Config, ConfigError, load_config
from reap.envs import make_env
from reap.metrics import MetricsLogger
from reap.seeding import seed_everything


class WallClockExceeded(RuntimeError):
    """Raised when a run hits its configured wall-clock cap."""


def _env_kwargs(cfg: Config) -> dict:
    kwargs = {"horizon": cfg.env.horizon}
    if cfg.env.id == "overcooked":
        kwargs["layout"] = cfg.env.layout
        kwargs["encoding"] = cfg.env.encoding
    elif cfg.env.id == "mpe_spread":
        kwargs["num_agents"] = cfg.env.num_agents
    return kwargs


def run_random(cfg: Config, out_dir: Path, resume: bool) -> dict:
    """Uniform-random rollout runner over the configured environment."""
    env = make_env(cfg.env.id, **_env_kwargs(cfg))
    rng = np.random.default_rng(cfg.run.seed)
    logger = MetricsLogger(out_dir, jsonl=cfg.logging.jsonl, csv_enabled=cfg.logging.csv)
    ckpt_dir = out_dir / "checkpoints"

    env_step = 0
    episodes = 0
    successes = 0
    return_sum = 0.0
    ep_return = 0.0
    if resume:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise ConfigError(f"--resume requested but no checkpoint found in {ckpt_dir}")
        state = load_checkpoint(ckpt_path)
        env_step = state["env_step"]
        episodes = state["episodes"]
        successes = state["successes"]
        return_sum = state["return_sum"]
        ep_return = state["ep_return"]
        rng.bit_generator.state = state["rng_state"]
        env.set_state(state["env_snapshot"])  # exact mid-episode continuation
    else:
        env.reset(seed=cfg.run.seed)

    def snapshot() -> dict:
        return {
            "env_step": env_step,
            "episodes": episodes,
            "successes": successes,
            "return_sum": return_sum,
            "ep_return": ep_return,
            "env_snapshot": env.get_state(),
            "rng_state": rng.bit_generator.state,
            "config": cfg.to_dict(),
        }

    start_time = time.monotonic()
    deadline = start_time + cfg.run.max_wall_clock_minutes * 60

    def next_on_grid(interval: int) -> int:
        # next multiple of `interval` strictly after env_step, so a resumed run
        # rejoins the same logging/checkpoint grid as an uninterrupted one
        return (env_step // interval + 1) * interval

    next_log = next_on_grid(cfg.logging.interval_env_steps)
    next_ckpt = next_on_grid(cfg.checkpoint.interval_env_steps)

    while env_step < cfg.algo.total_env_steps:
        if time.monotonic() > deadline:
            raise WallClockExceeded(
                f"run exceeded max_wall_clock_minutes={cfg.run.max_wall_clock_minutes}"
            )
        actions = rng.integers(0, env.num_actions, size=env.num_agents).tolist()
        result = env.step(actions)
        env_step += 1
        ep_return += result.extrinsic_reward

        if result.terminated or result.truncated:
            episodes += 1
            successes += int(result.info.get("success", False))
            return_sum += ep_return
            ep_return = 0.0
            env.reset()

        if env_step >= next_log:
            logger.log(
                env_step,
                extrinsic={
                    "episode_return_mean": (return_sum / episodes) if episodes else 0.0,
                    "episodes": episodes,
                    "success_rate": (successes / episodes) if episodes else 0.0,
                },
                diag={"wall_time_s": time.monotonic() - start_time},
            )
            next_log += cfg.logging.interval_env_steps

        if env_step >= next_ckpt:
            save_checkpoint(snapshot(), ckpt_dir / checkpoint_name(env_step))
            prune_checkpoints(ckpt_dir, cfg.checkpoint.keep_last)
            next_ckpt += cfg.checkpoint.interval_env_steps

    save_checkpoint(snapshot(), ckpt_dir / checkpoint_name(env_step))
    prune_checkpoints(ckpt_dir, cfg.checkpoint.keep_last)
    return {
        "env_step": env_step,
        "episodes": episodes,
        "success_rate": (successes / episodes) if episodes else 0.0,
    }


def run_mappo(cfg: Config, out_dir: Path, resume: bool) -> dict:
    """MAPPO training (optionally with an intrinsic bonus) on the configured env."""
    from reap.algos.mappo import MappoTrainer

    env = make_env(cfg.env.id, **_env_kwargs(cfg))
    trainer = MappoTrainer(env, cfg.algo.params, seed=cfg.run.seed)
    logger = MetricsLogger(out_dir, jsonl=cfg.logging.jsonl, csv_enabled=cfg.logging.csv)
    ckpt_dir = out_dir / "checkpoints"

    if resume:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise ConfigError(f"--resume requested but no checkpoint found in {ckpt_dir}")
        trainer.load_state_dict(load_checkpoint(ckpt_path)["trainer"])

    start_time = time.monotonic()
    deadline = start_time + cfg.run.max_wall_clock_minutes * 60

    def grid_next(interval: int) -> int:
        return (trainer.env_step // interval + 1) * interval

    next_log = grid_next(cfg.logging.interval_env_steps)
    next_ckpt = grid_next(cfg.checkpoint.interval_env_steps)

    def save() -> None:
        save_checkpoint(
            {"trainer": trainer.state_dict(), "config": cfg.to_dict()},
            ckpt_dir / checkpoint_name(trainer.env_step),
        )
        prune_checkpoints(ckpt_dir, cfg.checkpoint.keep_last)

    last_logged = -1
    rollout: dict | None = None
    diags: dict = {}

    def log_now(rollout: dict, diags: dict) -> None:
        nonlocal last_logged
        logger.log(
            trainer.env_step,
            extrinsic=trainer.episode_stats(),
            shaped={
                "term_mean": float(rollout["shaping"].mean()),
                "term_abs_max": float(np.abs(rollout["shaping"]).max()),
            },
            intrinsic={
                "bonus_mean": float(rollout["intrinsic"].mean()),
                **{k: v for k, v in rollout["bonus_diag"].items()},
            },
            diag={
                **diags,
                "updates": float(trainer.updates),
                "wall_time_s": time.monotonic() - start_time,
            },
        )
        last_logged = trainer.env_step

    while trainer.env_step < cfg.algo.total_env_steps:
        if time.monotonic() > deadline:
            save()  # long runs preserve progress before stopping
            raise WallClockExceeded(
                f"run exceeded max_wall_clock_minutes={cfg.run.max_wall_clock_minutes}"
            )
        remaining = cfg.algo.total_env_steps - trainer.env_step
        rollout = trainer.collect_rollout(max_steps=remaining)
        diags = trainer.update(rollout)

        if trainer.env_step >= next_log:
            log_now(rollout, diags)
            next_log = grid_next(cfg.logging.interval_env_steps)
        if trainer.env_step >= next_ckpt:
            save()
            next_ckpt = grid_next(cfg.checkpoint.interval_env_steps)

    if rollout is not None and trainer.env_step != last_logged:
        log_now(rollout, diags)  # final record lands exactly at the budget step
    save()
    final = trainer.episode_stats()
    return {"env_step": trainer.env_step, **final}


def run_reap_mappo(cfg: Config, out_dir: Path, resume: bool) -> dict:
    """MAPPO with REAP shaping under snapshot pinning and gated enablement.

    Shaping is enabled only when the committed teacher-quality artifact passed
    its gates; otherwise the controller produces zero shaping terms and the
    gate reason is logged. Predictors (p-hat/f-hat) come from the teacher
    pipeline output directory.
    """
    import json

    from reap.algos.mappo import MappoTrainer
    from reap.algos.reap_shaping import (
        PredictorPotential,
        ReapShapingController,
    )
    from reap.calibration import CalibrationLadder
    from reap.signals.distill import DistilledPredictor

    params = dict(cfg.algo.params)
    quality_path = params.pop("quality_report", "reports/teacher_quality_cramped.json")
    predictors_path = params.pop("predictors", "runs/teacher_cramped/predictors.pt")
    payload_path = params.pop(
        "calibration_payload", "runs/teacher_cramped/calibration_holdout.npz"
    )
    refresh_every = int(params.pop("refresh_every_k_updates", 50))
    beta = float(params.pop("reap_beta", 1.0))
    tau_gate = float(params.pop("tau_gate", 0.5))
    refresh_mode = params.pop("refresh_mode", "fixed_predictors")
    teacher_path = params.pop("teacher", "runs/teacher_hybrid_cramped/teacher.pt")
    probes_path = params.pop(
        "probe_observations", "runs/teacher_hybrid_cramped/probe_observations.npy"
    )

    quality = json.loads(Path(quality_path).read_text())
    shaping_enabled = bool(quality.get("shaping_enabled", False))

    env = make_env(cfg.env.id, **_env_kwargs(cfg))
    if shaping_enabled and cfg.env.id == "overcooked":
        # the teacher's anchors/predictors live in the delivery-augmented
        # state; training must visit the same state space
        from reap.teacher_pipeline import DeliveryAugmentedOvercooked

        env = DeliveryAugmentedOvercooked(env)
    trainer = MappoTrainer(env, params, seed=cfg.run.seed)

    calibration_fn = None
    refresher = None
    if not shaping_enabled:
        # gate-disabled scope: no teacher artifacts required; the run is the
        # honest disabled-REAP arm with the gate reason in the audit trail
        from reap.algos.reap_shaping import ZeroPotential

        potential = ZeroPotential()
        p_hat = f_hat = None
    else:
        import torch as _torch

        predictor_state = _torch.load(predictors_path, map_location="cpu", weights_only=False)
        state_dim = len(predictor_state["p_hat"]["input_mean"])
        p_hat = DistilledPredictor(state_dim, hidden=predictor_state["p_hat"]["net"]["0.weight"].shape[0])
        p_hat.load_state_dict(predictor_state["p_hat"])
        f_hat = DistilledPredictor(state_dim, hidden=predictor_state["f_hat"]["net"]["0.weight"].shape[0])
        f_hat.load_state_dict(predictor_state["f_hat"])

        # scheduled refreshes must be calibrated: refuse to run without the
        # persisted, episode-disjoint calibration payload
        if not Path(payload_path).is_file():
            raise ConfigError(
                f"calibration payload not found: {payload_path}; reap_mappo refuses "
                "uncalibrated refreshes (build it via reap.teacher_pipeline."
                "build_calibration_payload or rerun the teacher pipeline)"
            )
        payload_npz = np.load(payload_path, allow_pickle=True)
        refresh_anchors = payload_npz["refresh_anchors"].astype(np.float32)
        cal_anchors = payload_npz["calibration_anchors"].astype(np.float32)
        cal_realized = payload_npz["calibration_realized"].astype(np.float64)

        if refresh_mode == "policy_conditioned":
            # enabled scope: re-query the frozen teacher with the CURRENT
            # policy's behavioral embedding and refit the distilled predictors
            from reap.hybrid_teacher import PolicyConditionedRefresher, make_teacher_sampler
            from reap.signals import BehavioralPolicyEmbedding

            if "refresh_anchors_features" not in payload_npz:
                raise ConfigError(
                    "policy_conditioned refresh requires a hybrid calibration "
                    "payload with paired feature anchors"
                )
            refresher = PolicyConditionedRefresher(
                nets_provider=lambda: trainer.nets,
                embedding=BehavioralPolicyEmbedding(np.load(probes_path)),
                sampler=make_teacher_sampler(teacher_path, seed=cfg.run.seed),
                lossless_anchors=refresh_anchors,
                feature_anchors=payload_npz["refresh_anchors_features"].astype(np.float32),
                feasibility=np.clip(f_hat.predict(refresh_anchors), 0.0, 1.0),
                p_hat=p_hat,
                f_hat=f_hat,
            )
        elif refresh_mode == "fixed_predictors":
            def refresher():
                prop = np.clip(p_hat.predict(refresh_anchors), 0.0, 1.0)
                feas = np.clip(f_hat.predict(refresh_anchors), 0.0, 1.0)
                return refresh_anchors, prop, feas
        else:
            raise ConfigError(f"unknown refresh_mode {refresh_mode!r}")

        # the predictor-backed potential generalizes shaping beyond anchor
        # keys; its state_dict carries p_hat/f_hat so refit weights ride in
        # every checkpoint and restore on resume
        potential = PredictorPotential(p_hat, f_hat, tau_gate)

        def calibration_fn():
            # evaluated AFTER the refresher inside maybe_refresh, so the
            # logged ECE/Brier always describe the just-refreshed predictor
            return (
                np.clip(p_hat.predict(cal_anchors), 0.0, 1.0),
                cal_realized,
            )

    controller = ReapShapingController(
        potential=potential,
        ladder=CalibrationLadder(beta=beta),
        quality_report=quality,
        refresh_every_updates=refresh_every,
        refresher=refresher,
    )
    trainer.shaping_provider = controller

    logger = MetricsLogger(out_dir, jsonl=cfg.logging.jsonl, csv_enabled=cfg.logging.csv)
    ckpt_dir = out_dir / "checkpoints"
    if resume:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise ConfigError(f"--resume requested but no checkpoint found in {ckpt_dir}")
        payload = load_checkpoint(ckpt_path)
        trainer.load_state_dict(payload["trainer"])
        controller.load_state_dict(payload["controller"])
        # the resumed run's configured cadence wins over the checkpointed one
        controller.refresh_every = refresh_every

    events_path = out_dir / "shaping_events.jsonl"

    def event_sink(event: dict) -> None:  # events land on disk as they occur
        with events_path.open("a") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")

    # attach the sink AFTER any resume restore so a resumed run does not
    # append a duplicate construction-time gate event; fresh runs flush the
    # buffered gate event exactly once
    controller.event_sink = event_sink
    if not resume:
        for buffered in controller.events:
            event_sink(buffered)

    start_time = time.monotonic()
    deadline = start_time + cfg.run.max_wall_clock_minutes * 60

    def grid_next(interval: int) -> int:
        return (trainer.env_step // interval + 1) * interval

    next_log = grid_next(cfg.logging.interval_env_steps)
    next_ckpt = grid_next(cfg.checkpoint.interval_env_steps)

    def save() -> None:
        save_checkpoint(
            {
                "trainer": trainer.state_dict(),
                "controller": controller.state_dict(),
                "config": cfg.to_dict(),
            },
            ckpt_dir / checkpoint_name(trainer.env_step),
        )
        prune_checkpoints(ckpt_dir, cfg.checkpoint.keep_last)

    def gpu_mem_mb() -> float:
        import torch as _t

        if _t.cuda.is_available():
            return float(_t.cuda.max_memory_allocated() / 1e6)
        return 0.0

    last_logged = -1
    rollout: dict | None = None
    diags: dict = {}

    def log_now(rollout: dict, diags: dict) -> None:
        nonlocal last_logged
        logger.log(
            trainer.env_step,
            extrinsic=trainer.episode_stats(),
            shaped={
                "term_mean": float(rollout["shaping"].mean()),
                "term_abs_max": float(np.abs(rollout["shaping"]).max()),
                "snapshot_id": float(rollout["shaping_snapshot"] or 0),
                "enabled": float(controller.enabled),
            },
            intrinsic={"bonus_mean": float(rollout["intrinsic"].mean())},
            diag={
                **diags,
                "updates": float(trainer.updates),
                "refresh_count": float(controller.refresh_count),
                "wall_time_s": time.monotonic() - start_time,
                "gpu_mem_mb": gpu_mem_mb(),
            },
        )
        last_logged = trainer.env_step

    while trainer.env_step < cfg.algo.total_env_steps:
        if time.monotonic() > deadline:
            save()
            raise WallClockExceeded(
                f"run exceeded max_wall_clock_minutes={cfg.run.max_wall_clock_minutes}"
            )
        # gate-disabled scopes never refresh; enabled scopes refresh on the
        # K-schedule with POST-refresh calibration on the persisted holdout
        if controller.enabled:
            controller.maybe_refresh(trainer.updates, calibration_fn=calibration_fn)
        remaining = cfg.algo.total_env_steps - trainer.env_step
        rollout = trainer.collect_rollout(max_steps=remaining)
        diags = trainer.update(rollout)

        if trainer.env_step >= next_log:
            log_now(rollout, diags)
            next_log = grid_next(cfg.logging.interval_env_steps)
        if trainer.env_step >= next_ckpt:
            save()
            next_ckpt = grid_next(cfg.checkpoint.interval_env_steps)

    if rollout is not None and trainer.env_step != last_logged:
        log_now(rollout, diags)  # final record at exactly the budget step
    save()
    return {"env_step": trainer.env_step, **trainer.episode_stats(),
            "shaping_enabled": controller.enabled}


RUNNERS = {
    "random": run_random,
    "mappo": run_mappo,
    "reap_mappo": run_reap_mappo,
}


def run_from_config(cfg: Config, resume: bool = False) -> dict:
    """Seed, prepare the output directory, and execute the configured runner."""
    if cfg.algo.name not in RUNNERS:
        raise ConfigError(
            f"unknown algo.name {cfg.algo.name!r}; available: {sorted(RUNNERS)}"
        )
    seed_everything(cfg.run.seed, cfg.run.deterministic_torch)
    out_dir = Path(cfg.run.out_dir) / cfg.run.name / f"seed{cfg.run.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True))
    return RUNNERS[cfg.algo.name](cfg, out_dir, resume=resume)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override run.seed (for multi-seed protocols sharing one config)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.seed is not None:
            import dataclasses

            cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, seed=args.seed))
        summary = run_from_config(cfg, resume=args.resume)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(f"done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
