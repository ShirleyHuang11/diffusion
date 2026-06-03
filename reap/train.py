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


RUNNERS = {
    "random": run_random,
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
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        summary = run_from_config(cfg, resume=args.resume)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(f"done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
