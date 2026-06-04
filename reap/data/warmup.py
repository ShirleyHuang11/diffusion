"""Warmup trajectory collection with a success-gated policy ladder.

The generative teacher needs a buffer containing enough successful episodes.
Collection walks a ladder of increasingly exploratory policies (e.g. vanilla
MAPPO, then MAPPO+RND): each rung collects until either the success minimum
is met (collection stops early) or the rung's share of the step budget is
exhausted. If the full ladder finishes below the success minimum, collection
fails loudly with the diagnostic report — downstream teacher training must
never proceed silently on an insufficient buffer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from reap.data.buffer import TrajectoryBuffer
from reap.envs.base import CoopEnv

# a policy maps (per-agent local obs, joint state) -> list of action indices
PolicyFn = Callable[[list, np.ndarray], list]


class WarmupGateError(RuntimeError):
    """Raised when warmup collection cannot meet the success-count gate."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def random_policy(env: CoopEnv, rng: np.random.Generator) -> PolicyFn:
    def policy(local_obs, joint_state):
        return rng.integers(0, env.num_actions, size=env.num_agents).tolist()

    return policy


def mappo_policy(trainer) -> PolicyFn:
    """Greedy-stochastic policy from a (possibly checkpoint-restored) trainer."""

    def policy(local_obs, joint_state):
        actions, _ = trainer.nets.act(local_obs)
        return actions.tolist()

    return policy


def collect_warmup(
    env: CoopEnv,
    ladder: Sequence[tuple[str, PolicyFn]],
    min_successes: int = 25,
    max_env_steps: int = 5_000_000,
    report_path: str | Path | None = None,
) -> tuple[TrajectoryBuffer, dict]:
    """Collect a warmup buffer along the policy ladder.

    Returns ``(buffer, report)`` when the success gate is met; raises
    :class:`WarmupGateError` carrying the diagnostic report otherwise. The
    report is also written to ``report_path`` (if given) in both cases.
    """
    if not ladder:
        raise ValueError("warmup ladder must contain at least one policy")
    if min_successes < 0:
        raise ValueError("min_successes must be non-negative")

    buffer = TrajectoryBuffer(env.joint_state_dim)
    steps_per_rung = max(1, max_env_steps // len(ladder))
    total_steps = 0
    truncated_collection = False

    for rung_name, policy in ladder:
        rung_steps = 0
        while buffer.success_count < min_successes and rung_steps < steps_per_rung:
            local_obs, joint = env.reset()
            states = [joint]
            ep_return = 0.0
            success = False
            while True:
                if total_steps >= max_env_steps:  # hard global cap, even mid-episode
                    truncated_collection = True
                    break
                result = env.step(policy(local_obs, joint))
                rung_steps += 1
                total_steps += 1
                states.append(result.joint_state)
                ep_return += result.extrinsic_reward
                local_obs, joint = result.local_obs, result.joint_state
                if result.terminated or result.truncated:
                    success = bool(result.info.get("success", False))
                    break
            if len(states) >= 2:  # a capped partial episode is still useful data
                buffer.add_episode(np.stack(states), ep_return, success, source=rung_name)
            if truncated_collection:
                break
        if truncated_collection or buffer.success_count >= min_successes:
            break

    report = buffer.report()
    report["gate"] = {
        "min_successes": min_successes,
        "max_env_steps": max_env_steps,
        "met": buffer.success_count >= min_successes,
        "ladder": [name for name, _ in ladder],
        "collection_truncated_at_cap": truncated_collection,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    if buffer.success_count < min_successes:
        raise WarmupGateError(
            f"warmup collection ended with {buffer.success_count} successful episodes "
            f"(< required {min_successes}) after exhausting the ladder "
            f"{[name for name, _ in ladder]}; teacher training must not proceed. "
            f"Diagnostic report{f' written to {report_path}' if report_path else ''}.",
            report,
        )
    return buffer, report
