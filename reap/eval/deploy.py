"""Deployment evaluation of a trained policy checkpoint.

This module is the deployment boundary: it constructs actors from a training
checkpoint and rolls them in the environment. It must never import the
trajectory teacher, signal estimators, shaping, or the teacher pipeline —
the shipped agent is an ordinary MARL policy. An import-graph test enforces
this, and a raising-stub test proves evaluation works with those modules
unavailable.
"""

from __future__ import annotations

import numpy as np
import torch

from reap.algos.mappo import DEFAULT_PARAMS, MappoNets
from reap.checkpoint import load_checkpoint
from reap.envs.base import CoopEnv

FORBIDDEN_IMPORTS = (
    "reap.diffusion",
    "reap.signals",
    "reap.shaping",
    "reap.teacher_pipeline",
    "reap.algos.reap_shaping",
)


def load_policy(checkpoint_path, env: CoopEnv) -> MappoNets:
    """Rebuild actor/critic nets from a training checkpoint (policy only)."""
    payload = load_checkpoint(checkpoint_path)
    trainer_state = payload["trainer"]
    hidden = (
        payload.get("config", {})
        .get("algo", {})
        .get("params", {})
        .get("hidden_size", DEFAULT_PARAMS["hidden_size"])
    )
    nets = MappoNets(
        env.local_obs_dim, env.joint_state_dim, env.num_agents, env.num_actions, hidden
    )
    nets.load_state_dict(trainer_state["nets"])
    nets.eval()
    return nets


def evaluate_policy(
    nets: MappoNets, env: CoopEnv, episodes: int = 10, seed: int = 0
) -> dict:
    """Roll the policy for ``episodes`` episodes; extrinsic metrics only."""
    torch.manual_seed(seed)
    returns, successes = [], []
    first_actions: list[list[int]] = []
    for _ in range(episodes):
        local_obs, _ = env.reset()
        ep_return = 0.0
        recorded_first = False
        while True:
            with torch.no_grad():
                actions, _ = nets.act(local_obs)
            if not recorded_first:
                first_actions.append([int(a) for a in actions])
                recorded_first = True
            result = env.step(actions.tolist())
            ep_return += result.extrinsic_reward
            local_obs = result.local_obs
            if result.terminated or result.truncated:
                successes.append(float(result.info.get("success", False)))
                returns.append(ep_return)
                break
    return {
        "episodes": episodes,
        "extrinsic_return_mean": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        # reference trace for boundary tests: deterministic given seed+nets
        "first_actions": first_actions,
    }


def evaluate_checkpoint(checkpoint_path, env: CoopEnv, episodes: int = 10, seed: int = 0) -> dict:
    return evaluate_policy(load_policy(checkpoint_path, env), env, episodes, seed)
