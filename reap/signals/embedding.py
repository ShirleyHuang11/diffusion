"""Behavioral policy embeddings on a fixed probe-observation set.

A policy is summarized by its action distributions on a fixed set of probe
observations sampled once (e.g. from the warmup phase). The embedding is the
concatenation of those distributions — cheap, policy-faithful, refreshable
without touching the trajectory teacher, and far smaller than raw weights.
"""

from __future__ import annotations

import numpy as np
import torch


def collect_probe_observations(env, policy_fn, n_probes: int, rng: np.random.Generator):
    """Sample ``n_probes`` per-agent observation tuples by rolling the env."""
    probes = []
    local_obs, joint = env.reset()
    while len(probes) < n_probes:
        if rng.random() < 0.25:  # spread probes across the visitation, not just starts
            probes.append(np.asarray(local_obs).copy())
        result = env.step(policy_fn(local_obs, joint))
        local_obs, joint = result.local_obs, result.joint_state
        if result.terminated or result.truncated:
            local_obs, joint = env.reset()
    return np.stack(probes)  # (P, N, obs_dim)


class BehavioralPolicyEmbedding:
    """Embeds actor networks by their action probabilities on probe states."""

    def __init__(self, probe_observations: np.ndarray):
        probe_observations = np.asarray(probe_observations, dtype=np.float32)
        if probe_observations.ndim != 3:
            raise ValueError(
                f"probe observations must be (P, N, obs_dim), got {probe_observations.shape}"
            )
        self.probes = probe_observations

    @property
    def dim(self) -> int:
        raise AttributeError("dim is known only after the first embed() call")

    def embed(self, nets) -> np.ndarray:
        """Concatenated action distributions of ``nets`` on the probe set."""
        p, n, _ = self.probes.shape
        with torch.no_grad():
            probs = []
            for i in range(p):
                logits = nets.actor(nets.actor_input(self.probes[i]))
                probs.append(torch.softmax(logits, dim=-1).reshape(-1))
        return torch.cat(probs).numpy().astype(np.float32)
