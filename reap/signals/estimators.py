"""Direct-query propensity and feasibility estimation from the teacher.

Propensity: pin a window's start at the query state, sample continuations
conditioned on the current policy embedding (classifier-free guidance), and
measure the fraction of endpoints inside the goal set. This is the on-policy
success-value estimate.

Feasibility: pin both the start (query state) and the end (a goal state) and
denoise the bridge; each bridge is filtered by transition-consistency (every
adjacent pair must be realizable under the environment dynamics) and weighted
by a model-likelihood proxy, guarding against the success-conditioned
selection effect of backward sampling. Feasibility is used downstream as a
gate only, never as a reward magnitude.
"""

from __future__ import annotations

import abc

import numpy as np
import torch


class TransitionChecker(abc.ABC):
    """Tests whether a state transition is realizable under the dynamics."""

    @abc.abstractmethod
    def realizable(self, state: np.ndarray, next_state: np.ndarray) -> bool: ...


def _pin_batch(value_per_state: np.ndarray, samples_per_state: int) -> torch.Tensor:
    """Tile per-state pin values across their sample groups: (n*M, D)."""
    tensor = torch.as_tensor(np.asarray(value_per_state), dtype=torch.float32)
    return tensor.repeat_interleave(samples_per_state, dim=0)


def estimate_propensity(
    diffusion,
    model,
    dataset,
    states: np.ndarray,  # (n, D) denormalized query states
    policy_embedding: np.ndarray | None,
    success_fn,
    validator,
    samples_per_state: int = 16,
    guidance_scale: float = 2.0,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Per-state success fraction of policy-conditioned forward samples."""
    states = np.asarray(states, dtype=np.float32)
    n = len(states)
    total = n * samples_per_state
    pin = {0: _pin_batch(dataset.normalize(states), samples_per_state)}
    cond = None
    if policy_embedding is not None:
        cond = torch.as_tensor(policy_embedding, dtype=torch.float32).expand(total, -1)
    windows = diffusion.sample(
        model, n=total, pin=pin, cond=cond,
        guidance_scale=guidance_scale, generator=generator,
    )
    endpoints = validator.project(dataset.denormalize(windows[:, -1].numpy()))
    hits = np.array([bool(success_fn(s)) for s in endpoints], dtype=np.float64)
    propensity = hits.reshape(n, samples_per_state).mean(axis=1)
    if np.any((propensity < 0) | (propensity > 1)):
        raise ValueError("propensity outside [0, 1] — estimator invariant violated")
    return propensity


def _likelihood_proxy_weights(
    diffusion, model, windows: torch.Tensor, generator: torch.Generator | None
) -> np.ndarray:
    """Higher weight for windows the teacher reconstructs with lower error.

    Exact DDPM likelihoods are impractical; the denoising error at a fixed
    mid-schedule noise level is a standard monotone proxy. Weights are
    softmax-normalized per call.
    """
    t_mid = diffusion.num_steps // 2
    t = torch.full((windows.shape[0],), t_mid, dtype=torch.long)
    noise = torch.randn(windows.shape, generator=generator)
    noisy = diffusion.q_sample(windows, t, noise)
    with torch.no_grad():
        predicted = model(noisy, t, None)
    errors = ((predicted - noise) ** 2).mean(dim=(1, 2)).numpy()
    logits = -errors
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def estimate_feasibility(
    diffusion,
    model,
    dataset,
    states: np.ndarray,  # (n, D) denormalized query states
    goal_states: np.ndarray,  # (G, D) goal-set examples to bridge toward
    validator,
    checker: TransitionChecker,
    samples_per_state: int = 16,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Per-state likelihood-weighted fraction of dynamics-consistent bridges."""
    states = np.asarray(states, dtype=np.float32)
    goal_states = np.asarray(goal_states, dtype=np.float32)
    if len(goal_states) == 0:
        raise ValueError("feasibility needs at least one goal-set example")
    n = len(states)
    rng_goals = torch.randint(
        0, len(goal_states), (n * samples_per_state,), generator=generator
    ).numpy()
    pin = {
        0: _pin_batch(dataset.normalize(states), samples_per_state),
        model.window - 1: torch.as_tensor(
            dataset.normalize(goal_states[rng_goals]), dtype=torch.float32
        ),
    }
    windows = diffusion.sample(
        model, n=n * samples_per_state, pin=pin, generator=generator
    )

    denorm = dataset.denormalize(windows.numpy())
    flat_shape = (-1, model.state_dim)
    projected = validator.project(denorm.reshape(*flat_shape)).reshape(denorm.shape)
    consistent = np.empty(len(projected), dtype=np.float64)
    for i, window in enumerate(projected):
        consistent[i] = float(
            all(checker.realizable(window[t], window[t + 1]) for t in range(len(window) - 1))
        )

    feasibility = np.empty(n, dtype=np.float64)
    for i in range(n):
        group = slice(i * samples_per_state, (i + 1) * samples_per_state)
        weights = _likelihood_proxy_weights(diffusion, model, windows[group], generator)
        feasibility[i] = float(np.dot(weights, consistent[group]))
    if np.any((feasibility < 0) | (feasibility > 1)):
        raise ValueError("feasibility outside [0, 1] — estimator invariant violated")
    return feasibility
