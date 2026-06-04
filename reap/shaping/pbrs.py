"""Potential-based reward shaping (PBRS) with dynamic potentials.

The shaping term added to the task reward for the transition at time t is

    F_t = beta * (gamma * Phi(s_{t+1}, k_{t+1}) - Phi(s_t, k_t))

where ``k`` is the remaining-horizon index and ``gamma`` must equal the
learner's discount (validated at construction). The potential at any episode
end — goal-reaching termination or horizon timeout — is defined to be zero,
so shaping telescopes cleanly and the return adjustment vanishes at episode
boundaries.

Because potentials here may be refreshed during training (dynamic
potentials, Devlin & Kudenko 2012), both endpoints of any single transition
must be evaluated under the same potential snapshot; the shaper takes the
potential as an explicit argument so callers can pin one snapshot per batch.
"""

from __future__ import annotations

import abc

import numpy as np


class Potential(abc.ABC):
    """A state potential Phi(s, k); ``k`` is the remaining-horizon index."""

    @abc.abstractmethod
    def value(self, state: np.ndarray, steps_remaining: int) -> float:
        """Potential at a (state, remaining-horizon) pair."""

    def values(self, states: np.ndarray, steps_remaining: np.ndarray) -> np.ndarray:
        """Vectorized potential; default loops over :meth:`value`."""
        return np.array(
            [self.value(s, int(k)) for s, k in zip(states, steps_remaining)],
            dtype=np.float64,
        )


class TabularPotential(Potential):
    """Hand-crafted potential from a state-key function and a lookup table.

    Useful for sanity experiments: e.g. on Overcooked, a small table keyed on
    coarse features (held objects, pot fullness) expresses task progress.
    Missing keys default to zero.
    """

    def __init__(self, key_fn, table: dict, time_scaled: bool = False):
        self.key_fn = key_fn
        self.table = dict(table)
        self.time_scaled = time_scaled

    def value(self, state: np.ndarray, steps_remaining: int) -> float:
        base = float(self.table.get(self.key_fn(state), 0.0))
        if self.time_scaled and steps_remaining <= 0:
            return 0.0
        return base


class PotentialShaper:
    """Computes the PBRS adjustment for batches of transitions."""

    def __init__(self, beta: float, gamma: float, learner_gamma: float):
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if abs(gamma - learner_gamma) > 1e-12:
            raise ValueError(
                f"shaping discount ({gamma}) must equal the learner's discount "
                f"({learner_gamma}); a mismatch breaks the invariance argument"
            )
        self.beta = beta
        self.gamma = gamma

    def shape(
        self,
        potential: Potential,
        states: np.ndarray,
        next_states: np.ndarray,
        steps_remaining: np.ndarray,
        episode_ends: np.ndarray,
    ) -> np.ndarray:
        """Shaping terms for a batch of transitions under ONE potential snapshot.

        ``steps_remaining[i]`` is the remaining horizon at ``states[i]``;
        ``episode_ends[i]`` is true when the transition ends the episode
        (goal termination or timeout), forcing Phi(s_{t+1}) = 0.
        """
        states = np.asarray(states)
        next_states = np.asarray(next_states)
        steps_remaining = np.asarray(steps_remaining, dtype=np.int64)
        episode_ends = np.asarray(episode_ends, dtype=bool)
        if not (len(states) == len(next_states) == len(steps_remaining) == len(episode_ends)):
            raise ValueError("transition batch arrays must have equal length")
        if np.any(steps_remaining <= 0):
            raise ValueError("steps_remaining must be positive at the pre-transition state")

        phi_s = potential.values(states, steps_remaining)
        phi_next = potential.values(next_states, steps_remaining - 1)
        phi_next = np.where(episode_ends, 0.0, phi_next)  # episode end: potential is zero
        return (self.beta * (self.gamma * phi_next - phi_s)).astype(np.float64)
