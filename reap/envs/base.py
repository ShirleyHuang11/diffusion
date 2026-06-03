"""Common cooperative multi-agent environment interface.

All environments expose:

- decentralized per-agent observations (what actors see),
- a centralized joint-state feature vector (what training-time components such
  as the centralized critic and the trajectory model see),
- a single team extrinsic reward per step (the task reward, kept strictly
  separate from any auxiliary signal added later in the training stack),
- a success flag describing whether the episode reached the task goal.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    """One transition as seen by the training loop."""

    local_obs: list[np.ndarray]  # per-agent observation vectors
    joint_state: np.ndarray  # centralized feature vector
    extrinsic_reward: float  # team task reward (sparse)
    terminated: bool  # episode ended by the environment (goal or failure)
    truncated: bool  # episode ended by horizon
    info: dict


class CoopEnv(abc.ABC):
    """Cooperative multi-agent environment with centralized state access."""

    num_agents: int
    num_actions: int
    horizon: int
    local_obs_dim: int
    joint_state_dim: int

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
        """Start an episode; returns (per-agent observations, joint state)."""

    @abc.abstractmethod
    def step(self, actions: list[int]) -> StepResult:
        """Advance one timestep with one discrete action index per agent."""

    @abc.abstractmethod
    def is_success(self) -> bool:
        """Whether the current episode has reached the task goal."""

    @property
    def steps_elapsed(self) -> int:
        raise NotImplementedError

    @property
    def steps_remaining(self) -> int:
        return self.horizon - self.steps_elapsed

    def get_state(self) -> dict:
        """Snapshot simulator + wrapper state for trajectory-faithful resume."""
        raise NotImplementedError(f"{type(self).__name__} does not support state snapshots")

    def set_state(self, snapshot: dict) -> tuple[list[np.ndarray], np.ndarray]:
        """Restore a :meth:`get_state` snapshot; returns (local obs, joint state)."""
        raise NotImplementedError(f"{type(self).__name__} does not support state snapshots")
