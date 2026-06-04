"""Tiny deterministic cooperative chain MDP — a unit-test fixture only.

Two agents stand on a line of ``length`` cells. The team moves right only
when BOTH agents choose action 1; any other joint action stays. Reaching the
rightmost cell ends the episode with reward 1 (success); otherwise the
episode times out at the horizon. The joint state is the one-hot position.
"""

from __future__ import annotations

import numpy as np

from reap.envs.base import CoopEnv, StepResult


class ChainEnv(CoopEnv):
    def __init__(self, length: int = 5, horizon: int = 12):
        self.length = length
        self.horizon = horizon
        self.num_agents = 2
        self.num_actions = 2  # 0 = stay, 1 = right
        self.local_obs_dim = length
        self.joint_state_dim = length
        self._pos = 0
        self._steps = 0

    def _encode(self):
        one_hot = np.zeros(self.length, dtype=np.float32)
        one_hot[self._pos] = 1.0
        return [one_hot.copy(), one_hot.copy()], one_hot.copy()

    def reset(self, seed: int | None = None):
        self._pos = 0
        self._steps = 0
        return self._encode()

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError(f"expected {self.num_agents} actions, got {len(actions)}")
        if all(int(a) == 1 for a in actions):
            self._pos = min(self._pos + 1, self.length - 1)
        self._steps += 1
        success = self._pos == self.length - 1
        terminated = success
        truncated = (not success) and self._steps >= self.horizon
        local, joint = self._encode()
        return StepResult(
            local_obs=local,
            joint_state=joint,
            extrinsic_reward=1.0 if success else 0.0,
            terminated=terminated,
            truncated=truncated,
            info={"success": success},
        )

    def is_success(self) -> bool:
        return self._pos == self.length - 1

    @property
    def steps_elapsed(self) -> int:
        return self._steps

    def get_state(self) -> dict:
        return {"pos": self._pos, "steps": self._steps}

    def set_state(self, snapshot: dict):
        self._pos = snapshot["pos"]
        self._steps = snapshot["steps"]
        return self._encode()
