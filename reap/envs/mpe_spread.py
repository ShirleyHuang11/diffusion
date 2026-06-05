"""In-repo MPE Spread (Cooperative Navigation, Lowe et al. 2017).

A faithful reimplementation of the ``simple_spread`` scenario from the
original ``openai/multiagent-particle-envs`` codebase (the EPyMARL benchmark
fork preserves the same scenario), exposed through the shared
:class:`~reap.envs.base.CoopEnv` interface so COMA/QMIX/MAPPO run on it with
the exact trunk, logging, and protocol machinery used everywhere else.

Faithfulness notes (verified against the reference source):

- Physics: ``dt=0.1``, ``damping=0.25``, contact force ``1e2`` with margin
  ``1e-3`` between colliding agent pairs, force sensitivity ``5.0`` (default
  ``accel``), no ``max_speed`` cap, unit mass.
- Reset: agent and landmark positions uniform in ``[-1, +1]^2``, zero
  velocities.
- Observation (per agent, 18-dim for N=3): ``[own vel(2), own pos(2),
  landmark rel pos (2N), other-agent rel pos (2(N-1)), comm (2(N-1)) zeros]``
  in the reference ordering. Agents are silent, so the comm block is zero.
- Reward (per agent): ``-sum over landmarks of min-agent distance`` minus 1
  per colliding agent **including the agent itself** — the reference loop has
  no self-exclusion and ``dist(agent, agent)=0 < 2*size`` always holds, so
  every agent receives a constant -1 per step. This quirk is preserved
  deliberately: published EPyMARL returns include it, and the sanity check
  compares against those numbers.
- Team extrinsic reward: per-agent rewards summed over agents (the EPyMARL
  ``common_reward`` convention used by the published benchmark returns).
- Episodes end only by horizon (default 25, the benchmark episode limit).

The centralized joint state is the concatenation of the per-agent
observations (the EPyMARL state convention for MPE).
"""

from __future__ import annotations

import numpy as np

from reap.envs.base import CoopEnv, StepResult

DT = 0.1
DAMPING = 0.25
CONTACT_FORCE = 1e2
CONTACT_MARGIN = 1e-3
SENSITIVITY = 5.0  # default accel multiplier on the unit control force
AGENT_SIZE = 0.15
LANDMARK_SIZE = 0.05
DIM_C = 2  # communication channel width (always zero: agents are silent)

# action index -> control force direction (0 = noop)
ACTION_FORCES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
    dtype=np.float64,
)


class MpeSpreadEnv(CoopEnv):
    """N-agent cooperative navigation with the reference MPE dynamics."""

    def __init__(self, num_agents: int = 3, horizon: int = 25,
                 coverage_threshold: float = AGENT_SIZE + LANDMARK_SIZE):
        if num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.num_agents = int(num_agents)
        self.num_landmarks = int(num_agents)
        self.horizon = int(horizon)
        self.num_actions = len(ACTION_FORCES)
        self.coverage_threshold = float(coverage_threshold)

        n = self.num_agents
        self.local_obs_dim = 4 + 2 * self.num_landmarks + 4 * (n - 1)
        self.joint_state_dim = n * self.local_obs_dim

        self._rng = np.random.default_rng(0)
        self._pos = np.zeros((n, 2))
        self._vel = np.zeros((n, 2))
        self._landmarks = np.zeros((self.num_landmarks, 2))
        self._steps = 0
        self._all_covered = False

    # -- observation/state ---------------------------------------------------

    def _observe(self) -> tuple[list[np.ndarray], np.ndarray]:
        local: list[np.ndarray] = []
        for i in range(self.num_agents):
            rel_landmarks = (self._landmarks - self._pos[i]).ravel()
            others = [j for j in range(self.num_agents) if j != i]
            rel_others = (self._pos[others] - self._pos[i]).ravel()
            comm = np.zeros(DIM_C * len(others))
            local.append(
                np.concatenate(
                    [self._vel[i], self._pos[i], rel_landmarks, rel_others, comm]
                ).astype(np.float32)
            )
        return local, np.concatenate(local)

    # -- CoopEnv interface -----------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self._rng.uniform(-1.0, 1.0, size=(self.num_agents, 2))
        self._vel = np.zeros((self.num_agents, 2))
        self._landmarks = self._rng.uniform(-1.0, 1.0, size=(self.num_landmarks, 2))
        self._steps = 0
        self._all_covered = False
        return self._observe()

    def _validate_action(self, action) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(
                f"action must be an integer index, got {action!r} ({type(action).__name__})"
            )
        index = int(action)
        if not 0 <= index < self.num_actions:
            raise ValueError(f"action index {index} out of range [0, {self.num_actions})")
        return index

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != self.num_agents:
            raise ValueError(f"expected {self.num_agents} actions, got {len(actions)}")
        indices = [self._validate_action(a) for a in actions]

        # control forces, then pairwise contact forces between colliding agents
        forces = ACTION_FORCES[indices] * SENSITIVITY
        dist_min = 2 * AGENT_SIZE
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                delta = self._pos[i] - self._pos[j]
                dist = float(np.sqrt(np.sum(np.square(delta))))
                k = CONTACT_MARGIN
                penetration = np.logaddexp(0.0, -(dist - dist_min) / k) * k
                force = CONTACT_FORCE * delta / dist * penetration if dist > 0 else 0.0
                forces[i] = forces[i] + force
                forces[j] = forces[j] - force

        self._vel = self._vel * (1 - DAMPING) + forces * DT  # unit mass
        self._pos = self._pos + self._vel * DT
        self._steps += 1

        # per-agent reward: shared distance term + collision penalties
        # (self-collision included, matching the reference implementation)
        sum_min_dists = 0.0
        covered = 0
        for lm in self._landmarks:
            dists = np.sqrt(np.sum(np.square(self._pos - lm), axis=1))
            nearest = float(dists.min())
            sum_min_dists += nearest
            covered += int(nearest < self.coverage_threshold)
        pair_dists = np.sqrt(
            np.sum(np.square(self._pos[:, None, :] - self._pos[None, :, :]), axis=-1)
        )
        collisions_per_agent = (pair_dists < dist_min).sum(axis=1)  # diagonal counts self
        per_agent = -sum_min_dists - collisions_per_agent.astype(np.float64)
        extrinsic = float(per_agent.sum())

        if covered == self.num_landmarks:
            self._all_covered = True

        local, joint = self._observe()
        truncated = self._steps >= self.horizon
        return StepResult(
            local_obs=local,
            joint_state=joint,
            extrinsic_reward=extrinsic,
            terminated=False,  # MPE episodes end only by horizon
            truncated=truncated,
            info={
                "success": self._all_covered,
                "occupied_landmarks": covered,
                "sum_min_dists": sum_min_dists,
                "collisions": int(collisions_per_agent.sum() - self.num_agents),
            },
        )

    def is_success(self) -> bool:
        """All landmarks simultaneously covered at some step (diagnostic only).

        MPE Spread has no terminal goal; sanity verdicts use episode returns,
        never this flag.
        """
        return self._all_covered

    @property
    def steps_elapsed(self) -> int:
        return self._steps

    # -- state serialization (trajectory-faithful resume) ----------------------

    def get_state(self) -> dict:
        return {
            "pos": self._pos.copy(),
            "vel": self._vel.copy(),
            "landmarks": self._landmarks.copy(),
            "steps": self._steps,
            "all_covered": self._all_covered,
            "rng_state": self._rng.bit_generator.state,
        }

    def set_state(self, snapshot: dict) -> tuple[list[np.ndarray], np.ndarray]:
        self._pos = np.array(snapshot["pos"], dtype=np.float64)
        self._vel = np.array(snapshot["vel"], dtype=np.float64)
        self._landmarks = np.array(snapshot["landmarks"], dtype=np.float64)
        self._steps = int(snapshot["steps"])
        self._all_covered = bool(snapshot["all_covered"])
        self._rng.bit_generator.state = snapshot["rng_state"]
        return self._observe()
