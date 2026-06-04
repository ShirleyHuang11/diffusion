"""Sparse-reward Overcooked-AI wrapper.

Exposes Overcooked through the :class:`~reap.envs.base.CoopEnv` interface with
a strictly sparse task reward: the extrinsic reward is the native *sparse*
(delivery) reward only. The native dense ("shaped") reward is never added to
the extrinsic channel; it is surfaced in ``info`` purely so tests can assert
it leaks nowhere.

Success is defined as at least one soup delivered during the episode.
"""

from __future__ import annotations

import numpy as np

from reap.envs.base import CoopEnv, StepResult

# Friendly layout name -> candidate names across overcooked_ai versions.
LAYOUTS = {
    "cramped_room": ("cramped_room",),
    "asymmetric_advantages": ("asymmetric_advantages",),
    "coordination_ring": ("coordination_ring",),
    "forced_coordination": ("forced_coordination",),
    "counter_circuit": ("counter_circuit_o_1order", "counter_circuit"),
}

ENCODINGS = ("features", "lossless")


class OvercookedSparseEnv(CoopEnv):
    """Two-agent Overcooked with delivery-only extrinsic reward."""

    def __init__(self, layout: str, horizon: int = 400, encoding: str = "features"):
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; choose from {sorted(LAYOUTS)}")
        if encoding not in ENCODINGS:
            raise ValueError(f"unknown encoding {encoding!r}; choose from {ENCODINGS}")

        from overcooked_ai_py.mdp.actions import Action
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        self._Action = Action
        self.layout = layout
        self.encoding = encoding
        self.horizon = horizon
        self.num_agents = 2
        self.num_actions = len(Action.ALL_ACTIONS)

        self._mdp = None
        last_err: Exception | None = None
        for candidate in LAYOUTS[layout]:
            try:
                self._mdp = OvercookedGridworld.from_layout_name(candidate)
                self.native_layout_name = candidate
                break
            except Exception as exc:  # missing layout file in this version
                last_err = exc
        if self._mdp is None:
            raise ValueError(f"no usable layout file for {layout!r}: {last_err}")

        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=horizon, info_level=0)
        self._mlam = None  # built lazily; only the "features" encoding needs it

        self._env.reset()
        if encoding == "lossless":
            self.lossless_shape = tuple(
                np.asarray(self._mdp.lossless_state_encoding(self._env.state)[0]).shape
            )
        probe_local, probe_joint = self._encode(self._env.state)
        self.local_obs_dim = int(probe_local[0].size)
        self.joint_state_dim = int(probe_joint.size)

        self._deliveries = 0
        self._steps = 0

    # -- encoding ---------------------------------------------------------

    def _get_mlam(self):
        if self._mlam is None:
            self._mlam = self._env.mlam  # planner; computed/cached per layout
        return self._mlam

    def _encode(self, state) -> tuple[list[np.ndarray], np.ndarray]:
        """Per-agent observation vectors and the centralized joint state."""
        if self.encoding == "features":
            feats = self._mdp.featurize_state(state, self._get_mlam())
            local = [np.asarray(f, dtype=np.float32).ravel() for f in feats]
            joint = np.concatenate(local)
        else:
            grids = self._mdp.lossless_state_encoding(state)
            local = [np.asarray(g, dtype=np.float32).ravel() for g in grids]
            joint = local[0].copy()  # agent-0 grid view is full-information
        return local, joint

    # -- CoopEnv interface -------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
        # Overcooked layouts have deterministic start states; seed is accepted
        # for interface compatibility.
        self._env.reset()
        self._deliveries = 0
        self._steps = 0
        return self._encode(self._env.state)

    def _validate_action(self, action) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(
                f"action must be an integer index, got {action!r} ({type(action).__name__})"
            )
        index = int(action)
        if not 0 <= index < self.num_actions:
            raise ValueError(
                f"action index {index} out of range [0, {self.num_actions})"
            )
        return index

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != self.num_agents:
            raise ValueError(f"expected {self.num_agents} actions, got {len(actions)}")
        joint_action = tuple(
            self._Action.INDEX_TO_ACTION[self._validate_action(a)] for a in actions
        )
        next_state, sparse_reward, done, info = self._env.step(joint_action)
        self._steps += 1

        extrinsic = float(sparse_reward)
        if extrinsic > 0:
            self._deliveries += int(round(extrinsic / 20.0)) or 1

        local, joint = self._encode(next_state)
        truncated = bool(done) and self._steps >= self.horizon
        terminated = bool(done) and not truncated

        native_shaped = float(sum(info.get("shaped_r_by_agent", []) or [0.0]))
        native_sparse = float(sum(info.get("sparse_r_by_agent", []) or [0.0]))
        return StepResult(
            local_obs=local,
            joint_state=joint,
            extrinsic_reward=extrinsic,
            terminated=terminated,
            truncated=truncated,
            info={
                "deliveries": self._deliveries,
                "success": self.is_success(),
                # surfaced ONLY for leak tests; never a reward input
                "debug_native_shaped_reward": native_shaped,
                "debug_native_sparse_reward": native_sparse,
            },
        )

    def is_success(self) -> bool:
        return self._deliveries >= 1

    @property
    def steps_elapsed(self) -> int:
        return self._steps

    def progress_potential(self) -> float:
        """Hand-crafted, bounded task-progress proxy from the native state.

        Counts ingredients loaded into soups, ready soups, and held
        dishes/soups — a monotone "closer to a delivery" signal used only as
        a sanity-check shaping potential, never as a measurement. Weights are
        documented in the invariance report. Range: [0, ~1.5].
        """
        state = self._env.state
        ingredients_loaded = 0
        soups_ready = 0
        for obj in state.objects.values():
            if obj.name == "soup":
                if getattr(obj, "is_ready", False):
                    soups_ready += 1
                else:
                    ingredients_loaded += len(getattr(obj, "ingredients", []) or [])
        held_dish = 0
        held_soup = 0
        for player in state.players:
            held = player.held_object
            if held is not None:
                if held.name == "dish":
                    held_dish += 1
                elif held.name == "soup":
                    held_soup += 1
        return (
            0.05 * min(ingredients_loaded, 6)
            + 0.30 * min(soups_ready, 2)
            + 0.15 * min(held_dish, 2)
            + 0.50 * min(held_soup, 2)
        )

    # -- state serialization (trajectory-faithful resume) -------------------

    def get_state(self) -> dict:
        """Snapshot the simulator and wrapper counters for exact resume."""
        import copy

        return {
            "native_state": copy.deepcopy(self._env.state),
            "steps": self._steps,
            "deliveries": self._deliveries,
        }

    def set_state(self, snapshot: dict) -> tuple[list[np.ndarray], np.ndarray]:
        """Restore a snapshot taken by :meth:`get_state` mid-episode or not."""
        import copy

        self._env.state = copy.deepcopy(snapshot["native_state"])
        self._steps = int(snapshot["steps"])
        self._deliveries = int(snapshot["deliveries"])
        return self._encode(self._env.state)
