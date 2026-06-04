"""Episode-level storage of joint-state trajectories.

The buffer holds full joint-state trajectories (one array of shape
``(length + 1, state_dim)`` per episode — initial state plus one state per
step) with per-episode metadata: extrinsic return, success flag, length, and
which collection source (ladder rung) produced it. Persistence is a single
``.npz`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TrajectoryBuffer:
    def __init__(self, state_dim: int):
        self.state_dim = int(state_dim)
        self.episodes: list[np.ndarray] = []
        self.returns: list[float] = []
        self.successes: list[bool] = []
        self.sources: list[str] = []

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def success_count(self) -> int:
        return int(sum(self.successes))

    @property
    def total_env_steps(self) -> int:
        return int(sum(len(ep) - 1 for ep in self.episodes))

    def add_episode(
        self, states: np.ndarray, ret: float, success: bool, source: str = "unknown"
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"episode states must have shape (T+1, {self.state_dim}), got {states.shape}"
            )
        if len(states) < 2:
            raise ValueError("an episode needs at least one transition (two states)")
        self.episodes.append(states)
        self.returns.append(float(ret))
        self.successes.append(bool(success))
        self.sources.append(source)

    # -- reporting -----------------------------------------------------------

    def report(self, max_success_examples: int = 3) -> dict:
        """Coverage and success summary used by the warmup gate."""
        all_states = (
            np.concatenate(self.episodes, axis=0)
            if self.episodes
            else np.empty((0, self.state_dim), dtype=np.float32)
        )
        success_examples = []
        for ep, ok in zip(self.episodes, self.successes):
            if ok and len(success_examples) < max_success_examples:
                success_examples.append(ep[-1].tolist())
        per_source: dict[str, dict] = {}
        for source, ok in zip(self.sources, self.successes):
            entry = per_source.setdefault(source, {"episodes": 0, "successes": 0})
            entry["episodes"] += 1
            entry["successes"] += int(ok)
        return {
            "episodes": len(self.episodes),
            "success_count": self.success_count,
            "total_env_steps": self.total_env_steps,
            "return_mean": float(np.mean(self.returns)) if self.returns else 0.0,
            "episode_length_mean": (
                float(np.mean([len(ep) - 1 for ep in self.episodes])) if self.episodes else 0.0
            ),
            "state_coverage": {
                "dim": self.state_dim,
                "mean_abs": float(np.abs(all_states).mean()) if len(all_states) else 0.0,
                "per_dim_std_mean": float(all_states.std(axis=0).mean()) if len(all_states) else 0.0,
                "states_total": int(len(all_states)),
            },
            "success_state_examples": success_examples,
            "per_source": per_source,
        }

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.array([len(ep) for ep in self.episodes], dtype=np.int64)
        flat = (
            np.concatenate(self.episodes, axis=0)
            if self.episodes
            else np.empty((0, self.state_dim), dtype=np.float32)
        )
        np.savez_compressed(
            path,
            state_dim=np.int64(self.state_dim),
            lengths=lengths,
            states=flat,
            returns=np.array(self.returns, dtype=np.float64),
            successes=np.array(self.successes, dtype=bool),
            sources=np.array(self.sources, dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryBuffer":
        data = np.load(Path(path), allow_pickle=True)
        buffer = cls(int(data["state_dim"]))
        offsets = np.concatenate([[0], np.cumsum(data["lengths"])])
        for i in range(len(data["lengths"])):
            buffer.add_episode(
                data["states"][offsets[i] : offsets[i + 1]],
                ret=float(data["returns"][i]),
                success=bool(data["successes"][i]),
                source=str(data["sources"][i]),
            )
        return buffer
