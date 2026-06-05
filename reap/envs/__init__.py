"""Environment wrappers exposing a common cooperative multi-agent interface."""

from reap.envs.base import CoopEnv, StepResult

__all__ = ["CoopEnv", "StepResult", "make_env"]


def make_env(env_id: str, **kwargs) -> CoopEnv:
    """Construct an environment by id ("overcooked", ...)."""
    if env_id == "overcooked":
        from reap.envs.overcooked_env import OvercookedSparseEnv

        return OvercookedSparseEnv(**kwargs)
    if env_id == "mpe_spread":
        from reap.envs.mpe_spread import MpeSpreadEnv

        return MpeSpreadEnv(**kwargs)
    raise ValueError(f"unknown env id: {env_id!r}")
