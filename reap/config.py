"""Config loading and validation.

Configs are YAML files validated against typed schemas. Required fields with no
default must be present; unknown keys are rejected so typos fail loudly instead
of silently falling back to defaults.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


_REQUIRED = object()  # sentinel: field must be provided explicitly


@dataclass
class RunConfig:
    """Run-level settings shared by every entrypoint."""

    name: str = _REQUIRED  # type: ignore[assignment]
    seed: int = _REQUIRED  # type: ignore[assignment]
    mode: str = _REQUIRED  # type: ignore[assignment]  # "smoke" | "paper"
    out_dir: str = _REQUIRED  # type: ignore[assignment]
    max_wall_clock_minutes: float = _REQUIRED  # type: ignore[assignment]
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    deterministic_torch: bool = True

    def validate(self) -> None:
        if self.mode not in ("smoke", "paper"):
            raise ConfigError(f"run.mode must be 'smoke' or 'paper', got {self.mode!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ConfigError(f"run.seed must be an integer, got {self.seed!r}")
        if self.max_wall_clock_minutes <= 0:
            raise ConfigError("run.max_wall_clock_minutes must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError(f"run.device must be auto/cpu/cuda, got {self.device!r}")


@dataclass
class EnvConfig:
    """Environment selection and episode settings."""

    id: str = _REQUIRED  # type: ignore[assignment]  # e.g. "overcooked"
    layout: str = ""  # overcooked layout name; empty for non-overcooked envs
    horizon: int = 400
    num_agents: int = 2
    encoding: str = "features"  # "features" (compact vectors) | "lossless" (grid)

    def validate(self) -> None:
        if self.id == "overcooked" and not self.layout:
            raise ConfigError("env.layout is required when env.id is 'overcooked'")
        if self.encoding not in ("features", "lossless"):
            raise ConfigError(f"env.encoding must be features/lossless, got {self.encoding!r}")
        if self.horizon <= 0:
            raise ConfigError("env.horizon must be positive")
        if self.num_agents < 1:
            raise ConfigError("env.num_agents must be >= 1")


@dataclass
class AlgoConfig:
    """Algorithm selection; algorithm-specific keys live in `params`."""

    name: str = _REQUIRED  # type: ignore[assignment]  # e.g. "random", "mappo"
    total_env_steps: int = _REQUIRED  # type: ignore[assignment]
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.total_env_steps <= 0:
            raise ConfigError("algo.total_env_steps must be positive")


@dataclass
class LoggingConfig:
    """Metrics logging cadence and sinks."""

    interval_env_steps: int = 1000
    csv: bool = True
    jsonl: bool = True

    def validate(self) -> None:
        if self.interval_env_steps <= 0:
            raise ConfigError("logging.interval_env_steps must be positive")
        if not (self.csv or self.jsonl):
            raise ConfigError("logging requires at least one of csv/jsonl enabled")


@dataclass
class CheckpointConfig:
    """Checkpoint cadence; resume is requested via the CLI, not the config."""

    interval_env_steps: int = 50_000
    keep_last: int = 2

    def validate(self) -> None:
        if self.interval_env_steps <= 0:
            raise ConfigError("checkpoint.interval_env_steps must be positive")
        if self.keep_last < 1:
            raise ConfigError("checkpoint.keep_last must be >= 1")


@dataclass
class Config:
    """Top-level config: run + env + algo + logging + checkpoint."""

    run: RunConfig = _REQUIRED  # type: ignore[assignment]
    env: EnvConfig = _REQUIRED  # type: ignore[assignment]
    algo: AlgoConfig = _REQUIRED  # type: ignore[assignment]
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def validate(self) -> None:
        self.run.validate()
        self.env.validate()
        self.algo.validate()
        self.logging.validate()
        self.checkpoint.validate()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build_section(cls: type, raw: Any, path: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"section '{path}' must be a mapping, got {type(raw).__name__}")
    raw = copy.deepcopy(raw)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - field_names
    if unknown:
        raise ConfigError(f"unknown keys in '{path}': {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        key = f.name
        if key in raw:
            value = raw[key]
            if dataclasses.is_dataclass(f.type) if isinstance(f.type, type) else False:
                value = _build_section(f.type, value, f"{path}.{key}")
            kwargs[key] = value
        else:
            has_default = f.default is not dataclasses.MISSING and f.default is not _REQUIRED
            has_factory = f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            if not (has_default or has_factory):
                raise ConfigError(f"missing required key '{path}.{key}'")
    return cls(**kwargs)


_SECTION_TYPES = {
    "run": RunConfig,
    "env": EnvConfig,
    "algo": AlgoConfig,
    "logging": LoggingConfig,
    "checkpoint": CheckpointConfig,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a typed Config."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ConfigError(f"unknown top-level sections: {sorted(unknown)}")

    kwargs = {}
    for name, cls in _SECTION_TYPES.items():
        if name in raw:
            kwargs[name] = _build_section(cls, raw[name], name)
        elif name in ("logging", "checkpoint"):
            kwargs[name] = cls()
        else:
            raise ConfigError(f"missing required section '{name}'")

    cfg = Config(**kwargs)
    cfg.validate()
    return cfg
