"""Checkpoint save/load with integrity verification.

Checkpoints are torch payloads with a SHA-256 sidecar. Loading verifies the
digest and raises ``CheckpointError`` on any mismatch or unreadable file, so a
corrupted checkpoint can never silently restart a run from scratch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupted, or unreadable."""


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(state: dict, path: str | Path) -> Path:
    """Atomically save ``state`` to ``path`` and write a digest sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    sidecar = {"sha256": _digest(path), "file": path.name}
    path.with_suffix(path.suffix + ".sha256").write_text(json.dumps(sidecar))
    return path


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load ``path`` after verifying its digest sidecar."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    sidecar_path = path.with_suffix(path.suffix + ".sha256")
    if not sidecar_path.is_file():
        raise CheckpointError(f"checkpoint digest sidecar missing: {sidecar_path}")
    try:
        expected = json.loads(sidecar_path.read_text())["sha256"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise CheckpointError(f"unreadable digest sidecar: {sidecar_path}") from exc
    actual = _digest(path)
    if actual != expected:
        raise CheckpointError(
            f"checkpoint integrity failure for {path}: sha256 {actual} != expected {expected}"
        )
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # torch raises various errors on truncated files
        raise CheckpointError(f"failed to deserialize checkpoint {path}: {exc}") from exc


def checkpoint_name(env_step: int) -> str:
    """Zero-padded filename so lexicographic order equals step order."""
    return f"step_{env_step:012d}.pt"


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the highest-step checkpoint in ``directory``, or None."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def prune_checkpoints(directory: str | Path, keep_last: int) -> None:
    """Delete all but the newest ``keep_last`` checkpoints (and sidecars)."""
    directory = Path(directory)
    candidates = sorted(directory.glob("step_*.pt"))
    for stale in candidates[:-keep_last] if keep_last > 0 else []:
        stale.unlink(missing_ok=True)
        stale.with_suffix(stale.suffix + ".sha256").unlink(missing_ok=True)
