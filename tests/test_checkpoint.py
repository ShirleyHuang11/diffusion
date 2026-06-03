"""Checkpoint integrity tests."""

import json

import pytest

from reap.checkpoint import (
    CheckpointError,
    checkpoint_name,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "step_000000000100.pt"
    save_checkpoint({"env_step": 100, "data": [1, 2, 3]}, path)
    state = load_checkpoint(path)
    assert state["env_step"] == 100
    assert state["data"] == [1, 2, 3]


def test_corrupted_payload_fails_loudly(tmp_path):
    path = tmp_path / "step_000000000100.pt"
    save_checkpoint({"env_step": 100}, path)
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip a byte mid-file
    path.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError, match="integrity failure"):
        load_checkpoint(path)


def test_missing_sidecar_fails_loudly(tmp_path):
    path = tmp_path / "step_000000000100.pt"
    save_checkpoint({"env_step": 100}, path)
    path.with_suffix(path.suffix + ".sha256").unlink()
    with pytest.raises(CheckpointError, match="sidecar missing"):
        load_checkpoint(path)


def test_unreadable_sidecar_fails_loudly(tmp_path):
    path = tmp_path / "step_000000000100.pt"
    save_checkpoint({"env_step": 100}, path)
    path.with_suffix(path.suffix + ".sha256").write_text("not json")
    with pytest.raises(CheckpointError, match="unreadable digest sidecar"):
        load_checkpoint(path)


def test_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt")


def test_latest_checkpoint_orders_by_step(tmp_path):
    for step in (100, 2000, 30):
        save_checkpoint({"env_step": step}, tmp_path / checkpoint_name(step))
    latest = latest_checkpoint(tmp_path)
    assert load_checkpoint(latest)["env_step"] == 2000


def test_prune_keeps_newest(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint({"env_step": step}, tmp_path / checkpoint_name(step))
    prune_checkpoints(tmp_path, keep_last=2)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == [checkpoint_name(3), checkpoint_name(4)]
