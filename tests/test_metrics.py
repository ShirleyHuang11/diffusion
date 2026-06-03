"""Metrics logging tests: channel separation, schema stability, determinism view."""

import pytest

from reap.metrics import MetricsError, MetricsLogger, deterministic_view, read_jsonl


def test_channels_written_separately(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(10, extrinsic={"return": 0.0, "success_rate": 0.0}, shaped={"return": 1.5})
    records = read_jsonl(tmp_path / "metrics.jsonl")
    assert records[0]["extrinsic"]["return"] == 0.0
    assert records[0]["shaped"]["return"] == 1.5
    # extrinsic channel holds only what was logged to it
    assert set(records[0]["extrinsic"]) == {"return", "success_rate"}


def test_unknown_channel_rejected(tmp_path):
    logger = MetricsLogger(tmp_path)
    with pytest.raises(MetricsError, match="unknown metric channels"):
        logger.log(0, bonus={"x": 1.0})


def test_non_numeric_value_rejected(tmp_path):
    logger = MetricsLogger(tmp_path)
    with pytest.raises(MetricsError, match="not numeric"):
        logger.log(0, extrinsic={"return": "high"})


def test_csv_header_prefixes_channels(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(5, extrinsic={"return": 1.0}, intrinsic={"bonus": 0.1})
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    assert "extrinsic/return" in header
    assert "intrinsic/bonus" in header


def test_csv_schema_drift_rejected(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(1, extrinsic={"return": 1.0})
    with pytest.raises(MetricsError, match="not in CSV header"):
        logger.log(2, extrinsic={"return": 1.0, "surprise": 2.0})


def test_deterministic_view_strips_wall_clock(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(1, extrinsic={"return": 1.0}, diag={"loss": 0.5, "wall_time_s": 12.3})
    records = read_jsonl(tmp_path / "metrics.jsonl")
    view = deterministic_view(records)
    assert "wall_clock_s" not in view[0]
    assert "wall_time_s" not in view[0]["diag"]
    assert view[0]["diag"]["loss"] == 0.5
