"""Structured metrics logging with strict reward-channel separation.

Metric records carry separate channels so task-reward results can never be
conflated with auxiliary reward signals:

- ``extrinsic``: environment task reward and success statistics only
- ``shaped``: potential-based shaping quantities
- ``intrinsic``: exploration-bonus quantities (RND, counts, ...)
- ``diag``: everything else (losses, timings, memory, ...)

The logger writes JSONL (one record per line) and a flat CSV mirror with
channel-prefixed column names (``extrinsic/return``, ``shaped/return``, ...).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Mapping, Optional

CHANNELS = ("extrinsic", "shaped", "intrinsic", "diag")


class MetricsError(ValueError):
    """Raised on invalid metric channel usage."""


class MetricsLogger:
    """Append-only metrics writer producing JSONL and CSV artifacts."""

    def __init__(self, out_dir: str | Path, jsonl: bool = True, csv_enabled: bool = True):
        if not (jsonl or csv_enabled):
            raise MetricsError("at least one of jsonl/csv must be enabled")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.out_dir / "metrics.jsonl" if jsonl else None
        self._csv_path = self.out_dir / "metrics.csv" if csv_enabled else None
        self._csv_columns: Optional[list[str]] = None
        self._start_time = time.monotonic()

    def log(self, env_step: int, **channels: Mapping[str, float]) -> dict:
        """Log one record at ``env_step``; kwargs must be channel names."""
        unknown = set(channels) - set(CHANNELS)
        if unknown:
            raise MetricsError(
                f"unknown metric channels {sorted(unknown)}; allowed: {list(CHANNELS)}"
            )
        record: dict = {
            "env_step": int(env_step),
            "wall_clock_s": round(time.monotonic() - self._start_time, 3),
        }
        for channel in CHANNELS:
            values = channels.get(channel) or {}
            record[channel] = {k: _to_float(channel, k, v) for k, v in values.items()}

        if self._jsonl_path is not None:
            with self._jsonl_path.open("a") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self._csv_path is not None:
            self._write_csv_row(record)
        return record

    def _write_csv_row(self, record: dict) -> None:
        flat: dict[str, float] = {
            "env_step": record["env_step"],
            "wall_clock_s": record["wall_clock_s"],
        }
        for channel in CHANNELS:
            for key, value in record[channel].items():
                flat[f"{channel}/{key}"] = value

        if self._csv_columns is None:
            if self._csv_path.exists():  # resuming: keep the existing header
                with self._csv_path.open() as fh:
                    header = fh.readline().strip()
                self._csv_columns = header.split(",") if header else sorted(flat)
            else:
                self._csv_columns = sorted(flat)
                with self._csv_path.open("w", newline="") as fh:
                    csv.writer(fh).writerow(self._csv_columns)

        new_cols = set(flat) - set(self._csv_columns)
        if new_cols:
            raise MetricsError(
                f"metric columns {sorted(new_cols)} not in CSV header; "
                "log a consistent schema from the first record onward"
            )
        with self._csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([flat.get(col, "") for col in self._csv_columns])


def _to_float(channel: str, key: str, value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"metric {channel}/{key} is not numeric: {value!r}") from exc


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a metrics.jsonl file back into a list of records."""
    records = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def deterministic_view(records: list[dict]) -> list[dict]:
    """Strip wall-clock timing so same-seed runs can be compared for equality.

    Wall-clock and memory readings vary between runs by nature; every other
    field (env steps, rewards, losses, ...) must match exactly under the same
    seed and config.
    """
    nondeterministic_diag_prefixes = ("time_", "wall_", "gpu_mem", "cpu_mem")
    view = []
    for record in records:
        clean = {k: v for k, v in record.items() if k != "wall_clock_s"}
        if "diag" in clean:
            clean["diag"] = {
                k: v
                for k, v in clean["diag"].items()
                if not k.startswith(nondeterministic_diag_prefixes)
            }
        view.append(clean)
    return view
