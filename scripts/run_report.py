#!/usr/bin/env python
"""Generate a protocol-validated benchmark/hardness report from multi-seed runs.

Usage:
    python scripts/run_report.py --run-name gate_mappo_cramped \
        --expected-seeds 0,1,2 --expected-env-steps 5000000 \
        --threshold 0.8 --out reports/gate_mappo_cramped.json

The generator refuses to bless artifacts that violate the experiment
protocol: seed directories must parse strictly as ``seed<N>``, the configured
seed set must be present exactly (extras rejected unless
``--allow-extra-seeds``), every seed's metrics must exist, be non-empty,
carry the extrinsic success/return fields, and end at exactly the expected
environment-step budget. Conclusions reference the extrinsic channel only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reap.metrics import read_jsonl  # noqa: E402

SEED_DIR_RE = re.compile(r"^seed(\d+)$")


class ProtocolError(ValueError):
    """Raised when run artifacts violate the experiment protocol."""


def discover_seeds(run_root: Path) -> dict[int, Path]:
    if not run_root.is_dir():
        raise ProtocolError(f"run directory not found: {run_root}")
    seeds: dict[int, Path] = {}
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        match = SEED_DIR_RE.match(child.name)
        if not match:
            raise ProtocolError(
                f"malformed seed directory name {child.name!r} under {run_root}; "
                "expected 'seed<N>'"
            )
        seeds[int(match.group(1))] = child
    if not seeds:
        raise ProtocolError(f"no seed directories under {run_root}")
    return seeds


def summarize_seed(seed_dir: Path, expected_env_steps: int) -> dict:
    metrics_path = seed_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise ProtocolError(f"missing metrics file: {metrics_path}")
    records = read_jsonl(metrics_path)
    if not records:
        raise ProtocolError(f"empty metrics file: {metrics_path}")
    final = records[-1]
    extrinsic = final.get("extrinsic", {})
    for field in ("success_rate", "episode_return_mean"):
        if field not in extrinsic:
            raise ProtocolError(
                f"{metrics_path} final record lacks extrinsic field {field!r}"
            )
    if final["env_step"] != expected_env_steps:
        raise ProtocolError(
            f"{seed_dir.name} ended at env_step {final['env_step']}, expected "
            f"exactly {expected_env_steps} (fixed-budget protocol)"
        )
    return {
        "env_step_final": final["env_step"],
        "success_rate_final": extrinsic["success_rate"],
        "episode_return_mean_final": extrinsic["episode_return_mean"],
        "episodes_total": extrinsic.get("episodes"),
        "records": len(records),
    }


def build_report(
    run_name: str,
    runs_dir: str,
    expected_seeds: list[int],
    expected_env_steps: int,
    threshold: float | None,
    allow_extra_seeds: bool,
) -> dict:
    run_root = Path(runs_dir) / run_name
    found = discover_seeds(run_root)

    missing = sorted(set(expected_seeds) - set(found))
    if missing:
        raise ProtocolError(f"missing expected seeds {missing} under {run_root}")
    extra = sorted(set(found) - set(expected_seeds))
    if extra and not allow_extra_seeds:
        raise ProtocolError(
            f"unexpected extra seeds {extra} under {run_root}; pass "
            "--allow-extra-seeds to include runs beyond the registered protocol"
        )

    seeds = {
        f"seed{n}": summarize_seed(found[n], expected_env_steps)
        for n in sorted(expected_seeds)
    }
    rates = [s["success_rate_final"] for s in seeds.values()]
    mean_rate = sum(rates) / len(rates)
    return {
        "run_name": run_name,
        "expected_seeds": sorted(expected_seeds),
        "expected_env_steps": expected_env_steps,
        "seeds": seeds,
        "success_rate_mean": mean_rate,
        "success_rate_min": min(rates),
        "success_rate_max": max(rates),
        "threshold": threshold,
        "gate_passed": (None if threshold is None else bool(mean_rate >= threshold)),
        "extra_seeds_ignored": extra,
        "metric_basis": "extrinsic channel only (task success rate, episode return)",
    }


def write_markdown(report: dict, out: Path) -> None:
    lines = [
        f"# Run report: {report['run_name']}",
        "",
        f"Protocol: seeds {report['expected_seeds']}, fixed budget "
        f"{report['expected_env_steps']} env steps (exact), extrinsic metrics only.",
        "",
        "| seed | final success rate | final return mean | episodes | env steps |",
        "|------|--------------------|-------------------|----------|-----------|",
    ]
    for name, s in report["seeds"].items():
        lines.append(
            f"| {name} | {s['success_rate_final']:.3f} | "
            f"{s['episode_return_mean_final']:.2f} | {s['episodes_total']:.0f} | "
            f"{s['env_step_final']} |"
        )
    lines += ["", f"Mean success rate: **{report['success_rate_mean']:.3f}**"]
    if report["threshold"] is not None:
        verdict = "PASSED" if report["gate_passed"] else "MISSED"
        lines.append(f"Gate (>= {report['threshold']}): **{verdict}**")
        lines.append("")
        lines.append(
            "A missed gate is a documented measurement, not a hidden failure; "
            "see plan thresholds (configurable defaults)."
        )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--expected-seeds", required=True,
        help="comma-separated seed list the protocol registered, e.g. 0,1,2",
    )
    parser.add_argument(
        "--expected-env-steps", type=int, required=True,
        help="exact per-run environment-step budget",
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="success-rate gate; omit for hardness profiles")
    parser.add_argument("--allow-extra-seeds", action="store_true")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    try:
        expected_seeds = [int(s) for s in args.expected_seeds.split(",") if s.strip()]
        if not expected_seeds:
            raise ProtocolError("--expected-seeds must list at least one seed")
        report = build_report(
            args.run_name, args.runs_dir, expected_seeds,
            args.expected_env_steps, args.threshold, args.allow_extra_seeds,
        )
    except (ProtocolError, ValueError) as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    write_markdown(report, out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
