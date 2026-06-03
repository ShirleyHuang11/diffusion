#!/usr/bin/env python
"""Generate a benchmark/hardness report from multi-seed run metrics.

Usage:
    python scripts/run_report.py --run-name gate_mappo_cramped \
        --runs-dir runs --threshold 0.8 --out reports/gate_mappo_cramped.json

Reads ``runs/<run-name>/seed*/metrics.jsonl``, summarizes the final
extrinsic success rate and episode return per seed, evaluates the optional
success-rate threshold, and writes a JSON artifact plus a markdown summary.
Conclusions reference extrinsic metrics only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reap.metrics import read_jsonl  # noqa: E402


def summarize_seed(metrics_path: Path) -> dict:
    records = read_jsonl(metrics_path)
    if not records:
        raise ValueError(f"no metric records in {metrics_path}")
    final = records[-1]
    return {
        "env_step_final": final["env_step"],
        "success_rate_final": final["extrinsic"].get("success_rate"),
        "episode_return_mean_final": final["extrinsic"].get("episode_return_mean"),
        "episodes_total": final["extrinsic"].get("episodes"),
        "records": len(records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--threshold", type=float, default=None,
                        help="success-rate gate; omit for hardness profiles")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    run_root = Path(args.runs_dir) / args.run_name
    seed_dirs = sorted(run_root.glob("seed*"))
    if not seed_dirs:
        print(f"error: no seed directories under {run_root}", file=sys.stderr)
        return 1

    seeds = {}
    for seed_dir in seed_dirs:
        metrics = seed_dir / "metrics.jsonl"
        if not metrics.is_file():
            print(f"error: missing {metrics}", file=sys.stderr)
            return 1
        seeds[seed_dir.name] = summarize_seed(metrics)

    rates = [s["success_rate_final"] for s in seeds.values()]
    mean_rate = sum(rates) / len(rates)
    report = {
        "run_name": args.run_name,
        "seeds": seeds,
        "success_rate_mean": mean_rate,
        "success_rate_min": min(rates),
        "success_rate_max": max(rates),
        "threshold": args.threshold,
        "gate_passed": (None if args.threshold is None else bool(mean_rate >= args.threshold)),
        "metric_basis": "extrinsic channel only (task success rate, episode return)",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    md_lines = [
        f"# Run report: {args.run_name}",
        "",
        "| seed | final success rate | final return mean | episodes | env steps |",
        "|------|--------------------|-------------------|----------|-----------|",
    ]
    for name, s in seeds.items():
        md_lines.append(
            f"| {name} | {s['success_rate_final']:.3f} | "
            f"{s['episode_return_mean_final']:.2f} | {s['episodes_total']:.0f} | "
            f"{s['env_step_final']} |"
        )
    md_lines += ["", f"Mean success rate: **{mean_rate:.3f}**"]
    if args.threshold is not None:
        verdict = "PASSED" if report["gate_passed"] else "MISSED"
        md_lines.append(f"Gate (>= {args.threshold}): **{verdict}**")
        md_lines.append("")
        md_lines.append(
            "A missed gate is a documented measurement, not a hidden failure; "
            "see plan thresholds (configurable defaults)."
        )
    out.with_suffix(".md").write_text("\n".join(md_lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
