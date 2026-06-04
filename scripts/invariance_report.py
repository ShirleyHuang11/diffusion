#!/usr/bin/env python
"""Compare shaped vs. unshaped arms for the PBRS invariance sanity check.

Usage:
    python scripts/invariance_report.py \
        --shaped-run invariance_shaped_rnd_cramped \
        --unshaped-run probe_mappo_rnd_cramped \
        --expected-seeds 0,1,2 --expected-env-steps 5000000 \
        --out reports/invariance_rnd_cramped.json

Both arms are protocol-validated with the same rules as run_report.py. The
verdict is a sanity check, not a statistical claim: the per-seed final
extrinsic ranges of the two arms must overlap (or their means differ by at
most ``--tolerance`` of the larger arm mean).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_report", Path(__file__).resolve().parent / "run_report.py"
)
run_report = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_report", run_report)
_spec.loader.exec_module(run_report)

POTENTIAL_DEFINITION = (
    "overcooked_progress (reap/envs/overcooked_env.py::progress_potential): "
    "0.05*min(ingredients_loaded,6) + 0.30*min(soups_ready,2) + "
    "0.15*min(held_dishes,2) + 0.50*min(held_soups,2); beta=5.0, gamma equal "
    "to the MAPPO discount; potential is zero at every episode end."
)


def arm_stats(report: dict, metric: str) -> dict:
    values = [s[metric] for s in report["seeds"].values()]
    return {
        "values": values,
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def ranges_overlap(a: dict, b: dict) -> bool:
    return a["min"] <= b["max"] and b["min"] <= a["max"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shaped-run", required=True)
    parser.add_argument("--unshaped-run", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="allowed relative mean gap when ranges do not overlap")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    expected_seeds = [int(s) for s in args.expected_seeds.split(",") if s.strip()]
    try:
        arms = {
            "shaped": run_report.build_report(
                args.shaped_run, args.runs_dir, expected_seeds,
                args.expected_env_steps, None, False,
            ),
            "unshaped": run_report.build_report(
                args.unshaped_run, args.runs_dir, expected_seeds,
                args.expected_env_steps, None, False,
            ),
        }
    except run_report.ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1

    comparison = {}
    overlapping = {}
    for metric in ("success_rate_final", "episode_return_mean_final"):
        shaped = arm_stats(arms["shaped"], metric)
        unshaped = arm_stats(arms["unshaped"], metric)
        overlap = ranges_overlap(shaped, unshaped)
        if not overlap:
            larger = max(abs(shaped["mean"]), abs(unshaped["mean"]), 1e-9)
            overlap = abs(shaped["mean"] - unshaped["mean"]) / larger <= args.tolerance
        comparison[metric] = {"shaped": shaped, "unshaped": unshaped, "comparable": overlap}
        overlapping[metric] = overlap

    report = {
        "check": "PBRS invariance sanity check (not a statistical claim)",
        "potential_definition": POTENTIAL_DEFINITION,
        "protocol": {
            "expected_seeds": sorted(expected_seeds),
            "expected_env_steps": args.expected_env_steps,
            "shaped_run": args.shaped_run,
            "unshaped_run": args.unshaped_run,
            "tolerance": args.tolerance,
        },
        "comparison": comparison,
        "invariance_holds": bool(all(overlapping.values())),
        "metric_basis": "extrinsic channel only (task success rate, episode return)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        "# PBRS invariance sanity check",
        "",
        report["check"] + ".",
        "",
        f"Potential: {POTENTIAL_DEFINITION}",
        f"Protocol: seeds {sorted(expected_seeds)}, exact budget "
        f"{args.expected_env_steps} env steps; arms `{args.shaped_run}` (shaped) vs. "
        f"`{args.unshaped_run}` (unshaped).",
        "",
        "| metric | shaped (mean [min, max]) | unshaped (mean [min, max]) | comparable |",
        "|--------|--------------------------|----------------------------|------------|",
    ]
    for metric, c in comparison.items():
        s, u = c["shaped"], c["unshaped"]
        lines.append(
            f"| {metric} | {s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}] | "
            f"{u['mean']:.3f} [{u['min']:.3f}, {u['max']:.3f}] | {c['comparable']} |"
        )
    lines += ["", f"**Invariance holds: {report['invariance_holds']}**"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
