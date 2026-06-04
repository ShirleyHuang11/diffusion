#!/usr/bin/env python
"""First-integration report (AC-9.1): REAP vs MAPPO vs MAPPO+RND.

Usage:
    python scripts/first_integration_report.py \
        --reap-run first_integration_reap_forced \
        --vanilla-run hardness_mappo_forced \
        --rnd-run first_integration_rnd_forced \
        --expected-seeds 0,1,2 --expected-env-steps 5000000 \
        --quality-report reports/teacher_quality_hybrid_forced.json \
        --out reports/first_integration_forced.json

All arms are protocol-validated (exact seeds and budget). Claims reference
the extrinsic channel only. The REAP arm's shaping-gate state is read from
the scope quality artifact and reported explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "run_report", Path(__file__).resolve().parent / "run_report.py"
)
run_report = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_report", run_report)
_spec.loader.exec_module(run_report)


def arm_summary(report: dict) -> dict:
    rates = [s["success_rate_final"] for s in report["seeds"].values()]
    returns = [s["episode_return_mean_final"] for s in report["seeds"].values()]
    return {
        "seeds": report["seeds"],
        "success_rate_mean": sum(rates) / len(rates),
        "success_rate_range": [min(rates), max(rates)],
        "return_mean": sum(returns) / len(returns),
        "return_range": [min(returns), max(returns)],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reap-run", required=True)
    parser.add_argument("--vanilla-run", required=True)
    parser.add_argument("--rnd-run", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.expected_seeds.split(",") if s.strip()]
    try:
        arms = {
            name: arm_summary(run_report.build_report(
                run, args.runs_dir, seeds, args.expected_env_steps, None, False))
            for name, run in (("reap", args.reap_run),
                              ("vanilla_mappo", args.vanilla_run),
                              ("mappo_rnd", args.rnd_run))
        }
    except run_report.ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1

    quality = json.loads(Path(args.quality_report).read_text())
    report = {
        "experiment": "first integration: REAP vs vanilla MAPPO vs MAPPO+RND",
        "task": "Overcooked-AI Forced Coordination, sparse delivery-only reward",
        "protocol": {
            "expected_seeds": sorted(seeds),
            "expected_env_steps": args.expected_env_steps,
            "metric_basis": "extrinsic channel only (success rate, episode return)",
            "arms": {"reap": args.reap_run, "vanilla_mappo": args.vanilla_run,
                     "mappo_rnd": args.rnd_run},
        },
        "reap_arm_shaping": {
            "enabled": bool(quality.get("shaping_enabled", False)),
            "gate_violations": quality.get("gate_violations", []),
            "note": (
                "the REAP arm ran with shaping DISABLED by the scope-specific "
                "teacher-quality gate; its result is therefore expected to "
                "match vanilla MAPPO up to seed noise"
                if not quality.get("shaping_enabled", False)
                else "REAP arm ran with shaping enabled"
            ),
        },
        "arms": arms,
        "outcome_policy": "win, loss, and null results reported with equal prominence",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        "# First integration experiment (AC-9.1)",
        "",
        report["experiment"] + f" — {report['task']}.",
        f"Protocol: seeds {sorted(seeds)}, exact budget {args.expected_env_steps} "
        "env steps per arm; extrinsic metrics only.",
        "",
        f"**REAP arm shaping:** {'ENABLED' if report['reap_arm_shaping']['enabled'] else 'DISABLED by the scope gate'}.",
        f"Gate detail: {'; '.join(report['reap_arm_shaping']['gate_violations']) or 'gates passed'}.",
        "",
        "| arm | success rate mean [range] | return mean [range] |",
        "|-----|---------------------------|----------------------|",
    ]
    for name, a in arms.items():
        lines.append(
            f"| {name} | {a['success_rate_mean']:.3f} "
            f"[{a['success_rate_range'][0]:.3f}, {a['success_rate_range'][1]:.3f}] | "
            f"{a['return_mean']:.2f} [{a['return_range'][0]:.2f}, {a['return_range'][1]:.2f}] |"
        )
    lines += ["", report["reap_arm_shaping"]["note"] + "."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["arms"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
