#!/usr/bin/env python
"""First-integration report: REAP vs MAPPO vs MAPPO+RND.

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


T_975_DF2 = 4.302652729911275  # two-sided 95% t critical value for n=3 seeds


def mean_ci(values: list[float]) -> dict:
    import math

    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "ci95": [mean, mean], "n": n}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = T_975_DF2 * math.sqrt(var / n) if n == 3 else None
    if half is None:  # generic fallback (not expected under the protocol)
        half = 2.0 * math.sqrt(var / n)
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": n}


def arm_summary(report: dict) -> dict:
    rates = [s["success_rate_final"] for s in report["seeds"].values()]
    returns = [s["episode_return_mean_final"] for s in report["seeds"].values()]
    return {
        "seeds": report["seeds"],
        "success_rate": mean_ci(rates),
        "success_rate_mean": sum(rates) / len(rates),
        "success_rate_range": [min(rates), max(rates)],
        "return": mean_ci(returns),
        "return_mean": sum(returns) / len(returns),
        "return_range": [min(returns), max(returns)],
    }


def reap_evidence(runs_dir: str, reap_run: str, seeds: list[int]) -> dict:
    """Per-seed shaping event and runtime evidence for the REAP arm."""
    evidence = {}
    for seed in seeds:
        seed_dir = Path(runs_dir) / reap_run / f"seed{seed}"
        entry: dict = {
            "metrics_path": str(seed_dir / "metrics.jsonl"),
            "shaping_events_path": str(seed_dir / "shaping_events.jsonl"),
        }
        events_path = seed_dir / "shaping_events.jsonl"
        if events_path.is_file():
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            gates = [e for e in events if e["type"] == "gate"]
            refreshes = [e for e in events if e["type"] == "refresh"]
            entry["gate"] = ({"enabled": gates[0]["enabled"], "reason": gates[0]["reason"]}
                             if gates else None)
            # the event log is append-only across SLURM preemption restarts:
            # each restart re-runs the seed from scratch and appends a fresh
            # gate event, so gate_records-1 = restart count and the
            # cumulative refresh count spans aborted segments; the final
            # clean 5M-step segment is characterized by the metrics file
            entry["gate_records"] = len(gates)
            entry["restarts_detected"] = max(0, len(gates) - 1)
            entry["refresh_count_cumulative_across_segments"] = len(refreshes)
            cals = [e["calibration"] for e in refreshes if "calibration" in e]
            if cals:
                entry["calibration_actions"] = [c["action"] for c in cals]
                entry["calibration_ece_last"] = cals[-1]["raw_ece"]
                entry["calibration_brier_last"] = cals[-1]["brier"]
        metrics_path = seed_dir / "metrics.jsonl"
        if metrics_path.is_file():
            last = json.loads(metrics_path.read_text().splitlines()[-1])
            entry["wall_time_s"] = last.get("diag", {}).get("wall_time_s")
            entry["gpu_mem_mb"] = last.get("diag", {}).get("gpu_mem_mb")
            entry["final_segment_refresh_count"] = last.get("diag", {}).get("refresh_count")
            entry["final_segment_snapshot_id"] = last.get("shaped", {}).get("snapshot_id")
        evidence[f"seed{seed}"] = entry
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reap-run", required=True)
    parser.add_argument("--vanilla-run", required=True)
    parser.add_argument("--rnd-run", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--calibration-report", default=None)
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
        "evidence": {
            "quality_report": args.quality_report,
            "calibration_report": args.calibration_report,
            "warmup_report": args.quality_report.replace("teacher_quality", "warmup_buffer"),
            "potential_table": args.quality_report.replace("teacher_quality", "potential_table"),
            "distill_fidelity": args.quality_report.replace("teacher_quality", "distill_fidelity"),
            "pipeline_summary": args.quality_report.replace(
                "teacher_quality", "teacher_pipeline_summary"),
            "per_arm_metrics": {
                name: [f"{args.runs_dir}/{run}/seed{s}/metrics.jsonl" for s in seeds]
                for name, run in (("reap", args.reap_run),
                                  ("vanilla_mappo", args.vanilla_run),
                                  ("mappo_rnd", args.rnd_run))
            },
            "reap_per_seed": reap_evidence(args.runs_dir, args.reap_run, seeds),
            "restart_accounting_note": (
                "shaping_events.jsonl is append-only across SLURM preemption "
                "restarts; each restart is a clean from-scratch rerun of the "
                "seed, appending a fresh gate event, so gate_records-1 = "
                "restarts_detected and refresh_count_cumulative_across_segments "
                "spans aborted segments, while final_segment_refresh_count "
                "(from the final metrics row) characterizes the clean "
                "5,000,000-step segment that produced the reported results"
            ),
        },
        "outcome_policy": "win, loss, and null results reported with equal prominence",
    }

    # the conclusion lives IN the artifact, stated plainly (negative included)
    reap_sr = arms["reap"]["success_rate"]["mean"]
    vanilla_sr = arms["vanilla_mappo"]["success_rate"]["mean"]
    rnd_sr = arms["mappo_rnd"]["success_rate"]["mean"]
    if quality.get("shaping_enabled", False):
        h4_supported = reap_sr > rnd_sr
        report["conclusion"] = {
            "summary": (
                f"REAP (shaping enabled) final extrinsic success {reap_sr:.3f}, "
                f"vanilla MAPPO {vanilla_sr:.3f}, MAPPO+RND {rnd_sr:.3f}."
            ),
            "h4_beats_generic_novelty": h4_supported,
            "interpretation": (
                "H4 is supported in this configuration." if h4_supported else
                "H4 is NOT supported in this configuration: the enabled, "
                "calibrated REAP signal did not outperform generic novelty "
                "(RND). The calibrated propensity honestly reads near-zero for "
                "a policy that never succeeds, so the shaping signal vanishes "
                "exactly where exploration is the bottleneck — a first-class "
                "negative result reported with equal prominence."
            ),
        }
    else:
        report["conclusion"] = {
            "summary": (
                f"REAP arm ran shaping-DISABLED by the scope gate; success "
                f"{reap_sr:.3f} vs vanilla {vanilla_sr:.3f} and RND {rnd_sr:.3f}."
            ),
            "h4_beats_generic_novelty": None,
            "interpretation": "no shaped comparison available for this scope",
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        "# First integration experiment",
        "",
        report["experiment"] + f" — {report['task']}.",
        f"Protocol: seeds {sorted(seeds)}, exact budget {args.expected_env_steps} "
        "env steps per arm; extrinsic metrics only.",
        "",
        f"**REAP arm shaping:** {'ENABLED' if report['reap_arm_shaping']['enabled'] else 'DISABLED by the scope gate'}.",
        f"Gate detail: {'; '.join(report['reap_arm_shaping']['gate_violations']) or 'gates passed'}.",
        "",
        "| arm | success rate mean ± 95% CI | return mean ± 95% CI |",
        "|-----|-----------------------------|------------------------|",
    ]
    for name, a in arms.items():
        sr, rt = a["success_rate"], a["return"]
        lines.append(
            f"| {name} | {sr['mean']:.3f} [{sr['ci95'][0]:.3f}, {sr['ci95'][1]:.3f}] | "
            f"{rt['mean']:.2f} [{rt['ci95'][0]:.2f}, {rt['ci95'][1]:.2f}] |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        report["conclusion"]["summary"],
        "",
        report["conclusion"]["interpretation"],
        "",
        f"Evidence: quality report `{args.quality_report}`"
        + (f", calibration report `{args.calibration_report}`" if args.calibration_report else "")
        + "; warmup/potential-table/fidelity/pipeline-summary paths, per-arm "
        "metrics paths, per-seed shaping events and wall-clock/GPU-memory in "
        "the JSON artifact. Restart accounting: the shaping event log is "
        "append-only across SLURM preemption restarts and each restart is a "
        "clean from-scratch rerun, so cumulative cross-segment refresh counts "
        "are reported separately from the final clean-segment counts that "
        "produced these results.",
        "",
        report["reap_arm_shaping"]["note"] + ".",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["arms"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
