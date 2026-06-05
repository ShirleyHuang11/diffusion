#!/usr/bin/env python
"""H2/H4 baseline-completion report (task18).

Usage:
    python scripts/h2_h4_report.py \
        --manifest configs/h2_h4_manifest.yaml \
        --out reports/h2_h4_completion.json

H2: REAP vs critic-based credit assignment (COMA, QMIX). COMA/QMIX results
enter H2 claims ONLY when their MPE Spread sanity report says PASS; a FAIL
keeps the arm's numbers in the artifact but excludes it from the hypothesis
verdict, with the exclusion stated.

H4: REAP vs generic novelty (MAPPO+RND, MAPPO+count-based) at equal tuning
budget.

Every arm is protocol-validated (exact seeds and budget) and harness-checked
(same env id, layout, encoding, and horizon within a layout). Claims reference
the extrinsic channel only. Conclusions are computed from the data; win, loss,
and null results are reported with equal prominence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


run_report = _load("run_report")
_fir = _load("first_integration_report")
mean_ci = _fir.mean_ci
reap_evidence = _fir.reap_evidence

ProtocolError = run_report.ProtocolError


def harness_check(runs_dir: str, layout: str, arms: dict, seeds: list[int]) -> dict:
    """Same wrapper/observation contract for every compared arm (hard gate)."""
    contract = None
    for arm, run in arms.items():
        for seed in seeds:
            cfg_path = Path(runs_dir) / run / f"seed{seed}" / "config_resolved.yaml"
            if not cfg_path.is_file():
                raise ProtocolError(f"missing resolved config: {cfg_path}")
            env = yaml.safe_load(cfg_path.read_text())["env"]
            this = {
                "id": env.get("id"),
                "layout": env.get("layout"),
                "encoding": env.get("encoding"),
                "horizon": env.get("horizon"),
            }
            if contract is None:
                contract = this
            elif this != contract:
                raise ProtocolError(
                    f"harness mismatch in layout {layout}: arm {arm!r} seed{seed} "
                    f"runs {this}, others run {contract}"
                )
    if contract["layout"].split("_")[0] != layout.split("_")[0]:
        raise ProtocolError(
            f"manifest/run layout mismatch: {contract['layout']} under {layout}"
        )
    return contract


def arm_stats(report: dict) -> dict:
    rates = [s["success_rate_final"] for s in report["seeds"].values()]
    returns = [s["episode_return_mean_final"] for s in report["seeds"].values()]
    return {
        "success_rate": mean_ci(rates),
        "return": mean_ci(returns),
        "per_seed": report["seeds"],
    }


def sanity_gate(path: str) -> dict:
    """COMA/QMIX usability gate from the committed sanity artifact."""
    p = Path(path)
    if not p.is_file():
        return {"path": path, "verdict": "MISSING", "usable_in_h2_claims": False,
                "note": "sanity report not found; the arm cannot enter H2 claims"}
    report = json.loads(p.read_text())
    conclusion = report.get("conclusion", {})
    return {
        "path": path,
        "verdict": conclusion.get("verdict", "MALFORMED"),
        "usable_in_h2_claims": bool(conclusion.get("usable_in_h2_claims", False)),
        "measured_max_eval_return": report.get("measured", {})
        .get("max_eval_return", {}).get("mean"),
        "published_mean": report.get("published", {}).get("mean"),
    }


def reap_gate(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"path": path, "shaping_enabled": False,
                "note": "scope quality artifact not found; REAP arm semantics unknown"}
    quality = json.loads(p.read_text())
    return {
        "path": path,
        "shaping_enabled": bool(quality.get("shaping_enabled", False)),
        "gate_violations": quality.get("gate_violations", []),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    seeds = sorted(int(s) for s in manifest["expected_seeds"])
    budget = int(manifest["expected_env_steps"])

    sanity = {algo: sanity_gate(path)
              for algo, path in manifest["sanity_reports"].items()}

    layouts: dict = {}
    try:
        for layout, spec in manifest["layouts"].items():
            arms_cfg = spec["arms"]
            contract = harness_check(args.runs_dir, layout, arms_cfg, seeds)
            arms = {
                arm: arm_stats(run_report.build_report(
                    run, args.runs_dir, seeds, budget, None, False))
                for arm, run in arms_cfg.items()
            }
            gate = reap_gate(spec["reap_quality_report"])
            layouts[layout] = {
                "harness_contract": contract,
                "arms": arms,
                "runs": arms_cfg,
                "reap_shaping": gate,
                "reap_per_seed": reap_evidence(args.runs_dir, arms_cfg["reap"], seeds),
            }
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1

    # -- computed hypothesis verdicts (extrinsic success rate, mean over seeds) --
    h2, h4 = {}, {}
    for layout, data in layouts.items():
        sr = {arm: data["arms"][arm]["success_rate"]["mean"] for arm in data["arms"]}
        excluded = [a for a in ("coma", "qmix") if not sanity[a]["usable_in_h2_claims"]]
        usable = [a for a in ("coma", "qmix") if sanity[a]["usable_in_h2_claims"]]
        h2[layout] = {
            "reap": sr["reap"],
            "coma": sr["coma"],
            "qmix": sr["qmix"],
            "excluded_from_claims": excluded,
            # None = no verdict possible (every critic-based arm sanity-excluded)
            "supported": (all(sr["reap"] > sr[a] for a in usable)
                          if usable else None),
        }
        h4[layout] = {
            "reap": sr["reap"],
            "mappo_rnd": sr["mappo_rnd"],
            "mappo_count": sr["mappo_count"],
            "supported": sr["reap"] > sr["mappo_rnd"] and sr["reap"] > sr["mappo_count"],
        }

    h2_overall = (
        None if all(v["supported"] is None for v in h2.values())
        else all(v["supported"] for v in h2.values() if v["supported"] is not None)
    )
    h4_overall = all(v["supported"] for v in h4.values())

    invariance_path = manifest.get("invariance_report")
    invariance = None
    if invariance_path and Path(invariance_path).is_file():
        inv = json.loads(Path(invariance_path).read_text())
        invariance = {"path": invariance_path, **{
            k: inv[k] for k in
            ("invariance_holds", "check", "comparison", "protocol") if k in inv
        }}

    def fmt(ci):
        return f"{ci['mean']:.3f} [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]"

    interpretation_parts = []
    for layout in layouts:
        if h2[layout]["supported"] is None:
            interpretation_parts.append(
                f"H2 on {layout}: no verdict (all critic-based arms excluded by "
                f"sanity gates: {h2[layout]['excluded_from_claims']})"
            )
        else:
            verdict = "supported" if h2[layout]["supported"] else "NOT supported"
            note = (f" (excluded: {h2[layout]['excluded_from_claims']})"
                    if h2[layout]["excluded_from_claims"] else "")
            interpretation_parts.append(f"H2 on {layout}: {verdict}{note}")
        verdict4 = "supported" if h4[layout]["supported"] else "NOT supported"
        interpretation_parts.append(f"H4 on {layout}: {verdict4}")

    report = {
        "experiment": "baseline completion: H2 (vs COMA/QMIX) and H4 (vs RND/count)",
        "tasks": sorted(layouts),
        "protocol": {
            "expected_seeds": seeds,
            "expected_env_steps": budget,
            "metric_basis": "extrinsic channel only (success rate, episode return)",
            "equal_tuning_budget_note": (
                "H4 arms share the identical MAPPO trunk and intrinsic_coef; "
                "only the bonus type differs (rnd vs count)"
            ),
            "template": "reports/templates/h2_h4_completion_template.md",
        },
        "sanity_gates": sanity,
        "layouts": layouts,
        "hypotheses": {"h2": h2, "h4": h4},
        "invariance_check": invariance,
        "conclusion": {
            "h2_supported_overall": h2_overall,
            "h4_supported_overall": h4_overall,
            "summary": "; ".join(interpretation_parts),
            "interpretation": (
                ("H4 holds nowhere in this configuration: the enabled or "
                 "gate-disabled REAP arms do not outperform generic novelty. "
                 if not h4_overall else
                 "H4 holds in this configuration. ")
                + ("H2 has no verdict because every critic-based arm failed "
                   "sanity validation. " if h2_overall is None else
                   ("H2 holds against the sanity-validated critic-based arms. "
                    if h2_overall else
                    "H2 does not hold against the sanity-validated critic-based "
                    "arms. "))
                + "Win, loss, and null results are reported with equal prominence."
            ),
        },
        "evidence": {
            "manifest": args.manifest,
            "per_arm_metrics": {
                layout: {
                    arm: [f"{args.runs_dir}/{run}/seed{s}/metrics.jsonl" for s in seeds]
                    for arm, run in data["runs"].items()
                }
                for layout, data in layouts.items()
            },
            "sanity_reports": {a: sanity[a]["path"] for a in sanity},
            "reap_quality_reports": {
                layout: data["reap_shaping"]["path"] for layout, data in layouts.items()
            },
            "invariance_report": invariance_path,
            "restart_accounting_note": (
                "shaping_events.jsonl files are append-only across SLURM "
                "preemption restarts; each restart is a clean from-scratch "
                "rerun, so cumulative cross-segment refresh counts are "
                "reported separately from final-segment counts"
            ),
        },
        "outcome_policy": "win, loss, and null results reported with equal prominence",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        "# Baseline-completion experiments (H2 / H4)",
        "",
        f"Protocol: seeds {seeds}, exact budget {budget} env steps per arm; "
        "extrinsic metrics only; pre-registered template "
        "`reports/templates/h2_h4_completion_template.md`.",
        "",
        "## Sanity gates (COMA/QMIX usability in H2 claims)",
        "",
        "| algorithm | verdict | usable in H2 claims |",
        "|---|---|---|",
    ]
    for algo, g in sorted(sanity.items()):
        lines.append(f"| {algo} | {g['verdict']} | {g['usable_in_h2_claims']} |")
    for layout, data in layouts.items():
        lines += [
            "",
            f"## {layout}",
            "",
            f"REAP shaping: {'ENABLED' if data['reap_shaping']['shaping_enabled'] else 'DISABLED by the scope gate'} "
            f"({'; '.join(data['reap_shaping'].get('gate_violations', [])) or 'gates passed'}).",
            "",
            "| arm | success rate mean [95% CI] | return mean [95% CI] |",
            "|-----|-----------------------------|------------------------|",
        ]
        for arm, stats in data["arms"].items():
            lines.append(
                f"| {arm} | {fmt(stats['success_rate'])} | {fmt(stats['return'])} |"
            )
    lines += [
        "",
        "## Invariance check",
        "",
        (f"Included from `{invariance_path}`: invariance_holds = "
         + json.dumps(invariance.get("invariance_holds"))
         if invariance else "(invariance artifact missing)"),
        "",
        "## Conclusion",
        "",
        report["conclusion"]["summary"] + ".",
        "",
        report["conclusion"]["interpretation"],
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["conclusion"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
