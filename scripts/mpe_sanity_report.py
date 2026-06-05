#!/usr/bin/env python
"""MPE Spread sanity-validation report for in-repo COMA/QMIX (task17).

Usage:
    python scripts/mpe_sanity_report.py \
        --algo qmix --run sanity_qmix_spread \
        --expected-seeds 0,1,2 --expected-env-steps 2000000 \
        --out reports/sanity_qmix_spread.json

Compares the in-repo implementation against the published EPyMARL benchmark
returns on MPE Spread and emits a PASS/FAIL verdict. A FAIL verdict gates the
algorithm out of H2 claims (plan AC-9.2 negative test); it is a first-class
result, not a generator error. Generator errors (exit 1) are protocol
violations: missing seeds, budget mismatches, harness mismatches, or a
missing evaluation channel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "first_integration_report",
    Path(__file__).resolve().parent / "first_integration_report.py",
)
_fir = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("first_integration_report", _fir)
_spec.loader.exec_module(_fir)
mean_ci = _fir.mean_ci

# Published sanity anchors: Papoudakis, Christianos, Schaefer & Albrecht,
# "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in
# Cooperative Tasks" (NeurIPS 2021 Datasets and Benchmarks), Table 3,
# MPE Spread, parameter sharing, MAXIMUM returns over 41 greedy evaluation
# points (100 episodes each), 25-step episodes, per-agent rewards summed
# over the 3 agents. Budgets: 2M env steps (off-policy) / 20M (on-policy).
PUBLISHED = {
    "qmix": {"mean": -126.62, "ci95": 2.96, "budget": 2_000_000},
    "coma": {"mean": -204.31, "ci95": 6.30, "budget": 20_000_000},
    "mappo": {"mean": -133.54, "ci95": 3.08, "budget": 20_000_000},  # reference
}
SOURCE = (
    "Papoudakis et al. 2021 (arXiv:2006.07869, NeurIPS Datasets & Benchmarks), "
    "Table 3, MPE Spread, parameter sharing, maximum returns"
)

HARNESS_CONTRACT = {
    "env_id": "mpe_spread",
    "num_agents": 3,
    "horizon": 25,
    "observation": (
        "per-agent 18-dim [own vel(2), own pos(2), landmark rel pos(6), "
        "other-agent rel pos(4), comm zeros(4)] in the reference ordering"
    ),
    "reward": (
        "per-agent: -(sum over landmarks of min-agent distance) - 1 per "
        "colliding agent INCLUDING the agent itself (reference quirk "
        "preserved); team reward = per-agent rewards summed over agents"
    ),
    "evaluation": "greedy policy, separate env instance, deterministic seeds",
}


class ProtocolError(RuntimeError):
    pass


def load_seed(run_dir: Path, seed: int, expected_steps: int, algo: str) -> dict:
    seed_dir = run_dir / f"seed{seed}"
    metrics_path = seed_dir / "metrics.jsonl"
    config_path = seed_dir / "config_resolved.yaml"
    if not metrics_path.is_file():
        raise ProtocolError(f"missing expected seeds: no metrics at {metrics_path}")
    if not config_path.is_file():
        raise ProtocolError(f"missing resolved config at {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    env_cfg, algo_cfg = cfg["env"], cfg["algo"]
    for key, want in (("id", HARNESS_CONTRACT["env_id"]),
                      ("num_agents", HARNESS_CONTRACT["num_agents"]),
                      ("horizon", HARNESS_CONTRACT["horizon"])):
        got = env_cfg.get(key)
        if got != want:
            raise ProtocolError(
                f"harness mismatch in seed{seed}: env.{key}={got!r}, expected {want!r}"
            )
    if algo_cfg.get("name") != algo:
        raise ProtocolError(
            f"harness mismatch in seed{seed}: algo.name={algo_cfg.get('name')!r}, "
            f"expected {algo!r}"
        )
    if int(algo_cfg.get("total_env_steps", -1)) != expected_steps:
        raise ProtocolError(
            f"budget mismatch in seed{seed}: configured "
            f"{algo_cfg.get('total_env_steps')}, expected exactly {expected_steps}"
        )

    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if not records:
        raise ProtocolError(f"empty metrics for seed{seed}")
    final_step = records[-1]["env_step"]
    if final_step != expected_steps:
        raise ProtocolError(
            f"budget mismatch in seed{seed}: final metrics at env_step "
            f"{final_step}, expected exactly {expected_steps}"
        )

    eval_points = [
        {"env_step": r["env_step"],
         "eval_return_mean": r["extrinsic"]["eval_return_mean"],
         "eval_episodes": r["extrinsic"].get("eval_episodes")}
        for r in records if "eval_return_mean" in r.get("extrinsic", {})
    ]
    if not eval_points:
        raise ProtocolError(
            f"seed{seed} has no greedy-evaluation channel; sanity validation "
            "compares MAXIMUM evaluation returns and cannot use training returns"
        )
    if "eval_return_mean" not in records[-1]["extrinsic"]:
        raise ProtocolError(f"seed{seed} final record lacks the evaluation block")

    best = max(p["eval_return_mean"] for p in eval_points)
    return {
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
        "eval_points": len(eval_points),
        "eval_episodes_per_point": eval_points[-1]["eval_episodes"],
        "max_eval_return": best,
        "final_eval_return": eval_points[-1]["eval_return_mean"],
        "final_train_return": records[-1]["extrinsic"]["episode_return_mean"],
    }


def random_reference(episodes: int = 300, seed: int = 0) -> float:
    """Random-policy mean episode return measured in the same harness."""
    from reap.envs.mpe_spread import MpeSpreadEnv

    rng = np.random.default_rng(seed)
    env = MpeSpreadEnv()
    returns = []
    for ep in range(episodes):
        env.reset(seed=10_000_000 + ep)
        total, done = 0.0, False
        while not done:
            result = env.step(rng.integers(0, env.num_actions, env.num_agents).tolist())
            total += result.extrinsic_reward
            done = result.terminated or result.truncated
        returns.append(total)
    return float(np.mean(returns))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=sorted(PUBLISHED))
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    parser.add_argument(
        "--tolerance", type=float, default=0.15,
        help="two-sided fractional band around the published mean (default 0.15); "
        "a large DEFICIT or a suspicious EXCESS both fail (excess indicates a "
        "reward-scale mismatch, not a better implementation)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    seeds = sorted(int(s) for s in args.expected_seeds.split(",") if s.strip())
    run_dir = Path(args.runs_dir) / args.run
    try:
        per_seed = {f"seed{s}": load_seed(run_dir, s, args.expected_env_steps, args.algo)
                    for s in seeds}
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1

    max_returns = [per_seed[f"seed{s}"]["max_eval_return"] for s in seeds]
    published = PUBLISHED[args.algo]
    band_half = args.tolerance * abs(published["mean"])
    band = [published["mean"] - band_half, published["mean"] + band_half]
    measured = mean_ci(max_returns)
    in_band = band[0] <= measured["mean"] <= band[1]
    verdict = "PASS" if in_band else "FAIL"
    random_ref = random_reference()
    beats_random = measured["mean"] > random_ref

    if in_band:
        interpretation = (
            f"the in-repo {args.algo.upper()} reaches maximum evaluation returns "
            f"({measured['mean']:.2f}) within {args.tolerance:.0%} of the published "
            f"value ({published['mean']:.2f}); the implementation is sanity-"
            f"validated for use in H2 claims"
        )
    elif measured["mean"] < band[0]:
        interpretation = (
            f"the in-repo {args.algo.upper()} underperforms the published value "
            f"({measured['mean']:.2f} vs {published['mean']:.2f}, band {band}); "
            f"it MUST NOT be used in H2 claims until fixed (plan AC-9.2 gate)"
        )
    else:
        interpretation = (
            f"the in-repo {args.algo.upper()} exceeds the published band "
            f"({measured['mean']:.2f} vs {published['mean']:.2f}, band {band}), "
            f"which indicates a probable reward/protocol mismatch rather than a "
            f"better implementation; it MUST NOT be used in H2 claims until the "
            f"discrepancy is explained"
        )

    report = {
        "experiment": f"sanity validation: in-repo {args.algo.upper()} on MPE Spread",
        "protocol": {
            "expected_seeds": seeds,
            "expected_env_steps": args.expected_env_steps,
            "published_budget": published["budget"],
            "metric_basis": (
                "maximum greedy-evaluation return over training (published "
                "convention); evaluation episodes never consume training budget"
            ),
            "harness_contract": HARNESS_CONTRACT,
        },
        "published": {**published, "source": SOURCE},
        "tolerance": args.tolerance,
        "acceptance_band": band,
        "measured": {
            "max_eval_return": measured,
            "per_seed": per_seed,
            "random_policy_reference": random_ref,
            "beats_random_reference": beats_random,
        },
        "conclusion": {
            "verdict": verdict,
            "usable_in_h2_claims": bool(in_band),
            "interpretation": interpretation,
        },
        "evidence": {
            "per_seed_metrics": [per_seed[f"seed{s}"]["metrics_path"] for s in seeds],
            "per_seed_configs": [per_seed[f"seed{s}"]["config_path"] for s in seeds],
            "published_source": SOURCE,
        },
        "outcome_policy": "PASS and FAIL verdicts are reported with equal prominence",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    sr = measured
    lines = [
        f"# MPE Spread sanity validation: in-repo {args.algo.upper()}",
        "",
        f"Protocol: seeds {seeds}, exact budget {args.expected_env_steps} env "
        f"steps, maximum greedy-evaluation returns ({SOURCE}).",
        "",
        f"| | max eval return |",
        f"|---|---|",
        f"| in-repo {args.algo.upper()} (mean [95% CI], n={sr['n']}) | "
        f"{sr['mean']:.2f} [{sr['ci95'][0]:.2f}, {sr['ci95'][1]:.2f}] |",
        f"| published | {published['mean']:.2f} ± {published['ci95']:.2f} |",
        f"| acceptance band (±{args.tolerance:.0%}) | [{band[0]:.2f}, {band[1]:.2f}] |",
        f"| random-policy reference (same harness) | {random_ref:.2f} |",
        "",
        f"## Verdict: {verdict}",
        "",
        report["conclusion"]["interpretation"] + ".",
        "",
        f"Evidence: per-seed metrics/configs and the harness contract "
        f"(observation layout, reward convention incl. the reference "
        f"self-collision term, greedy evaluation) are recorded in the JSON "
        f"artifact.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["conclusion"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
