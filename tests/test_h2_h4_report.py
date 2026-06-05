"""H2/H4 report generator tests: gates, harness checks, computed verdicts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "h2_h4_report",
    Path(__file__).resolve().parent.parent / "scripts" / "h2_h4_report.py",
)
hhr = importlib.util.module_from_spec(spec)
sys.modules["h2_h4_report"] = hhr
spec.loader.exec_module(hhr)

ARMS = ("reap", "vanilla_mappo", "mappo_rnd", "mappo_count", "coma", "qmix")


def write_arm(runs_root, run, success, env_steps=1000, layout="forced_coordination",
              horizon=400, seeds=(0, 1, 2)):
    for seed in seeds:
        seed_dir = runs_root / run / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        cfg = {"env": {"id": "overcooked", "layout": layout,
                       "encoding": "lossless", "horizon": horizon},
               "algo": {"total_env_steps": env_steps}}
        (seed_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))
        record = {
            "env_step": env_steps,
            "extrinsic": {"success_rate": success, "episode_return_mean": success * 20,
                          "episodes": 10},
            "shaped": {}, "intrinsic": {}, "diag": {},
        }
        (seed_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n")


def write_sanity(path, verdict="PASS"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "conclusion": {"verdict": verdict,
                       "usable_in_h2_claims": verdict == "PASS"},
        "measured": {"max_eval_return": {"mean": -130.0}},
        "published": {"mean": -126.62},
    }))


def write_quality(path, enabled=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"shaping_enabled": enabled, "gate_violations": []}))


def setup_world(tmp_path, success_by_arm=None, sanity=("PASS", "PASS"),
                one_layout=True):
    runs = tmp_path / "runs"
    success = {"reap": 0.0, "vanilla_mappo": 0.0, "mappo_rnd": 1.0,
               "mappo_count": 0.5, "coma": 0.0, "qmix": 0.0}
    success.update(success_by_arm or {})
    layouts = {"forced_coordination": {
        "arms": {arm: f"{arm}_forced" for arm in ARMS},
        "reap_quality_report": str(tmp_path / "q_forced.json"),
    }}
    for arm in ARMS:
        write_arm(runs, f"{arm}_forced", success[arm])
    if not one_layout:
        layouts["counter_circuit"] = {
            "arms": {arm: f"{arm}_counter" for arm in ARMS},
            "reap_quality_report": str(tmp_path / "q_counter.json"),
        }
        for arm in ARMS:
            write_arm(runs, f"{arm}_counter", success[arm], layout="counter_circuit")
        write_quality(tmp_path / "q_counter.json")
    write_quality(tmp_path / "q_forced.json")
    write_sanity(tmp_path / "sanity_coma.json", sanity[0])
    write_sanity(tmp_path / "sanity_qmix.json", sanity[1])
    manifest = {
        "expected_seeds": [0, 1, 2],
        "expected_env_steps": 1000,
        "layouts": layouts,
        "sanity_reports": {"coma": str(tmp_path / "sanity_coma.json"),
                           "qmix": str(tmp_path / "sanity_qmix.json")},
    }
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(yaml.safe_dump(manifest))
    return mpath


def run_gen(tmp_path, mpath):
    out = tmp_path / "h2h4.json"
    code = hhr.main(["--manifest", str(mpath), "--runs-dir", str(tmp_path / "runs"),
                     "--out", str(out)])
    return code, out


def test_negative_verdicts_computed_and_reported(tmp_path):
    mpath = setup_world(tmp_path)
    code, out = run_gen(tmp_path, mpath)
    assert code == 0
    report = json.loads(out.read_text())
    h2 = report["hypotheses"]["h2"]["forced_coordination"]
    h4 = report["hypotheses"]["h4"]["forced_coordination"]
    assert h2["supported"] is False  # reap 0.0 does not beat coma/qmix 0.0
    assert h4["supported"] is False  # reap 0.0 loses to rnd 1.0
    assert report["conclusion"]["h4_supported_overall"] is False
    assert "NOT supported" in report["conclusion"]["summary"]
    md = out.with_suffix(".md").read_text()
    assert "equal prominence" in md


def test_h2_supported_when_reap_beats_validated_arms(tmp_path):
    mpath = setup_world(tmp_path, success_by_arm={"reap": 0.9})
    code, out = run_gen(tmp_path, mpath)
    assert code == 0
    report = json.loads(out.read_text())
    assert report["hypotheses"]["h2"]["forced_coordination"]["supported"] is True
    # H4 still fails: rnd 1.0 > reap 0.9
    assert report["hypotheses"]["h4"]["forced_coordination"]["supported"] is False


def test_sanity_fail_excludes_arm_from_h2(tmp_path):
    mpath = setup_world(tmp_path, success_by_arm={"reap": 0.9},
                        sanity=("FAIL", "PASS"))
    code, out = run_gen(tmp_path, mpath)
    report = json.loads(out.read_text())
    h2 = report["hypotheses"]["h2"]["forced_coordination"]
    assert h2["excluded_from_claims"] == ["coma"]
    assert h2["supported"] is True  # judged on qmix only
    assert report["sanity_gates"]["coma"]["usable_in_h2_claims"] is False


def test_all_sanity_failed_gives_no_h2_verdict(tmp_path):
    mpath = setup_world(tmp_path, sanity=("FAIL", "FAIL"))
    code, out = run_gen(tmp_path, mpath)
    report = json.loads(out.read_text())
    assert report["hypotheses"]["h2"]["forced_coordination"]["supported"] is None
    assert report["conclusion"]["h2_supported_overall"] is None
    assert "no verdict" in report["conclusion"]["summary"]


def test_missing_sanity_report_blocks_h2_use(tmp_path):
    mpath = setup_world(tmp_path)
    (tmp_path / "sanity_qmix.json").unlink()
    code, out = run_gen(tmp_path, mpath)
    assert code == 0
    report = json.loads(out.read_text())
    assert report["sanity_gates"]["qmix"]["verdict"] == "MISSING"
    assert "qmix" in report["hypotheses"]["h2"]["forced_coordination"][
        "excluded_from_claims"]


def test_harness_mismatch_rejected(tmp_path):
    mpath = setup_world(tmp_path)
    # corrupt one arm's harness: different horizon
    cfgp = (tmp_path / "runs" / "coma_forced" / "seed1" / "config_resolved.yaml")
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["env"]["horizon"] = 200
    cfgp.write_text(yaml.safe_dump(cfg))
    code, _ = run_gen(tmp_path, mpath)
    assert code == 1


def test_budget_mismatch_rejected(tmp_path, capsys):
    mpath = setup_world(tmp_path)
    metrics = (tmp_path / "runs" / "qmix_forced" / "seed0" / "metrics.jsonl")
    record = json.loads(metrics.read_text())
    record["env_step"] = 999
    metrics.write_text(json.dumps(record) + "\n")
    code, _ = run_gen(tmp_path, mpath)
    assert code == 1


def test_two_layouts_and_gate_disabled_reap(tmp_path):
    mpath = setup_world(tmp_path, one_layout=False)
    write_quality(tmp_path / "q_counter.json", enabled=False)
    code, out = run_gen(tmp_path, mpath)
    assert code == 0
    report = json.loads(out.read_text())
    assert set(report["layouts"]) == {"forced_coordination", "counter_circuit"}
    assert report["layouts"]["counter_circuit"]["reap_shaping"][
        "shaping_enabled"] is False
    md = out.with_suffix(".md").read_text()
    assert "DISABLED by the scope gate" in md


def test_evidence_lists_all_arm_metrics(tmp_path):
    mpath = setup_world(tmp_path)
    _, out = run_gen(tmp_path, mpath)
    report = json.loads(out.read_text())
    per_arm = report["evidence"]["per_arm_metrics"]["forced_coordination"]
    assert set(per_arm) == set(ARMS)
    assert all(len(paths) == 3 for paths in per_arm.values())
