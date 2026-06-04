"""First-integration report generator tests: CI fields, evidence, rejections."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "first_integration_report",
    Path(__file__).resolve().parent.parent / "scripts" / "first_integration_report.py",
)
fir = importlib.util.module_from_spec(spec)
sys.modules["first_integration_report"] = fir
spec.loader.exec_module(fir)


def write_arm(runs_root: Path, name: str, seeds=(0, 1, 2), env_step=1000,
              success=(0.0, 0.1, 0.2), events=False):
    for i, seed in enumerate(seeds):
        seed_dir = runs_root / name / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        record = {
            "env_step": env_step,
            "extrinsic": {"success_rate": success[i],
                          "episode_return_mean": success[i] * 20, "episodes": 10},
            "shaped": {}, "intrinsic": {},
            "diag": {"wall_time_s": 12.5, "gpu_mem_mb": 0.0},
        }
        (seed_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
        if events:
            (seed_dir / "shaping_events.jsonl").write_text(
                json.dumps({"type": "gate", "enabled": False,
                            "reason": "test gate"}) + "\n"
            )


def make_quality(tmp_path, enabled=False):
    p = tmp_path / "quality.json"
    p.write_text(json.dumps({"shaping_enabled": enabled,
                             "gate_violations": [] if enabled else ["test gate"]}))
    return str(p)


def run_report(tmp_path, **overrides):
    args = [
        "--reap-run", "reap_arm", "--vanilla-run", "vanilla_arm",
        "--rnd-run", "rnd_arm", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--quality-report", make_quality(tmp_path),
        "--out", str(tmp_path / "report.json"),
    ]
    return fir.main(args)


def test_report_includes_ci_and_evidence(tmp_path):
    runs = tmp_path / "runs"
    write_arm(runs, "reap_arm", events=True)
    write_arm(runs, "vanilla_arm")
    write_arm(runs, "rnd_arm", success=(0.9, 1.0, 1.0))
    assert run_report(tmp_path) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    for arm in report["arms"].values():
        assert "ci95" in arm["success_rate"] and "ci95" in arm["return"]
        lo, hi = arm["success_rate"]["ci95"]
        assert lo <= arm["success_rate"]["mean"] <= hi
    # n=3 95% t-interval hand check for rnd arm: mean .9667, sd .0577
    rnd = report["arms"]["mappo_rnd"]["success_rate"]
    assert rnd["mean"] == pytest.approx(0.96667, abs=1e-4)
    half = fir.T_975_DF2 * (0.057735 / (3 ** 0.5))
    assert rnd["ci95"][1] - rnd["mean"] == pytest.approx(half, abs=1e-3)
    # evidence block present with per-seed gate/runtime info
    ev = report["evidence"]
    assert ev["quality_report"].endswith("quality.json")
    assert ev["reap_per_seed"]["seed0"]["gate"]["enabled"] is False
    assert ev["reap_per_seed"]["seed0"]["wall_time_s"] == 12.5
    md = (tmp_path / "report.md").read_text()
    assert "95% CI" in md


def test_report_rejects_missing_seed(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_arm(runs, "reap_arm", seeds=(0, 1))
    write_arm(runs, "vanilla_arm")
    write_arm(runs, "rnd_arm")
    assert run_report(tmp_path) == 1
    assert "missing expected seeds" in capsys.readouterr().err


def test_report_rejects_budget_mismatch(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_arm(runs, "reap_arm", env_step=1192)
    write_arm(runs, "vanilla_arm")
    write_arm(runs, "rnd_arm")
    assert run_report(tmp_path) == 1
    assert "expected exactly 1000" in capsys.readouterr().err