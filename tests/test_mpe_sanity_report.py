"""MPE sanity-report generator tests: verdicts, rejections, evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "mpe_sanity_report",
    Path(__file__).resolve().parent.parent / "scripts" / "mpe_sanity_report.py",
)
msr = importlib.util.module_from_spec(spec)
sys.modules["mpe_sanity_report"] = msr
spec.loader.exec_module(msr)


def write_seed(runs_root, run, seed, algo="qmix", env_steps=1000,
               eval_returns=(-130.0, -127.0), horizon=25, with_eval=True,
               final_eval=True):
    seed_dir = runs_root / run / f"seed{seed}"
    seed_dir.mkdir(parents=True)
    cfg = {
        "env": {"id": "mpe_spread", "num_agents": 3, "horizon": horizon},
        "algo": {"name": algo, "total_env_steps": env_steps},
    }
    (seed_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))
    records = []
    steps = [env_steps // 2, env_steps]
    for i, step in enumerate(steps):
        extrinsic = {"episode_return_mean": eval_returns[i] - 5.0,
                     "episodes": 10.0, "success_rate": 0.0}
        if with_eval and (i < len(steps) - 1 or final_eval):
            extrinsic.update({"eval_return_mean": eval_returns[i],
                              "eval_return_std": 3.0, "eval_episodes": 100.0})
        records.append({"env_step": step, "extrinsic": extrinsic,
                        "shaped": {}, "intrinsic": {}, "diag": {}})
    (seed_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )


def run_gen(tmp_path, algo="qmix", run="sanity", env_steps=1000, tolerance=0.15):
    out = tmp_path / "report.json"
    code = msr.main([
        "--algo", algo, "--run", run, "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", str(env_steps),
        "--tolerance", str(tolerance), "--out", str(out),
    ])
    return code, out


def test_pass_verdict_within_band(tmp_path, monkeypatch):
    monkeypatch.setattr(msr, "random_reference", lambda **kw: -240.0)
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, eval_returns=(-150.0, -128.0 - s))
    code, out = run_gen(tmp_path)
    assert code == 0
    report = json.loads(out.read_text())
    assert report["conclusion"]["verdict"] == "PASS"
    assert report["conclusion"]["usable_in_h2_claims"] is True
    # max over eval points, not final: seed uses the better late value
    assert report["measured"]["max_eval_return"]["mean"] == pytest.approx(-129.0)
    assert "ci95" in report["measured"]["max_eval_return"]
    assert report["measured"]["beats_random_reference"] is True
    assert "Papoudakis" in report["published"]["source"]
    md = out.with_suffix(".md").read_text()
    assert "Verdict: PASS" in md


def test_fail_verdict_on_deficit_still_generates(tmp_path, monkeypatch):
    monkeypatch.setattr(msr, "random_reference", lambda **kw: -240.0)
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, eval_returns=(-260.0, -230.0))
    code, out = run_gen(tmp_path)
    assert code == 0  # a FAIL verdict is a result, not a generator error
    report = json.loads(out.read_text())
    assert report["conclusion"]["verdict"] == "FAIL"
    assert report["conclusion"]["usable_in_h2_claims"] is False
    assert "MUST NOT be used in H2 claims" in report["conclusion"]["interpretation"]


def test_fail_verdict_on_suspicious_excess(tmp_path, monkeypatch):
    monkeypatch.setattr(msr, "random_reference", lambda **kw: -240.0)
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, eval_returns=(-60.0, -50.0))
    code, out = run_gen(tmp_path)
    assert code == 0
    report = json.loads(out.read_text())
    assert report["conclusion"]["verdict"] == "FAIL"
    assert "mismatch" in report["conclusion"]["interpretation"]


def test_rejects_missing_seed(tmp_path, capsys):
    runs = tmp_path / "runs"
    for s in range(2):
        write_seed(runs, "sanity", s)
    code, _ = run_gen(tmp_path)
    assert code == 1
    assert "missing expected seeds" in capsys.readouterr().err


def test_rejects_budget_mismatch(tmp_path, capsys):
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, env_steps=999)
    code, _ = run_gen(tmp_path, env_steps=1000)
    assert code == 1
    assert "budget mismatch" in capsys.readouterr().err


def test_rejects_harness_mismatch(tmp_path, capsys):
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, horizon=50)  # wrong episode length
    code, _ = run_gen(tmp_path)
    assert code == 1
    assert "harness mismatch" in capsys.readouterr().err


def test_rejects_wrong_algo(tmp_path, capsys):
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, algo="coma")
    code, _ = run_gen(tmp_path, algo="qmix")
    assert code == 1
    assert "harness mismatch" in capsys.readouterr().err


def test_rejects_missing_eval_channel(tmp_path, capsys):
    runs = tmp_path / "runs"
    for s in range(3):
        write_seed(runs, "sanity", s, with_eval=False)
    code, _ = run_gen(tmp_path)
    assert code == 1
    assert "no greedy-evaluation channel" in capsys.readouterr().err


def test_random_reference_matches_harness_scale():
    ref = msr.random_reference(episodes=50, seed=0)
    assert -300.0 < ref < -180.0  # the measured harness scale for random play
