"""Report-generator protocol validation tests."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_report", Path(__file__).resolve().parent.parent / "scripts" / "run_report.py"
)
run_report = importlib.util.module_from_spec(spec)
sys.modules["run_report"] = run_report
spec.loader.exec_module(run_report)


def write_seed(run_root: Path, seed: int, env_step: int = 1000, success: float = 0.5):
    seed_dir = run_root / f"seed{seed}"
    seed_dir.mkdir(parents=True)
    record = {
        "env_step": env_step,
        "extrinsic": {
            "success_rate": success,
            "episode_return_mean": success * 20,
            "episodes": 10,
        },
        "shaped": {},
        "intrinsic": {},
        "diag": {},
    }
    (seed_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    return seed_dir


def make_run(tmp_path, seeds=(0, 1, 2), env_step=1000):
    run_root = tmp_path / "runs" / "exp"
    for s in seeds:
        write_seed(run_root, s, env_step=env_step)
    return run_root


def test_valid_three_seed_report(tmp_path):
    make_run(tmp_path)
    out = tmp_path / "reports" / "exp.json"
    rc = run_report.main([
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--threshold", "0.4", "--out", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["expected_seeds"] == [0, 1, 2]
    assert report["expected_env_steps"] == 1000
    assert report["gate_passed"] is True
    assert report["metric_basis"].startswith("extrinsic")
    assert out.with_suffix(".md").is_file()


def test_missing_seed_rejected(tmp_path, capsys):
    make_run(tmp_path, seeds=(0, 1))
    rc = run_report.main([
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--out", str(tmp_path / "r.json"),
    ])
    assert rc == 1
    assert "missing expected seeds [2]" in capsys.readouterr().err


def test_extra_seed_rejected_unless_allowed(tmp_path, capsys):
    make_run(tmp_path, seeds=(0, 1, 2, 3))
    args = [
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--out", str(tmp_path / "r.json"),
    ]
    assert run_report.main(args) == 1
    assert "extra seeds [3]" in capsys.readouterr().err
    assert run_report.main(args + ["--allow-extra-seeds"]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["extra_seeds_ignored"] == [3]
    assert "seed3" not in report["seeds"]


def test_malformed_seed_dir_rejected(tmp_path, capsys):
    run_root = make_run(tmp_path)
    (run_root / "seed_zero").mkdir()
    rc = run_report.main([
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--out", str(tmp_path / "r.json"),
    ])
    assert rc == 1
    assert "malformed seed directory" in capsys.readouterr().err


def test_missing_metrics_rejected(tmp_path, capsys):
    run_root = make_run(tmp_path, seeds=(0, 1))
    (run_root / "seed2").mkdir()  # seed dir exists but no metrics.jsonl
    rc = run_report.main([
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--out", str(tmp_path / "r.json"),
    ])
    assert rc == 1
    assert "missing metrics file" in capsys.readouterr().err


def test_budget_mismatch_rejected(tmp_path, capsys):
    run_root = make_run(tmp_path, seeds=(0, 1))
    write_seed(run_root, 2, env_step=1192)  # overshoot
    rc = run_report.main([
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--out", str(tmp_path / "r.json"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ended at env_step 1192" in err
    assert "expected exactly 1000" in err


def test_empty_metrics_rejected(tmp_path, capsys):
    run_root = make_run(tmp_path, seeds=(0, 1))
    seed_dir = run_root / "seed2"
    seed_dir.mkdir()
    (seed_dir / "metrics.jsonl").write_text("")
    rc = run_report.main([
        "--run-name", "exp", "--runs-dir", str(tmp_path / "runs"),
        "--expected-seeds", "0,1,2", "--expected-env-steps", "1000",
        "--out", str(tmp_path / "r.json"),
    ])
    assert rc == 1
    assert "empty metrics file" in capsys.readouterr().err