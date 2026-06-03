"""CLI entrypoint behavior: config errors are rejected with a clear exit code."""

import pytest

from reap.train import main


def test_invalid_config_path_exits_nonzero(capsys):
    assert main(["--config", "does/not/exist.yaml"]) == 2
    assert "config error" in capsys.readouterr().err


def test_incomplete_config_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "broken.yaml"
    p.write_text("run:\n  name: x\n")
    assert main(["--config", str(p)]) == 2
    assert "config error" in capsys.readouterr().err


def test_malformed_scalar_value_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "run:\n  name: x\n  seed: 0\n  mode: smoke\n  out_dir: runs\n"
        "  max_wall_clock_minutes: nope\n"
        "env:\n  id: overcooked\n  layout: cramped_room\n"
        "algo:\n  name: random\n  total_env_steps: 10\n"
    )
    assert main(["--config", str(p)]) == 2
    err = capsys.readouterr().err
    assert "config error" in err
    assert "max_wall_clock_minutes" in err


def test_unknown_algo_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "run:\n  name: x\n  seed: 0\n  mode: smoke\n  out_dir: runs\n"
        "  max_wall_clock_minutes: 1\n"
        "env:\n  id: overcooked\n  layout: cramped_room\n"
        "algo:\n  name: alphago\n  total_env_steps: 10\n"
    )
    assert main(["--config", str(p)]) == 2
    assert "unknown algo.name" in capsys.readouterr().err
