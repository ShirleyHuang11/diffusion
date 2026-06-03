"""Config loading and validation tests."""

import textwrap

import pytest

from reap.config import ConfigError, load_config

VALID = """
run:
  name: t
  seed: 0
  mode: smoke
  out_dir: runs
  max_wall_clock_minutes: 5
env:
  id: overcooked
  layout: cramped_room
  horizon: 100
algo:
  name: random
  total_env_steps: 100
"""


def write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_valid_config_loads(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.run.name == "t"
    assert cfg.env.layout == "cramped_room"
    assert cfg.logging.interval_env_steps > 0  # optional section defaulted


def test_missing_required_key_rejected(tmp_path):
    broken = VALID.replace("  seed: 0\n", "")
    with pytest.raises(ConfigError, match="run.seed"):
        load_config(write(tmp_path, broken))


def test_missing_required_section_rejected(tmp_path):
    broken = VALID.split("env:")[0]  # run section only
    with pytest.raises(ConfigError, match="missing required section"):
        load_config(write(tmp_path, broken))


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(write(tmp_path, VALID + "  typo_key: 1\n"))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(write(tmp_path, VALID + "mystery:\n  a: 1\n"))


def test_invalid_mode_rejected(tmp_path):
    with pytest.raises(ConfigError, match="run.mode"):
        load_config(write(tmp_path, VALID.replace("mode: smoke", "mode: turbo")))


def test_overcooked_requires_layout(tmp_path):
    with pytest.raises(ConfigError, match="env.layout"):
        load_config(write(tmp_path, VALID.replace("  layout: cramped_room\n", "")))


def test_nonpositive_steps_rejected(tmp_path):
    with pytest.raises(ConfigError, match="total_env_steps"):
        load_config(write(tmp_path, VALID.replace("total_env_steps: 100", "total_env_steps: 0")))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_rejected(tmp_path):
    with pytest.raises(ConfigError, match="malformed YAML"):
        load_config(write(tmp_path, "run: [unclosed"))
