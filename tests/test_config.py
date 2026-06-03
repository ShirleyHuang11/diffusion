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


@pytest.mark.parametrize(
    ("needle", "replacement", "field"),
    [
        ("  max_wall_clock_minutes: 5\n", "  max_wall_clock_minutes: nope\n", "max_wall_clock_minutes"),
        ("  horizon: 100\n", "  horizon: nope\n", "horizon"),
        ("  horizon: 100\n", "  horizon: 99.5\n", "horizon"),
        ("  horizon: 100\n", "  horizon: true\n", "horizon"),
        ("  total_env_steps: 100\n", "  total_env_steps: lots\n", "total_env_steps"),
        ("  total_env_steps: 100\n", "  total_env_steps: true\n", "total_env_steps"),
        ("  seed: 0\n", "  seed: zero\n", "seed"),
        ("  name: t\n", "  name: 5\n", "name"),
        ("  max_wall_clock_minutes: 5\n", "  max_wall_clock_minutes: true\n", "max_wall_clock_minutes"),
    ],
)
def test_wrong_scalar_types_rejected(tmp_path, needle, replacement, field):
    assert needle in VALID
    with pytest.raises(ConfigError, match=field):
        load_config(write(tmp_path, VALID.replace(needle, replacement)))


def test_wrong_logging_bool_rejected(tmp_path):
    with pytest.raises(ConfigError, match="logging.jsonl"):
        load_config(write(tmp_path, VALID + "logging:\n  jsonl: maybe\n"))


def test_wrong_checkpoint_cadence_type_rejected(tmp_path):
    with pytest.raises(ConfigError, match="checkpoint.interval_env_steps"):
        load_config(write(tmp_path, VALID + "checkpoint:\n  interval_env_steps: soon\n"))


def test_wrong_params_mapping_rejected(tmp_path):
    broken = VALID + "  params: 5\n"  # appended under algo
    with pytest.raises(ConfigError, match="algo.params"):
        load_config(write(tmp_path, broken))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_rejected(tmp_path):
    with pytest.raises(ConfigError, match="malformed YAML"):
        load_config(write(tmp_path, "run: [unclosed"))
