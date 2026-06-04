"""Deployment-boundary tests: raising stubs and import-graph enforcement."""

import subprocess
import sys
import types

import numpy as np
import pytest

from reap.algos.mappo import MappoTrainer
from reap.checkpoint import save_checkpoint
from reap.eval.deploy import FORBIDDEN_IMPORTS, evaluate_checkpoint, evaluate_policy
from tests.chain_env import ChainEnv

TINY = {"rollout_length": 16, "hidden_size": 16, "update_epochs": 1, "num_minibatches": 2}


@pytest.fixture()
def chain_checkpoint(tmp_path):
    trainer = MappoTrainer(ChainEnv(4, 8), TINY, seed=0)
    trainer.collect_rollout()
    path = tmp_path / "step_000000000016.pt"
    save_checkpoint(
        {"trainer": trainer.state_dict(),
         "config": {"algo": {"params": {"hidden_size": 16}}}},
        path,
    )
    return path


class _RaisingModule(types.ModuleType):
    def __getattr__(self, name):
        raise RuntimeError(f"deployment boundary violated: touched {self.__name__}.{name}")


def test_evaluation_runs_with_raising_stubs_and_matches_reference(chain_checkpoint):
    reference = evaluate_checkpoint(chain_checkpoint, ChainEnv(4, 8), episodes=4, seed=7)

    saved = {}
    try:
        for name in FORBIDDEN_IMPORTS:
            saved[name] = sys.modules.get(name)
            sys.modules[name] = _RaisingModule(name)
        stubbed = evaluate_checkpoint(chain_checkpoint, ChainEnv(4, 8), episodes=4, seed=7)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert stubbed["first_actions"] == reference["first_actions"]
    assert stubbed["extrinsic_return_mean"] == reference["extrinsic_return_mean"]
    assert stubbed["success_rate"] == reference["success_rate"]


def test_import_graph_excludes_training_scaffolding():
    """Importing the deployment module must not pull in teacher/shaping code."""
    code = (
        "import sys; import reap.eval.deploy; "
        "bad = [m for m in "
        f"{list(FORBIDDEN_IMPORTS)!r}"
        " if m in sys.modules]; "
        "assert not bad, f'forbidden modules imported: {bad}'; print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
        cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_evaluation_reports_extrinsic_metrics_only(chain_checkpoint):
    report = evaluate_checkpoint(chain_checkpoint, ChainEnv(4, 8), episodes=3, seed=1)
    assert set(report) == {
        "episodes", "extrinsic_return_mean", "success_rate", "first_actions"
    }
    assert 0.0 <= report["success_rate"] <= 1.0


def test_evaluate_policy_deterministic_for_seed(chain_checkpoint):
    env = ChainEnv(4, 8)
    from reap.eval.deploy import load_policy

    nets = load_policy(chain_checkpoint, env)
    a = evaluate_policy(nets, ChainEnv(4, 8), episodes=3, seed=5)
    b = evaluate_policy(nets, ChainEnv(4, 8), episodes=3, seed=5)
    assert a == b