"""Diffusion teacher tests on the chain fixture: training, inpainting, quality."""

import numpy as np
import pytest
import torch

from reap.data import TrajectoryBuffer, collect_warmup
from reap.diffusion import GaussianDiffusion, TrajectoryDenoiser, TrajectoryWindowDataset
from reap.diffusion.ddpm import train_teacher
from reap.diffusion.quality import StateValidator, generation_quality_report
from tests.chain_env import ChainEnv

LENGTH, WINDOW = 4, 4


class OneHotValidator(StateValidator):
    def project(self, states):
        out = np.zeros_like(states)
        out[np.arange(len(states)), np.argmax(states, axis=-1)] = 1.0
        return out

    def is_valid(self, states):
        return (np.abs(states.sum(axis=-1) - 1.0) < 1e-6) & (states.max(axis=-1) == 1.0)


class AlwaysInvalidValidator(StateValidator):
    def project(self, states):
        return states

    def is_valid(self, states):
        return np.zeros(len(states), dtype=bool)


@pytest.fixture(scope="module")
def chain_buffer():
    env = ChainEnv(length=LENGTH, horizon=8)
    buffer, _ = collect_warmup(
        env,
        ladder=[("solver", lambda lo, js: [1, 1])],
        min_successes=30,
        max_env_steps=2000,
    )
    return buffer


@pytest.fixture(scope="module")
def trained_teacher(chain_buffer):
    dataset = TrajectoryWindowDataset(chain_buffer, window=WINDOW, stride=1)
    torch.manual_seed(0)
    model = TrajectoryDenoiser(
        state_dim=LENGTH, window=WINDOW, d_model=32, nhead=2, num_layers=1
    )
    diffusion = GaussianDiffusion(num_steps=25)
    history = train_teacher(model, diffusion, dataset, steps=300, batch_size=32, seed=0)
    return dataset, model, diffusion, history


def test_dataset_windows_and_normalization(chain_buffer):
    dataset = TrajectoryWindowDataset(chain_buffer, window=WINDOW, stride=1)
    assert dataset.windows.shape[1:] == (WINDOW, LENGTH)
    roundtrip = dataset.denormalize(dataset.normalize(dataset.windows[:5]))
    assert np.allclose(roundtrip, dataset.windows[:5], atol=1e-5)


def test_dataset_rejects_too_long_window(chain_buffer):
    with pytest.raises(ValueError, match="no windows"):
        TrajectoryWindowDataset(chain_buffer, window=100)


def test_training_loss_decreases(trained_teacher):
    _, _, _, history = trained_teacher
    assert np.mean(history[-20:]) < np.mean(history[:20]) * 0.8


def test_pin_start_is_exact(trained_teacher):
    dataset, model, diffusion, _ = trained_teacher
    start = torch.as_tensor(dataset.normalize(np.eye(LENGTH, dtype=np.float32)[0]))
    samples = diffusion.sample(
        model, n=8, pin={0: start}, generator=torch.Generator().manual_seed(1)
    )
    assert samples.shape == (8, WINDOW, LENGTH)
    assert torch.allclose(samples[:, 0], start.expand(8, -1))


def test_bridge_pins_both_ends_exactly(trained_teacher):
    dataset, model, diffusion, _ = trained_teacher
    start = torch.as_tensor(dataset.normalize(np.eye(LENGTH, dtype=np.float32)[0]))
    goal = torch.as_tensor(dataset.normalize(np.eye(LENGTH, dtype=np.float32)[LENGTH - 1]))
    samples = diffusion.sample(
        model, n=6, pin={0: start, WINDOW - 1: goal},
        generator=torch.Generator().manual_seed(2),
    )
    assert torch.allclose(samples[:, 0], start.expand(6, -1))
    assert torch.allclose(samples[:, WINDOW - 1], goal.expand(6, -1))


def test_pin_index_out_of_window_rejected(trained_teacher):
    _, model, diffusion, _ = trained_teacher
    with pytest.raises(ValueError, match="pin index"):
        diffusion.sample(model, n=2, pin={WINDOW: torch.zeros(LENGTH)})


def test_conditional_model_with_guidance_runs():
    torch.manual_seed(0)
    model = TrajectoryDenoiser(
        state_dim=LENGTH, window=WINDOW, cond_dim=6, d_model=32, nhead=2, num_layers=1
    )
    diffusion = GaussianDiffusion(num_steps=10)
    cond = torch.randn(4, 6)
    samples = diffusion.sample(
        model, n=4, cond=cond, guidance_scale=2.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert samples.shape == (4, WINDOW, LENGTH)
    # the null-condition path is also exercised directly
    loss = diffusion.training_loss(model, torch.randn(4, WINDOW, LENGTH), cond, cond_dropout=1.0)
    assert torch.isfinite(loss)


def test_quality_report_passes_on_trained_teacher(trained_teacher, tmp_path):
    dataset, model, diffusion, _ = trained_teacher
    start = torch.as_tensor(dataset.normalize(np.eye(LENGTH, dtype=np.float32)[0]))
    samples = diffusion.sample(
        model, n=32, pin={0: start}, generator=torch.Generator().manual_seed(4)
    )
    denorm = dataset.denormalize(samples.numpy())
    report = generation_quality_report(
        denorm,
        validator=OneHotValidator(),
        success_fn=lambda s, s0: bool(np.argmax(s) == LENGTH - 1),
        report_path=tmp_path / "quality.json",
    )
    assert report["invalid_state_rate"] <= 0.10
    assert report["shaping_enabled"] is True
    assert 0.0 <= report["endpoint_success_rate"] <= 1.0
    assert (tmp_path / "quality.json").is_file()


def test_quality_gate_violation_disables_shaping(trained_teacher):
    dataset, model, diffusion, _ = trained_teacher
    samples = diffusion.sample(model, n=8, generator=torch.Generator().manual_seed(5))
    report = generation_quality_report(
        dataset.denormalize(samples.numpy()), validator=AlwaysInvalidValidator()
    )
    assert report["invalid_state_rate"] == 1.0
    assert report["shaping_enabled"] is False
    assert report["gate_violations"]


def test_quality_bridge_consistency_gate():
    samples = np.eye(LENGTH, dtype=np.float32)[np.zeros((5, WINDOW), dtype=int)]
    report = generation_quality_report(
        samples, validator=OneHotValidator(), bridge_consistency_rate=0.5
    )
    assert report["shaping_enabled"] is False
    assert any("bridge_consistency" in v for v in report["gate_violations"])