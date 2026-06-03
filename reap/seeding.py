"""Deterministic seeding for python, numpy, and torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNG sources used by the codebase.

    With ``deterministic_torch`` enabled, cuDNN benchmarking is disabled and
    deterministic kernels are requested so same-seed runs produce identical
    metric streams.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_rng(seed: int) -> np.random.Generator:
    """An isolated numpy generator for components that need their own stream."""
    return np.random.default_rng(seed)
