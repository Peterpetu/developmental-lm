"""Shared fixtures for the TEMPO suite: the `TINY` shape, seeded generators, CPU only.

Every acceptance row of docs/tempo runs on CPU in seconds at `TINY`; the GPU
rows (P12, N6) are measured by the run scripts, not here.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from tempo.config import TINY, TS_ADULT, TempoConfig

REPO = pathlib.Path(__file__).resolve().parent.parent


def corpus_available(root: str = "data/ts_8192") -> bool:
    r = REPO / root
    return (r / "validation.index.json").exists() and (r / "train.index.json").exists()


def requires_corpus(root: str = "data/ts_8192"):
    return pytest.mark.skipif(not corpus_available(root), reason=f"{root} is absent")


def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def tiny() -> TempoConfig:
    return TINY


@pytest.fixture
def adult() -> TempoConfig:
    return TS_ADULT


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    yield
