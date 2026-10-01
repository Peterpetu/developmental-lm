"""Shared fixtures for the ANLAGE suite.

Small shapes, so the whole suite runs in seconds on CPU; the reference decode of
D1 is `O(rows * cols * H)` Python loops and is exercised on tiny matrices only.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from anlage.codes import CodeTables
from anlage.config import GenomeShape, StageConfig
from anlage.latents import LatentBank

REPO = pathlib.Path(__file__).resolve().parent.parent


def corpus_available(root: str = "data/ts_8192") -> bool:
    r = REPO / root
    return (r / "validation.index.json").exists() and (r / "train.index.json").exists()


def requires_corpus(root: str = "data/ts_8192"):
    return pytest.mark.skipif(
        not corpus_available(root),
        reason=f"{root} is absent -- rebuilt with scripts/build_tokenizer.py + prepare_data.py")


def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def small_shape() -> GenomeShape:
    return GenomeShape(d_h=32, H=2, d_g=16, d_z=4, d_y=4, d_sigma=8)


@pytest.fixture
def tiny_stage() -> StageConfig:
    return StageConfig(0, d_model=16, L=4, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)


@pytest.fixture
def tiny_stage_next() -> StageConfig:
    return StageConfig(1, d_model=24, L=6, n_h=3, n_kv_attn=1, d_ff=32, d_head=8, emb_scale=4.0)


VOCAB = 64


@pytest.fixture
def vocab() -> int:
    return VOCAB


def make_tables(shape: GenomeShape, cfg: StageConfig, vocab: int = VOCAB, device="cpu"):
    rank = torch.randperm(vocab, generator=torch.Generator().manual_seed(1))
    return CodeTables(shape, vocab=vocab, rank=rank, L=cfg.L, device=device)


@pytest.fixture
def tiny_ctx(small_shape, tiny_stage, vocab):
    from anlage.genome import DecodeContext
    tables = make_tables(small_shape, tiny_stage, vocab)
    bank = LatentBank(small_shape, tiny_stage, vocab, seed=3)
    return DecodeContext(tiny_stage, tables, bank, vocab)
