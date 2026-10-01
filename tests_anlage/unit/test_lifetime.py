"""G-NOFREE-INVERSE: no decode runs inside a training step.

SPEC: 08 §8.4
SPEC: 04 §4.6

GNOMON decoded every weight on every step (§00.2); ANLAGE's training hot path
must be the free-weight baseline's. The sentinel replaces every decode entry
point of `Genome` with a function that raises; a lifetime must complete under
it, and a planted decode in a `before_step` hook must trip it (the positive
control that shows the sentinel is live).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
import torch

from anlage.data import N_STRATA, EvalSet
from anlage.genome import Genome
from anlage.lifetime import train
from anlage.model import Phenotype
from anlage.recipe import TrainConfig
from screen.crn import WindowPool

from tests_anlage.conftest import REPO, VOCAB

DECODE_ENTRY_POINTS = ("features", "decode_matrix", "decode_vector", "decode", "decode_all",
                       "decode_reference")


class _Shard:
    def __init__(self, tokens: np.ndarray):
        self.tokens = tokens


def _lifetime_pieces(tiny_stage, device: str):
    T = 16
    rng = np.random.default_rng(0)
    shard = _Shard(rng.integers(0, VOCAB, size=4000, dtype=np.int64))
    pool = WindowPool(shard, n_windows=64, T=T, pool_seed=1, device=device)
    win = torch.from_numpy(rng.integers(0, VOCAB, size=(8, T + 1), dtype=np.int64)).to(device)
    strata = torch.from_numpy(rng.integers(0, N_STRATA, size=(8, T), dtype=np.int64)).to(device)
    ev = EvalSet(windows=win, strata=strata, split="test", digest="synthetic")
    model = Phenotype(tiny_stage, VOCAB, seed=0)
    tc = TrainConfig(tokens=3 * 8 * T, batch_final=8, T=T, lr_provenance="test")
    return model, pool, ev, tc


def _arm_sentinel(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("G-NOFREE-INVERSE: a decode ran inside the lifetime")
    for name in DECODE_ENTRY_POINTS:
        assert hasattr(Genome, name), name
        monkeypatch.setattr(Genome, name, boom)


def test_G_NOFREE_INVERSE_lifetime_runs_with_every_decode_disabled(tiny_stage, monkeypatch):
    _arm_sentinel(monkeypatch)
    model, pool, ev, tc = _lifetime_pieces(tiny_stage, "cpu")
    res = train(model, pool, ev, tc, order_seed=0, compile_mode=None, device="cpu")
    assert res.tokens >= tc.tokens
    assert np.isfinite(res.final_ce)


def test_G_NOFREE_INVERSE_positive_control_planted_decode_is_caught(tiny_stage, monkeypatch):
    _arm_sentinel(monkeypatch)
    model, pool, ev, tc = _lifetime_pieces(tiny_stage, "cpu")
    G = Genome.__new__(Genome)

    def planted(step, tokens_done):
        G.decode_all(None)

    with pytest.raises(AssertionError, match="G-NOFREE-INVERSE"):
        train(model, pool, ev, tc, order_seed=0, compile_mode=None, device="cpu",
              hooks={"before_step": planted})


def test_G_NOFREE_INVERSE_hot_path_modules_do_not_import_the_genome():
    """The static half: the modules on the training path never import the decoder."""
    pat = re.compile(r"^\s*(from\s+\.?(anlage\.)?(genome|regress|grow|population)\b|import\s+anlage\.(genome|regress))",
                     re.M)
    for name in ("lifetime", "recipe", "model", "ops", "data"):
        src = (REPO / "anlage" / f"{name}.py").read_text(encoding="utf-8")
        assert not pat.search(src), f"anlage/{name}.py imports a decode-side module"


def test_G_NOFREE_INVERSE_sleep_hook_decodes_only_at_refits(tiny_stage, monkeypatch):
    """The sleep hook's per-step pull reads a cached decode; the decode itself
    runs only at refit steps (05 §5.7)."""
    from anlage.codes import CodeTables
    from anlage.config import GenomeShape
    from anlage.genome import DecodeContext, Genome
    from anlage.latents import LatentBank
    from anlage.sleep import Sleeper
    from tests_anlage.conftest import make_tables
    shape = GenomeShape(d_h=32, H=2, d_g=16, d_z=4)
    model, pool, ev, tc = _lifetime_pieces(tiny_stage, "cpu")
    G = Genome(shape, seed=1)
    bank = LatentBank(shape, tiny_stage, VOCAB, seed=1)
    ctx = DecodeContext(tiny_stage, make_tables(shape, tiny_stage, VOCAB), bank, VOCAB)
    # refit once, legitimately, so a cached decode exists; then arm the sentinel
    sl = Sleeper(G, ctx, model, tc, every=10_000, n_refit=2, eta_anchor=0.1)
    sl.refit(0)
    assert sl.D is not None and len(sl.records) == 1
    _arm_sentinel(monkeypatch)
    res = train(model, pool, ev, tc, order_seed=0, compile_mode=None, device="cpu",
                hooks={"before_step": sl.before_step})
    assert np.isfinite(res.final_ce) and len(sl.records) == 1
    # and a refit step under the sentinel is caught
    sl2 = Sleeper(G, ctx, model, tc, every=1, n_refit=1, eta_anchor=0.0)
    with pytest.raises(AssertionError, match="G-NOFREE-INVERSE"):
        train(model, pool, ev, tc, order_seed=0, compile_mode=None, device="cpu",
              hooks={"before_step": sl2.before_step})


def test_P14_gradient_accumulation_matches_the_full_batch_step(tiny_stage):
    """One optimiser step at batch 8 in one micro-batch and in two of 4 gives the same
    weights (fp32 on CPU; the losses are size-weighted means)."""
    import copy
    from anlage.recipe import micro_batch_for
    torch.manual_seed(0)
    model_a, pool, ev, tc = _lifetime_pieces(tiny_stage, "cpu")
    model_b = copy.deepcopy(model_a)
    tc_a = TrainConfig(tokens=8 * 16, batch_final=8, T=16, batch_min_frac=1.0, lr_provenance="test")
    tc_b = TrainConfig(tokens=8 * 16, batch_final=8, T=16, batch_min_frac=1.0, micro_batch=4, lr_provenance="test")
    train(model_a, pool, ev, tc_a, order_seed=1, compile_mode=None, device="cpu")
    train(model_b, pool, ev, tc_b, order_seed=1, compile_mode=None, device="cpu")
    for (k, a), (_, b) in zip(model_a.weights().items(), model_b.weights().items()):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), k
    assert micro_batch_for(13_000_000, 32) == 32 and micro_batch_for(36_000_000, 32) == 16 \
        and micro_batch_for(78_000_000, 32) == 8
