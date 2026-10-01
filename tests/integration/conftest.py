"""Shared fixtures for the end-to-end integration tests.

The unit suite exercises every mechanism in isolation. What it cannot show is
that the mechanisms *compose*: that a lifetime built from all of them runs to
completion, that every one of them actually fires, and that two lifetimes with
the same seed are the same lifetime. Those are the three claims here.

The configuration is a deliberately **narrow** SOMA — `d_model = 128`,
`U_max = 16`, `T_ctx = 64` — chosen so a full lifetime with selection, birth,
death, growth, the archive and certification completes in minutes on CPU. It is
narrow, not short and not shallow, and both of those are forced rather than
chosen:

* `phi_ann` has a floor of 50 steps and `validate()` requires `phi_ann < T_0`,
  which together put the **shortest lifetime SOMA admits at all** at about 1400
  steps. A 100-step integration config cannot be constructed.
* `L` stays at M-nano's 6. §02.5's gradient-flow band `[0.60, 1.70]` is a
  statistic over `L - 1` per-block ratios; at `L = 2` it is a single number with
  no median to take, and a config that fails the acceptance test would tell us
  nothing about the code under test.

Every structural constant that governs *whether* a mechanism fires (`T_0`,
`phi_ann`, `M_mat`, `T_arch`, `T_cert`, the growth schedule) is derived by the
same formulas from the same `SomaConfig`, so a mechanism that cannot fire here
cannot fire at M1 either.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import numpy as np
import pytest
import torch

from soma.config import M_NANO, PolicyConstants, SomaConfig
from soma.data import TokenizedCorpus, build_tokenizer, tokenize_corpus
from soma.data.stream import TrainingStream

DOCS = [
    "Once upon a time there was a little girl named Lily who liked the park.",
    "Tom and Sara went to the shop and bought a red ball and a blue kite.",
    "The cat sat on the mat and looked at the dog, and the dog barked.",
    "A small boy found a shiny stone near the river and showed his mother.",
] * 300

VOCAB = 512


def tiny_config() -> SomaConfig:
    """The smallest configuration that still validates and still evolves."""
    # `phi_ann = min(5000, max(50, 0.02 N_step))` has a floor of 50 steps, and
    # `validate()` requires `phi_ann < T_0`. Solving the two together gives
    # `N_step > ~1400` as the **smallest lifetime SOMA admits at all** — so the
    # integration configuration is tiny in *width* (`L = 2`, `d_model = 64`,
    # `U_max = 16`) and full-length in *steps*. Shortening the run instead would
    # produce a config that cannot be constructed, which is the correct
    # behaviour and not a thing to work around.
    pol = replace(M_NANO.policy, U_cap=12, M_mat=8, rho_0=0.15, n_par=4)
    cfg = replace(
        M_NANO,
        name="m-tiny",
        L=6,
        attn_period=3,
        d_model=128,
        n_h=4,
        n_kv=2,
        n_m=2,
        d_k=32,
        d_v=64,
        e_v=1,
        C_chunk=16,
        T_ctx=64,
        V=VOCAB,
        D_tok=128 * 1400,
        B_tok=128,
        k_route=2,
        U_max=16,
        side=4,
        U_0=8,
        d_key=16,
        policy=pol,
        eval_lengths=(40,),
    )
    cfg.validate()
    return cfg


@pytest.fixture(scope="session")
def tiny_corpus():
    td = tempfile.mkdtemp()
    tok = build_tokenizer(DOCS, VOCAB, out_path=os.path.join(td, "tok.json"))
    tokenize_corpus(DOCS, tok, td, "train", progress_every=0)
    corpus = TokenizedCorpus(td, "train")
    yield corpus
    corpus.close()


@pytest.fixture(scope="session")
def tiny_setup(tiny_corpus):
    """`(cfg, corpus, eval_batches, probe)` — everything a lifetime needs."""
    cfg = tiny_config()
    evs = TrainingStream(cfg, tiny_corpus, None, rows_per_batch=2,
                         rng=np.random.default_rng(3))
    eval_batches = [evs.batch(0, 0.0) for _ in range(2)]
    probe = torch.from_numpy(
        tiny_corpus.gather(tiny_corpus.windows(cfg.T_ctx)[:4], cfg.T_ctx)[0]
    )
    return cfg, tiny_corpus, eval_batches, probe
