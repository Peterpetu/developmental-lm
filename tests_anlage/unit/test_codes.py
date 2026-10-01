"""Acceptance tests I1-I12 of docs/anlage/02. SPEC: 02 §2.8"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anlage.codes import (
    CodeTables, bit_balance, channel_codes, frequency_rank, global_layer_code,
    layer_codes, role_codes, token_codes,
)
from anlage.config import AX_HEAD, AX_INNER, AX_RES, AX_TOK, GenomeShape, StageConfig
from anlage.latents import LatentBank, SIGMA_Z
from tests_anlage.conftest import make_tables

SHAPE = GenomeShape(d_h=32, H=2, d_g=16, d_z=4, d_y=4, d_sigma=8)


def test_I1_codomain_is_pm1():
    for c in (channel_codes(256, SHAPE), layer_codes(8, SHAPE), role_codes(SHAPE)):
        assert set(torch.unique(c).tolist()) <= {-1.0, 1.0}
    rank = torch.randperm(100)
    assert set(torch.unique(token_codes(100, SHAPE, rank)).tolist()) <= {-1.0, 1.0}


def test_I2_channel_map_is_injective():
    for d in (2, 16, 128, 1024, SHAPE.d_max):
        c = channel_codes(d, SHAPE)
        assert len({tuple(r.tolist()) for r in c}) == d


def test_I3_nesting_on_channels_and_tokens():
    small, big = channel_codes(64, SHAPE), channel_codes(512, SHAPE)
    assert torch.equal(small, big[:64])
    # A type keeps its code across V when its relative rank is unchanged: the
    # most frequent type at V=64 and at V=128.
    r64 = torch.arange(64)
    r128 = torch.arange(128)
    assert torch.equal(token_codes(64, SHAPE, r64)[0], token_codes(128, SHAPE, r128)[0])
    assert torch.equal(token_codes(64, SHAPE, r64)[63], token_codes(128, SHAPE, r128)[127])


def test_I4_spread_and_the_prefix_control():
    d = 1024
    bal = bit_balance(channel_codes(d, SHAPE)).abs()
    assert float(bal.max()) <= 3.0 / np.sqrt(d)
    # positive control: a prefix coding has constant top bits.
    from anlage.codes import bits_to_pm1
    prefix = bits_to_pm1(torch.arange(d), SHAPE.n_c)
    assert float(bit_balance(prefix).abs().max()) > 0.99


def test_I5_determinism_of_the_tables():
    a = channel_codes(300, SHAPE)
    b = channel_codes(300, GenomeShape(d_h=32, H=2, d_g=16, d_z=4, d_y=4, d_sigma=8))
    assert torch.equal(a, b)
    other = channel_codes(300, GenomeShape(d_h=32, H=2, d_g=16, d_z=4, d_y=4,
                                           d_sigma=8, seed_codes=7))
    assert not torch.equal(a, other)


def test_I6_gray_adjacency():
    L = 1 << SHAPE.n_L
    r = layer_codes(L, SHAPE)[:, :-1]
    for l in range(L - 1):
        assert int((r[l] != r[l + 1]).sum()) == 1
    # tokens: adjacent ranks differ in fewer than half as many bits as random.
    V = 1 << SHAPE.n_v
    e = token_codes(V, SHAPE, torch.arange(V))
    adj = (e[:-1] != e[1:]).float().sum(-1).mean()
    rnd = (e[torch.randperm(V)][:-1] != e[torch.randperm(V)][1:]).float().sum(-1).mean()
    assert adj == 1.0 and adj < 0.5 * rnd


def test_I7_relative_depth_and_the_global_flag():
    a, b = layer_codes(8, SHAPE), layer_codes(16, SHAPE)
    assert torch.equal(a[0], b[0]) and torch.equal(a[-1], b[-1])
    assert torch.all(a[:, -1] == 1.0)
    g = global_layer_code(SHAPE)
    assert float(g[0, -1]) == -1.0 and torch.all(g[0, :-1] == 0.0)


def test_I8_overflow_raises_and_names_the_width():
    with pytest.raises(ValueError, match="n_c"):
        channel_codes(SHAPE.d_max + 1, SHAPE)
    with pytest.raises(ValueError, match="n_v"):
        token_codes(SHAPE.v_max + 1, SHAPE, torch.arange(SHAPE.v_max + 1))
    with pytest.raises(ValueError, match="n_L"):
        layer_codes(SHAPE.L_max + 1, SHAPE)


def test_I9_zero_cost():
    cfg = StageConfig(0, d_model=16, L=4, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)
    t = make_tables(SHAPE, cfg)
    assert not any(isinstance(v, torch.nn.Parameter) for v in vars(t).values())


def test_I10_frequency_is_carried_and_the_bijection_control():
    V = 4096
    counts = np.exp(-np.arange(V) / 400.0) * 1e6 + 1
    rank = frequency_rank(torch.tensor(counts))
    e = token_codes(V, SHAPE, rank).double().numpy()
    y = np.log(counts)

    def r2(X):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        res = y - X1 @ beta
        return 1 - res.var() / y.var()

    assert r2(e) >= 0.3
    bij = channel_codes(V, SHAPE).double().numpy()          # the §02.2 control
    assert r2(bij) <= 0.05


def test_I11_latent_bank_shapes():
    cfg = StageConfig(0, d_model=16, L=4, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)
    bank = LatentBank(SHAPE, cfg, 64)
    t = bank.count_table()
    assert t["z_res"] == 16 * 4 and t["z_inner"] == 4 * 24 * 4
    assert t["z_head"] == 4 * 16 * 4 and t["z_gate"] == 4 * 4 * 4
    assert t["z_tok"] == 64 * 4 and t["y"] == 4 * 4
    assert bank.n_params() == sum(t.values())


def test_I12_sampling_has_the_axis_moments():
    cfg = StageConfig(0, d_model=512, L=2, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)
    bank = LatentBank(SHAPE, cfg, 64, seed=5)
    with torch.no_grad():
        bank.z_res.mul_(0.5).add_(2.0)
    g = torch.Generator().manual_seed(0)
    n = 4096
    s = bank.sample_new(AX_RES, n, None, g)
    mu, sd = bank.moments(AX_RES)
    assert torch.allclose(s.mean(0), mu, atol=4 / n ** 0.5 * float(sd.max()) + 1e-6)
    assert torch.allclose(s.std(0), sd, atol=4 / n ** 0.5 * float(sd.max()) + 1e-6)
    z0 = LatentBank(GenomeShape(d_h=32, H=2, d_g=16, d_z=0, d_y=4, d_sigma=8), cfg, 64)
    assert z0.sample_new(AX_INNER, 5, 0, g).shape == (5, 0)
    m = bank.sample_new(AX_HEAD, 7, 1, g, mode="mean")
    assert torch.allclose(m, bank.moments(AX_HEAD, 1)[0].expand(7, -1))
