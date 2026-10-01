"""Tests for the screening harness.

Every test here corresponds to a specific way a screen could report a confident
number that means nothing. They are not tests that the code runs; they are tests
that the *measurement* is a measurement.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from screen.bauplan import GLOBAL, Bauplan, BodyModel
from screen.crn import WindowPool, eval_windows
from screen.nondegeneracy import AxisNotLive, assert_axis_is_live, logit_distance
from screen.variance import (
    decompose, deterministic_term_check, pooled_within_sd, seeds_for_power,
    searchworthy,
)


# --------------------------------------------------------------- accounting

@pytest.mark.parametrize("plan", [
    Bauplan.uniform(4, "attn"),
    Bauplan.uniform(4, "delta"),
    Bauplan.uniform(6, "attn", d_ff=768, n_h=8),
    Bauplan.uniform(4).replace(ops=("attn", "delta", "attn", "delta")),
    Bauplan.uniform(4, share=(0, 0, 1, 1)),
    Bauplan.uniform(4, share=(0, 1, 0, 1)),
    Bauplan.uniform(4, share=(0, 0, 0, 0)),
    Bauplan.uniform(4).replace(d_ff=(1024, 256, 1024, 256)),
])
def test_A1_accounting_is_exact(plan):
    """`n_unique_params` must equal what `express()` actually allocates.

    Exact equality, not a bound. The price a screen charges a body plan is this
    number; if it drifts from the model, the search optimises a fiction.
    """
    model = BodyModel(plan, seed=0)
    assert plan.n_unique_params() == model.n_params(include_embedding=False)


def test_A2_shared_banks_are_one_tensor_not_two_equal_ones():
    """Sharing must be object identity.

    A bug giving each layer its own copy would leave the parameter count right
    when counted with `set()` and turn every sharing measurement into a
    measurement of nothing.
    """
    m = BodyModel(Bauplan.uniform(4, share=(0, 0, 1, 1)), seed=0)
    assert m.bank_of(0)["q"] is m.bank_of(1)["q"]
    assert m.bank_of(2)["q"] is m.bank_of(3)["q"]
    assert m.bank_of(0)["q"] is not m.bank_of(2)["q"]
    # And a gradient reaching a shared bank is the SUM over its layers, which is
    # the mechanism by which sharing can hurt.
    m.zero_grad()
    m.loss(torch.randint(0, 4096, (2, 33))).backward()
    assert m.bank_of(0)["q"].grad is m.bank_of(1)["q"].grad


def test_A3_sharing_cuts_params_but_never_flops():
    base = Bauplan.uniform(4)
    shared = Bauplan.uniform(4, share=(0, 0, 1, 1))
    assert shared.flops_per_token(T=128) == base.flops_per_token(T=128)
    assert shared.n_unique_params() < base.n_unique_params()


def test_A4_matching_both_resources_forbids_sharing():
    """The algebra that deleted an earlier draft's largest genome component.

    Unique params `P = K*A`, forward FLOPs `F ~ L*A`. Matching FLOPs pins `A`;
    matching params then pins `K = L`. Any plan that matches a no-sharing
    reference on both resources has no sharing. Asserted here so that a future
    ablation protocol cannot quietly reintroduce it.
    """
    ref = Bauplan.uniform(4)
    T = 128
    for share in [(0, 0, 1, 1), (0, 1, 0, 1), (0, 0, 0, 0), (0, 0, 0, 1)]:
        for f in (256, 512, 1024, 1152, 2048):
            p = Bauplan.uniform(4, d_ff=f, share=share)
            matches_flops = p.flops_per_token(T=T) == ref.flops_per_token(T=T)
            matches_params = p.n_unique_params() == ref.n_unique_params()
            if matches_flops and matches_params:
                assert p.n_banks == p.L, (
                    f"share={share} f={f} matched both resources with "
                    f"{p.n_banks} banks over {p.L} layers -- the K=L algebra is "
                    f"broken, which means the accounting is wrong")


def test_A5_kv_cache_separates_attention_from_delta():
    """The descriptor must see what the operator choice is actually for."""
    T = 1024
    attn = Bauplan.uniform(4, "attn", window=GLOBAL)
    delta = Bauplan.uniform(4, "delta")
    local = Bauplan.uniform(4, "attn", window=128)
    assert attn.kv_cache_bytes(T=T) > local.kv_cache_bytes(T=T)
    assert local.kv_cache_bytes(T=T) > delta.kv_cache_bytes(T=T)
    # A delta layer's state does not grow with context; an attention layer's does.
    assert delta.kv_cache_bytes(T=T) == delta.kv_cache_bytes(T=8 * T)
    assert attn.kv_cache_bytes(T=8 * T) == 8 * attn.kv_cache_bytes(T=T)


def test_A6_flops_track_the_window_and_the_hand_derivation():
    """One plan's FLOPs, derived by hand, term by term."""
    p = Bauplan.uniform(1, "attn", d_model=128, d_ff=512, n_h=4, window=GLOBAL)
    T, d, V = 128, 128, 4096
    d_qk, d_kv = 4 * 32, 2 * 32
    qkvo = 2 * d * d_qk + 2 * 2 * d * d_kv + 2 * d_qk * d
    span = (T * (T + 1) / 2) / T                      # w >= T: mean span (T+1)/2
    attn = int(2 * 2 * 4 * 32 * span)
    ffn = 2 * 3 * d * 512
    head = 2 * d * V
    assert p.flops_per_token(T=T) == qkvo + attn + ffn + head
    # And a narrower window costs strictly less.
    assert (Bauplan.uniform(1, "attn", window=32).flops_per_token(T=T)
            < p.flops_per_token(T=T))


def test_A7_a_bank_shared_across_differing_shapes_is_rejected():
    with pytest.raises(ValueError, match="differing shape"):
        Bauplan(ops=("attn", "attn"), window=(GLOBAL,) * 2, d_ff=(512, 1024),
                bank=(0, 0), n_h=(4, 4))


# --------------------------------------------------------- non-degeneracy

def test_B1_the_window_trap_is_caught():
    """`window == T` is global, so it is not an axis. The exact 240/256 defect."""
    base = Bauplan.uniform(4, "attn", window=GLOBAL)
    with pytest.raises(AxisNotLive, match="INERT"):
        assert_axis_is_live(base, base.replace(window=(128,) * 4),
                            axis="window", T=128, device="cpu")


def test_B2_the_guard_fires_in_both_directions():
    """A control never shown to fire in both directions is not a control."""
    base = Bauplan.uniform(4, "attn", window=GLOBAL)
    assert logit_distance(base, base.replace(window=(128,) * 4),
                          T=128, device="cpu") == 0.0
    for other, name in [
        (base.replace(window=(32,) * 4), "w32"),
        (base.replace(window=(64,) * 4), "w64"),
        (base.replace(ops=("delta",) * 4), "delta"),
        (base.replace(d_ff=(1024,) * 4), "wide"),
        (base.replace(bank=(0, 0, 1, 1)), "shared"),
    ]:
        assert assert_axis_is_live(base, other, axis=name, T=128,
                                   device="cpu") > 1e-3


def test_B3_the_same_window_becomes_live_when_T_grows():
    """Confirms the diagnosis: the axis was inert because of `T`, not the code."""
    base = Bauplan.uniform(4, "attn", window=GLOBAL)
    narrow = base.replace(window=(128,) * 4)
    assert logit_distance(base, narrow, T=128, device="cpu") == 0.0
    assert logit_distance(base, narrow, T=256, device="cpu") > 1e-3


def test_B4_identical_plans_are_rejected_rather_than_passed():
    base = Bauplan.uniform(4)
    with pytest.raises(AxisNotLive, match="identical"):
        assert_axis_is_live(base, Bauplan.uniform(4), axis="none", device="cpu")


# ------------------------------------------------------ common random numbers

class _FakeShard:
    def __init__(self, n=50_000, vocab=4096, seed=0):
        import numpy as np
        self.tokens = np.random.default_rng(seed).integers(
            0, vocab, size=n, dtype=np.int64)


def test_C1_same_seed_gives_identical_data_across_arms():
    """The CRN contract, and the whole reason a paired contrast is valid."""
    pool = WindowPool(_FakeShard(), n_windows=512, T=32, pool_seed=7,
                      device="cpu")
    a = list(pool.batches(batch=8, steps=5, seed=3))
    b = list(pool.batches(batch=8, steps=5, seed=3))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_C2_different_seeds_give_different_order_but_the_same_pool():
    pool = WindowPool(_FakeShard(), n_windows=512, T=32, pool_seed=7,
                      device="cpu")
    a = torch.cat(list(pool.batches(batch=8, steps=10, seed=0)))
    b = torch.cat(list(pool.batches(batch=8, steps=10, seed=1)))
    assert not torch.equal(a, b)
    # Same windows, different order: every row of `a` is a row of the pool.
    rows = {tuple(r.tolist()) for r in pool.windows}
    assert all(tuple(r.tolist()) in rows for r in a)
    assert all(tuple(r.tolist()) in rows for r in b)


def test_C3_the_pool_depends_only_on_the_pool_seed():
    p1 = WindowPool(_FakeShard(), n_windows=256, T=16, pool_seed=11, device="cpu")
    p2 = WindowPool(_FakeShard(), n_windows=256, T=16, pool_seed=11, device="cpu")
    p3 = WindowPool(_FakeShard(), n_windows=256, T=16, pool_seed=12, device="cpu")
    assert p1.digest() == p2.digest()
    assert p1.digest() != p3.digest()


def test_C4_wrapping_reshuffles_rather_than_repeating():
    """A run longer than one epoch must not replay the first epoch verbatim."""
    pool = WindowPool(_FakeShard(), n_windows=64, T=8, pool_seed=1, device="cpu")
    b = list(pool.batches(batch=16, steps=8, seed=0))     # 2 epochs of 4 steps
    first, second = torch.cat(b[:4]), torch.cat(b[4:])
    assert not torch.equal(first, second)


def test_C5_eval_windows_are_the_head_and_never_sampled():
    import numpy as np
    sh = _FakeShard(n=1000)
    ev = eval_windows(sh, n_windows=4, T=9, device="cpu")
    assert ev.shape == (4, 10)
    assert torch.equal(ev.reshape(-1), torch.from_numpy(sh.tokens[:40]))


# ------------------------------------------------------------- the statistics

def test_D1_pooled_sd_weights_by_degrees_of_freedom():
    """A 2-seed arm must not count as much as an 8-seed one."""
    table = {"a": [1.0, 1.0], "b": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    sd, df = pooled_within_sd(table)
    assert df == 1 + 7
    ss = 0.0 + sum((v - 3.5) ** 2 for v in table["b"])
    assert sd == pytest.approx(math.sqrt(ss / 8))


def test_D2_power_formula_matches_the_textbook_constants():
    # two-sample: n >= 2 (z_.975 + z_.8)^2 sigma^2 / delta^2 = 15.70 sigma^2/d^2
    assert seeds_for_power(1.0, 1.0, paired=False) == 16
    # paired drops the factor of two
    assert seeds_for_power(1.0, 1.0, paired=True) == 8
    # a below-noise effect demands many seeds
    assert seeds_for_power(0.0034, 0.0135, paired=False) > 200


def test_D3_sigma_G_is_corrected_for_measurement_noise():
    """Arm means carry `sigma_e^2/n` of noise; the raw spread overstates sigma_G.

    Arms drawn from one distribution must not report a positive genetic
    variance just because each mean was measured with error.
    """
    g = torch.Generator().manual_seed(0)
    table, seeds = {}, {}
    for i in range(12):
        vals = (torch.randn(8, generator=g) * 0.02).tolist()
        table[f"arm{i}"] = vals
        seeds[f"arm{i}"] = list(range(8))
    dec = decompose(table, seeds)
    assert dec.sigma_g < dec.sigma_e, (dec.sigma_g, dec.sigma_e)
    assert dec.heritability < 0.25


def test_D4_a_deterministic_price_is_flagged_when_it_dominates():
    """The named third failure mode: evolution succeeding at the wrong task."""
    # Arms with real genetic separation (0.2 nats apart) against 0.007 of noise.
    table = {"a": [2.00, 2.01], "b": [2.20, 2.21], "c": [2.40, 2.41]}
    seeds = {k: [0, 1] for k in table}
    dec = decompose(table, seeds)
    assert dec.sigma_g > dec.sigma_e
    prices = [1e6, 1.1e6, 0.9e6]
    assert not deterministic_term_check(prices, 1e-12, dec)["price_dominates"]
    assert deterministic_term_check(prices, 1e-5, dec)["price_dominates"]


def test_D4b_a_price_dominates_whenever_there_is_no_genetic_signal():
    """The flag must fire hardest exactly where the landscape is flat.

    If the genome does not move the loss, *any* positive price coefficient makes
    the price the only thing selection can see. That is not an edge case to be
    smoothed over -- it is the alarm doing its job, and it fires on a landscape
    the earlier draft of this work would have searched for six weeks.
    """
    table = {"a": [2.000, 2.010], "b": [2.005, 2.015], "c": [2.002, 2.012]}
    seeds = {k: [0, 1] for k in table}
    dec = decompose(table, seeds)
    assert dec.sigma_g == 0.0          # noise-corrected: no genetic variance
    out = deterministic_term_check([1e6, 1.1e6, 0.9e6], 1e-12, dec)
    assert out["price_dominates"]
    assert out["ratio_price_to_genetic"] == float("inf")


def test_D5_searchworthy_rejects_a_large_but_hill_climbable_axis():
    """The gate's whole point: 14.8 sigma that a hill-climb solves is not a search."""
    table = {"a": [2.0, 2.0], "b": [1.8, 1.8]}
    seeds = {"a": [0, 1], "b": [0, 1]}
    # Give the arms a little within-arm spread so sigma_e is estimable.
    table = {"a": [2.00, 2.01], "b": [1.80, 1.81]}
    dec = decompose(table, seeds, contrasts=[("big axis", "a", "b")])
    assert dec.contrasts[0].d_over_sigma_e > 3.0
    assert not searchworthy(dec, hill_climbable={"big axis": True})["passes"]
    assert searchworthy(dec, hill_climbable={"big axis": False})["passes"]
