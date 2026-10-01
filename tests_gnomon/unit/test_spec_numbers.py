"""G-SPECNUM: every number the specification states, recomputed from the code.

SPEC: 01 §1.1
SPEC: 01 §1.7
SPEC: 02 §2.2
SPEC: 03 §3.5
SPEC: 04 §4.2
SPEC: 04 §4.4
SPEC: 04 §4.5
SPEC: 07 §7.2
SPEC: 10 §10.4

SOMA carried a `G-SPECNUM` gate for exactly this and GNOMON did not, with the
result that the specification's headline quantity -- the compression ratio --
was stated three times in three places and agreed with the code in none of them:
§01.7 claimed `1 : 11,200` at M1, §03.5's table `10,830x`, §07.2 `10,324x`, and
the code computes **9,458x**. §03.5 also carried `|Gamma_a| ~ 550` against
§07.2's exact 796, and §02.2 still declared `n_c = 10` after `d_ff = 2048` at M1
forced it to 12.

None of these were caught because the only test that touched them asserted
`ratio > 100` and that the ratio improves with scale -- both true of every wrong
number as well as of the right one. A test that cannot fail is not a test, so
each constant below is asserted **exactly**, against a value recomputed from the
module tree rather than restated from the document.

The rule this file enforces: a number appears in the specification only if it
appears here, and it appears here only as an equality.
"""

from __future__ import annotations

import math

import pytest

from gnomon.config import (
    CONFIGS, M1, M_NANO, M_PICO, N_ROLES, N_RULE, GenomeShape, PhenoConfig,
    assert_scale_free,
)
from gnomon.genome.arch import D_A, D_GLOB, MENU_F, MENU_H, MENU_L, MENU_W, ArchHead
from gnomon.genome.decoder import Decoder
from gnomon.pheno.plan import Arch
from gnomon.qd.evolve import Genome

SHAPE = GenomeShape()


# ------------------------------------------------------------------ §02.2

def test_code_widths_are_the_spec_widths():
    """SPEC: 02 §2.2. `n_c = 12`, not the 10 the section carried until v1.1.

    `n_c` addresses every channel-*like* axis, not only the residual stream, and
    §03.5 records why: one code table is shared by the residual stream and the
    widest layer-internal axis, and `d_ff = 2048` at M1 -- doubled to 4096 by
    `MENU_F`'s top entry -- is the binding one. At `n_c = 10` a genome decoding
    `d_ff = 2048` would have raised.
    """
    assert (SHAPE.n_c, SHAPE.n_v, SHAPE.n_L, SHAPE.n_M) == (12, 14, 5, 4)
    assert SHAPE.d_max == 4096 and SHAPE.v_max == 16384
    assert 2 ** SHAPE.n_L == 32
    assert 2 ** SHAPE.n_M == N_ROLES == 16


def test_the_widest_addressable_axis_is_reachable_at_every_configuration():
    """SPEC: 02 §2.2. The bound has to hold for what `Gamma_a` may decode.

    Not merely for the configuration's own `d_ff`: the architecture head may
    decode `MENU_F`'s top entry, so the code table must address `8 * d_model`.
    """
    for cfg in CONFIGS.values():
        widest = max(max(MENU_F) * cfg.d_model,
                     int(max(MENU_H) * cfg.n_h) * cfg.d_head,
                     cfg.d_model, cfg.d_ff)
        assert widest <= SHAPE.d_max, (cfg.name, widest, SHAPE.d_max)
        assert cfg.vocab <= SHAPE.v_max, cfg.name
    assert_scale_free(SHAPE, list(CONFIGS.values()))


# ------------------------------------------------------------------ §03.5

#: SPEC: 03 §3.5's table, entry by entry, at the default shape.
GAMMA_W_TABLE = {
    "W_row": 896, "W_col": 928, "W_tok": 960, "W_ctx": 544, "c1": 32,
    "W_2": 1024, "c2": 32, "w_out": 32, "o_0": 1, "w_sig": 32, "sig_0": 1,
    "log_gain": 1,
}
GAMMA_W = 4483


def test_gamma_w_is_the_spec_table_tensor_by_tensor():
    """SPEC: 03 §3.5. Not just the total -- the total can be right by accident."""
    got = {n: p.numel() for n, p in Decoder(SHAPE).named_parameters()}
    assert got == GAMMA_W_TABLE
    assert sum(GAMMA_W_TABLE.values()) == GAMMA_W == SHAPE.n_params()


def test_gamma_w_entries_follow_their_defining_expressions():
    """SPEC: 03 §3.5. Each entry recomputed from `(d_h, n_c, n_v, n_L, n_M, d_z, d_y)`.

    A table of literals agreeing with a table of literals proves nothing; these
    are the expressions the section gives, so changing `d_z` moves both sides.
    """
    s = SHAPE
    d_h, d_z = s.d_h, s.d_z
    assert GAMMA_W_TABLE["W_row"] == d_h * (s.n_c + d_z)
    assert GAMMA_W_TABLE["W_col"] == d_h * (s.n_c + 1 + d_z)     # +1: §02.6's flag
    assert GAMMA_W_TABLE["W_tok"] == d_h * (s.n_v + d_z)
    assert GAMMA_W_TABLE["W_ctx"] == d_h * (s.n_L + s.n_M + s.d_y)
    assert GAMMA_W_TABLE["W_2"] == d_h * d_h
    for name in ("c1", "c2", "w_out", "w_sig"):
        assert GAMMA_W_TABLE[name] == d_h
    for name in ("o_0", "sig_0", "log_gain"):
        assert GAMMA_W_TABLE[name] == 1


@pytest.mark.parametrize("d_z", [0, 2, 4, 8, 16, 32, 64])
def test_H_log_N_one_more_address_bit_costs_a_fixed_price(d_z):
    """SPEC: 03 §3.5 [DERIVED]. `+2 d_h` per channel bit, `+d_h` per vocab bit.

    The claim that makes the genome scale-free is a *derivative*, so it is tested
    as one, at every `d_z` on §06.5's frontier: the price of an address bit must
    not depend on the capacity dial.
    """
    base = GenomeShape(d_z=d_z)
    assert (GenomeShape(d_z=d_z, n_c=base.n_c + 1).n_params()
            - base.n_params()) == 2 * base.d_h
    assert (GenomeShape(d_z=d_z, n_v=base.n_v + 1).n_params()
            - base.n_params()) == base.d_h


# ------------------------------------------------------------------ §07.2

GAMMA_A_TABLE = {
    "A_1": 208, "a_c1": 16, "A_2": 256, "a_c2": 16, "A_op": 32, "b_op": 2,
    "A_h": 64, "b_h": 4, "A_f": 64, "b_f": 4, "A_w": 64, "b_w": 4,
    "A_d": 48, "b_d": 6, "g_glob": 8,
}
GAMMA_A = 796
GAMMA = GAMMA_W + GAMMA_A                       # 5,279
N_STRATEGY = 27                                 # SPEC: 10 §10.4
GAMMA_TOTAL = GAMMA + N_STRATEGY                # 5,306, what is actually stored


def test_gamma_a_is_the_spec_table_tensor_by_tensor():
    """SPEC: 07 §7.2. §03.5's `~550` was never right; the head is 796."""
    got = {n: p.numel() for n, p in ArchHead(SHAPE).named_parameters()}
    assert got == GAMMA_A_TABLE
    assert sum(GAMMA_A_TABLE.values()) == GAMMA_A


def test_gamma_a_entries_follow_their_defining_expressions():
    """SPEC: 07 §7.2, with `d_a = 16` and `g_glob in R^8`."""
    assert (D_A, D_GLOB) == (16, 8)
    assert GAMMA_A_TABLE["A_1"] == D_A * (SHAPE.n_L + D_GLOB)
    assert GAMMA_A_TABLE["A_2"] == D_A * D_A
    assert GAMMA_A_TABLE["A_op"] == 2 * D_A
    for key, menu in (("A_h", MENU_H), ("A_f", MENU_F), ("A_w", MENU_W)):
        assert GAMMA_A_TABLE[key] == len(menu) * D_A
        assert len(menu) == 4
    assert GAMMA_A_TABLE["A_d"] == len(MENU_L) * D_GLOB
    assert len(MENU_L) == 6


def test_the_whole_genome_is_5279_plus_the_27_self_adapted_strengths():
    """SPEC: 07 §7.2 T1 and SPEC: 10 §10.4.

    §07.2 fixes `|Gamma| = 5,279`; §10.4 adds one `log_sigma` per genome tensor
    -- twelve in `Gamma_w`, fifteen in `Gamma_a` -- and those are inherited, so
    the object that is stored, mutated and crossed over is 5,306 reals. Both
    numbers are pinned because the specification uses both.
    """
    g = Genome(SHAPE)
    assert g.w.n_params() == GAMMA_W
    assert g.a.n_params() == GAMMA_A
    assert g.w.n_params() + g.a.n_params() == GAMMA == 5279
    assert len(GAMMA_W_TABLE) == 12 and len(GAMMA_A_TABLE) == 15
    assert g.s.n_params() == N_STRATEGY == 27
    assert g.n_params() == GAMMA_TOTAL == 5306


def test_the_genome_is_scale_free_across_every_configuration():
    """SPEC: 03 §3.5 D8. Not one entry moves when the phenotype does."""
    sizes = set()
    for cfg in CONFIGS.values():
        assert_scale_free(SHAPE, [cfg])
        sizes.add(Genome(SHAPE).n_params())
    assert sizes == {GAMMA_TOTAL}


# ------------------------------------------------------------------ §04.4

#: SPEC: 04 §4.4's table. Recomputed under `Arch.uniform`, the architecture the
#: configuration itself names -- alternating delta and attention layers.
#:
#: The document carried `1.57 M / 3.4 M / 54.5 M` and none of the three is what
#: the module tree produces. The measured values are authoritative and the
#: document is corrected to them; a specification number that disagrees with the
#: code is a defect in the specification, since the code is what runs.
DECODED = {"m-pico": 1_528_969, "m-nano": 4_060_883, "m1": 49_926_705}
SOMA_SIZE = {"m-pico": 67_616, "m-nano": 68_656, "m1": 139_360}


@pytest.mark.parametrize("name", sorted(DECODED))
def test_decoded_parameter_counts_are_exact(name):
    """SPEC: 04 §4.4, SPEC: 03 §3.5."""
    cfg = CONFIGS[name]
    assert cfg.n_decoded() == DECODED[name]
    assert Arch.uniform(cfg).n_decoded(cfg) == DECODED[name]


@pytest.mark.parametrize("name", sorted(SOMA_SIZE))
def test_soma_sizes_are_exact(name):
    """SPEC: 04 §4.5. `|Z| = (d_model + V) d_z + L d_y`."""
    cfg = CONFIGS[name]
    assert cfg.n_soma(SHAPE) == SOMA_SIZE[name]
    assert (cfg.n_soma(SHAPE)
            == (cfg.d_model + cfg.vocab) * SHAPE.d_z + cfg.L * SHAPE.d_y)


def test_the_m1_soma_is_the_139360_reals_the_spec_enumerates():
    """SPEC: 04 §4.5's table, row by row."""
    assert M1.d_model * SHAPE.d_z == 8_192
    assert M1.vocab * SHAPE.d_z == 131_072
    assert M1.L * SHAPE.d_y == 96
    assert 8_192 + 131_072 + 96 == 139_360


# --------------------------------------------------- the compression claim

#: SPEC: 01 §1.7, SPEC: 03 §3.5, SPEC: 07 §7.2 -- `|phenotype| : |Gamma|`.
#: Stated against 5,279, the genome §07.2 defines. §01.7's `1 : 11,200`,
#: §03.5's `10,830x` and §07.2's `10,324x` were all overstatements of the
#: same quantity; it is 9,458.
RATIO = {"m-pico": 289.6, "m-nano": 769.3, "m1": 9457.6}


@pytest.mark.parametrize("name", sorted(RATIO))
def test_compression_ratios_are_exact_not_merely_large(name):
    """SPEC: 03 §3.5's ratio column.

    The test this replaces asserted `ratio > 100` and monotonicity in scale,
    which every wrong value in the document also satisfied. That is how a
    15%-overstated headline claim survived a green suite.
    """
    got = CONFIGS[name].n_decoded() / GAMMA
    assert got == pytest.approx(RATIO[name], abs=0.05), (name, got)


def test_compression_improves_with_scale():
    """SPEC: 03 §3.5. The property an `O(m+n)` factorisation cannot deliver."""
    assert RATIO["m1"] > RATIO["m-nano"] > RATIO["m-pico"]
    # The genome does not grow at all between them; only the phenotype does.
    assert len({Genome(SHAPE).n_params() for _ in CONFIGS}) == 1


def test_the_shipped_artifact_is_gamma_plus_soma():
    """SPEC: 03 §3.7 [NORMATIVE — inference]. `|Gamma| + |Z|` in float32."""
    kb = (GAMMA_TOTAL + M1.n_soma(SHAPE)) * 4 / 1024
    assert GAMMA_TOTAL * 4 / 1024 == pytest.approx(20.7, abs=0.1)   # ~21 KB
    assert M1.n_soma(SHAPE) * 4 / 1024 == pytest.approx(544.4, abs=0.1)
    assert kb == pytest.approx(565.1, abs=0.2)


# ------------------------------------------------------------------ §04.2

def test_the_role_table_is_sixteen_roles_and_their_shapes():
    """SPEC: 04 §4.2. Fixed for the life of the project."""
    from gnomon.config import (
        ROLE_DELTA_BETA, ROLE_EMB, ROLE_FFN_DOWN, ROLE_FFN_GATE, ROLE_FFN_UP,
        ROLE_K, ROLE_LOGIT_SCALE, ROLE_NAMES, ROLE_NORM_FFN, ROLE_NORM_FINAL,
        ROLE_NORM_MIX, ROLE_O, ROLE_PLAST_CHAN, ROLE_PLAST_TOK, ROLE_Q,
        ROLE_SINK, ROLE_V,
    )
    assert len(ROLE_NAMES) == N_ROLES == 16
    assert sorted(ROLE_NAMES) == list(range(16))
    c = M1
    assert c.role_shape(ROLE_Q) == (c.n_h * c.d_head, c.d_model)
    assert c.role_shape(ROLE_K) == (c.n_kv * c.d_head, c.d_model)
    assert c.role_shape(ROLE_V) == (c.n_kv * c.d_head, c.d_model)
    assert c.role_shape(ROLE_O) == (c.d_model, c.n_h * c.d_head)
    assert c.role_shape(ROLE_FFN_GATE) == (c.d_ff, c.d_model)
    assert c.role_shape(ROLE_FFN_UP) == (c.d_ff, c.d_model)
    assert c.role_shape(ROLE_FFN_DOWN) == (c.d_model, c.d_ff)
    assert c.role_shape(ROLE_EMB) == (c.vocab, c.d_model)
    for r in (ROLE_NORM_MIX, ROLE_NORM_FFN, ROLE_NORM_FINAL):
        assert c.role_shape(r) == (c.d_model,)
    assert c.role_shape(ROLE_SINK) == (c.n_h,)
    assert c.role_shape(ROLE_LOGIT_SCALE) == (1,)
    # SPEC: 10 §10.3 -- the two plasticity roles are `(n, 4)`, a versioned
    # shape change from v0.9's vectors.
    assert c.role_shape(ROLE_PLAST_CHAN) == (c.d_model, N_RULE)
    assert c.role_shape(ROLE_PLAST_TOK) == (c.vocab, N_RULE)
    assert N_RULE == 4
    # `DELTA_BETA` packs alpha and beta, which is why it is `2 n_h` and not
    # `n_h`; the section's own table says `n_h*d_head x d_model`, and the code
    # is what the decoder and the mixer agree on.
    assert c.role_shape(ROLE_DELTA_BETA) == (2 * c.n_h, c.d_model)


def test_every_role_is_used_by_some_decoded_architecture():
    """SPEC: 02 §2.2 says "12 used"; all 16 are.

    Roles 0-3 are shared between the operators, `SINK` belongs to attention and
    `DELTA_BETA` to the mixer (§04.2), so no single layer uses all of them --
    but the union over the two operator choices is the whole table, and a role
    no architecture can reach would be dead genome.
    """
    used = set()
    for ops in (("attn",), ("delta",)):
        a = Arch(ops, (M_PICO.n_h,), (M_PICO.d_ff,), (M_PICO.window,))
        used |= {r for r, _ in a.decode_keys()}
    assert used == set(range(N_ROLES)), sorted(set(range(N_ROLES)) - used)


# ------------------------------------------------------------------ §01.1

def test_the_constitution_table_agrees_with_the_modules():
    """SPEC: 01 §1.1. The three parameter classes, sized at M1.

    §01.1 gave germline `~4,850`, which predates §07.2's head and §10.4's
    strengths both.
    """
    assert Genome(SHAPE).n_params() == GAMMA_TOTAL          # germline
    assert M1.n_soma(SHAPE) == SOMA_SIZE["m1"]              # soma
    # innate: a pure function of an index, costing zero.
    from gnomon.genome import codes as C
    for fn, args in ((C.channel_codes, (M1.d_model, SHAPE)),
                     (C.token_codes, (M1.vocab, SHAPE)),
                     (C.layer_codes, (M1.L, SHAPE)),
                     (C.role_codes, (SHAPE,))):
        t = fn(*args)
        assert not t.requires_grad and t.numel() > 0


# ------------------------------------------------------------------ §05.4

def test_the_lifetime_constants_are_the_spec_constants():
    """SPEC: 05 §5.4, §5.5."""
    from gnomon.lifetime.plasticity import (A_SCALE, B_SCALE, C_SCALE, D_SCALE,
                                            ADAM_POINT)
    from gnomon.genome.decoder import DEFAULT_GAIN
    from gnomon.genome.arch import SIGMA_FLOOR

    for cfg in CONFIGS.values():
        assert cfg.c_eta == 1.5                       # eta in [0.223, 4.48]
        assert cfg.lam_ratio == 0.04                  # the paper's 500/12500
        assert cfg.lam == pytest.approx(0.04 * cfg.N_life)
    assert math.exp(-1.5) == pytest.approx(0.2231, abs=1e-4)
    assert math.exp(1.5) == pytest.approx(4.4817, abs=1e-4)
    # SPEC: 10 §10.3 -- B centred on 1 so generation 0 is Adam exactly.
    assert (A_SCALE, B_SCALE, C_SCALE, D_SCALE) == (1.0, 1.0, 0.1, 1.0)
    assert ADAM_POINT == (0.0, 1.0, 0.0, 0.0)
    # SPEC: 03 §3.6 -- measured, and inside its own stated valid range [2, 8].
    assert DEFAULT_GAIN == 4.0 and 2.0 <= DEFAULT_GAIN <= 8.0
    # SPEC: 07 §7.5 -- the floor must be well below the tensors it protects,
    # and it exists *for* the tensors below it. `c1`, `c2`, `w_sig`, `sig_0`
    # and `o_0` are initialised at exactly zero (§03.6), so their rms is 0 and
    # the floor is what lets a mutation reach them at all; every *nonzero*
    # tensor must sit above it, or the perturbation is absolute throughout and
    # the module does the opposite of what §07.5 says. That is the M17 defect,
    # and it is what makes this an inequality rather than a constant.
    assert SIGMA_FLOOR == 0.1
    rms = sorted(float(p.pow(2).mean().sqrt()) for p in Decoder(SHAPE).parameters())
    zero = [r for r in rms if r == 0.0]
    nonzero = [r for r in rms if r > 0.0]
    assert len(zero) == 5, zero            # c1, c2, w_sig, sig_0, o_0
    assert SIGMA_FLOOR < min(nonzero), (SIGMA_FLOOR, min(nonzero))
    assert min(nonzero) / SIGMA_FLOOR > 1.5, "the floor is not well below them"


def test_the_configurations_are_the_spec_configurations():
    """SPEC: 04 §4.4's table, field by field."""
    want = {
        "m-pico": dict(d_model=128, L=4, n_h=4, n_kv=2, d_head=32, d_ff=512,
                       vocab=4096, T=128),
        "m-nano": dict(d_model=192, L=6, n_h=6, n_kv=2, d_head=32, d_ff=768,
                       vocab=4096, T=128),
        "m1": dict(d_model=512, L=12, n_h=8, n_kv=2, d_head=64, d_ff=2048,
                   vocab=8192, T=512),
    }
    for name, fields in want.items():
        cfg = CONFIGS[name]
        for k, v in fields.items():
            assert getattr(cfg, k) == v, (name, k, getattr(cfg, k), v)


# ------------------------------------------------------------------ §07.5

def test_the_generation_zero_tensor_scales_are_what_the_spec_measured():
    """SPEC: 07 §7.5. The spread that decides whether scale-relative sigma pays.

    §07.5 has now stated three different values for this. The first draft said
    the tensors "differ in scale by more than an order of magnitude"; the second
    said "0.17 to 0.20, a factor of 1.55" and concluded the mechanism buys
    nothing at generation 0. The second was written before §03.6's sweep moved
    `log_gain` from 0 to `log 4`, and pinning it here is what stops a fourth
    version: an untested sentence about a measurement drifts every time the
    thing it measures changes.
    """
    from gnomon.genome.arch import ArchHead

    want = {
        "W_row": 0.1941, "W_col": 0.1910, "W_tok": 0.1749, "W_ctx": 0.2360,
        "W_2": 0.1768, "w_out": 0.1596, "log_gain": math.log(4.0),
    }
    got = {n: float(p.pow(2).mean().sqrt())
           for n, p in Decoder(SHAPE, seed=0).named_parameters()}
    for n, v in want.items():
        assert got[n] == pytest.approx(v, abs=5e-4), (n, got[n], v)
    for n in ("c1", "c2", "o_0", "sig_0", "w_sig"):
        assert got[n] == 0.0, n

    # The spread, over the seeds a run actually draws.
    nz = [float(p.pow(2).mean().sqrt())
          for seed in range(5)
          for _, p in Decoder(SHAPE, seed=seed).named_parameters()
          if float(p.pow(2).mean().sqrt()) > 0]
    assert min(nz) == pytest.approx(0.137, abs=0.005)
    assert max(nz) == pytest.approx(math.log(4.0), abs=1e-6)
    assert max(nz) / min(nz) == pytest.approx(10.1, abs=0.3), (
        "§07.5's claimed factor of 1.55 is what this test exists to prevent"
    )

    anz = [float(p.pow(2).mean().sqrt())
           for _, p in ArchHead(SHAPE, seed=0).named_parameters()
           if float(p.pow(2).mean().sqrt()) > 0]
    assert min(anz) == pytest.approx(0.2036, abs=5e-4)
    assert max(anz) == pytest.approx(0.5178, abs=5e-4)


def test_log_gain_sits_inside_its_own_admissible_range():
    """SPEC: 03 §3.6. `s = exp(log_gain)` is evolved, and `[2, 8]` is its range.

    A mutation moves `log_gain` by `sigma_rel * rms = 0.15 * 1.386 = 0.21`
    under §07.5's scale-relative rule -- about a seventh of `[log 2, log 8]`.
    Recorded because it is the one genome scalar whose admissible range is
    stated, so a step size relative to it is checkable rather than a matter of
    taste.
    """
    from gnomon.genome.arch import SIGMA_FLOOR

    lo, hi = math.log(2.0), math.log(8.0)
    lg = float(Decoder(SHAPE, seed=0).log_gain)
    assert lo < lg < hi
    step = 0.15 * max(abs(lg), SIGMA_FLOOR)
    assert step == pytest.approx(0.2079, abs=1e-3)
    assert step / (hi - lo) == pytest.approx(1 / 6.67, abs=0.02)


# ------------------------------------------------------------------ §10.6

def test_every_configuration_declares_where_its_lr_came_from():
    """SPEC: 10 §10.6. A rate is only meaningful with its horizon and batch.

    §10.6 records, as a defect in every comparison the project had made, that
    `lr` was chosen from a sweep at 2,000 steps and then used at 300 and 600 --
    and then the defect was written down and not fixed: `M_PICO.lr` stayed at
    `3e-3` and the completed `EVO` run used it. A configured value that has not
    been measured must be **distinguishable** from one that has, which is the
    only thing that would have made that visible.
    """
    for cfg in CONFIGS.values():
        assert cfg.lr_provenance, cfg.name
        measured = cfg.lr_provenance.startswith("measured")
        unmeasured = cfg.lr_provenance.startswith("NOT MEASURED")
        assert measured or unmeasured, (
            f"{cfg.name}: provenance must begin 'measured' or 'NOT MEASURED', "
            f"so a reader can tell at a glance: {cfg.lr_provenance[:60]!r}"
        )
        if measured:
            # A measured claim names the artifact it was measured from, so
            # §09.6 rule 5 -- re-measure from disk, never quote -- is executable.
            assert "runs/gnomon/lrsweep-" in cfg.lr_provenance, cfg.name
            assert f"N_life={cfg.N_life}" in cfg.lr_provenance, (
                f"{cfg.name}: measured at a horizon that is not its own"
            )
            assert f"batch={cfg.batch}" in cfg.lr_provenance, cfg.name
        else:
            # An unmeasured one says what to run.
            assert "gnomon_lrsweep" in cfg.lr_provenance or "sweep" in cfg.lr_provenance


def test_m_pico_carries_the_rate_its_own_sweep_measured():
    """SPEC: 10 §10.6, and §09.6 rule 5 -- read back from the artifact."""
    import json
    import pathlib

    assert M_PICO.lr == 0.06
    assert M_PICO.lr_provenance.startswith("measured")

    art = pathlib.Path("runs/gnomon/lrsweep-m-pico-n600-b32.json")
    if not art.exists():
        pytest.skip(f"{art} absent; rerun scripts/gnomon_lrsweep.py "
                    f"--config m-pico --steps 600 --batch 32")
    d = json.loads(art.read_text())
    assert (d["N_life"], d["batch"]) == (M_PICO.N_life, M_PICO.batch)
    assert not d["edge_of_grid"], "the sweep found a bound, not an optimum"
    # The configured value is the smallest of the indistinguishable set.
    tied = [r["lr"] for r in d["rows"] if r["ci"][0] <= 0.0 <= r["ci"][1]]
    assert M_PICO.lr == min(tied + [d["best_lr"]]), (M_PICO.lr, tied)


def test_the_unmeasured_configurations_are_marked_and_not_silently_used():
    """SPEC: 10 §10.6. M-nano and M1 have not been swept; they say so."""
    assert M_NANO.lr_provenance.startswith("NOT MEASURED")
    assert M1.lr_provenance.startswith("NOT MEASURED")
    # And the reason is stated rather than left to be guessed: for M1 a 9-point
    # sweep is 54 full lifetimes against §09.1's 8-hour budget for one.
    assert "GPU-days" in M1.lr_provenance or "many" in M1.lr_provenance
    assert "FRONTIER" in M_NANO.lr_provenance
