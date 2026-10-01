"""SPEC: 03 §3.8 -- the acceptance tests D1..D12 for the decoder."""

from __future__ import annotations

import math

import pytest
import torch

from gnomon.config import (
    GAIN_ROLES, N_ROLES, ROLE_EMB, ROLE_FFN_DOWN, ROLE_FFN_GATE, ROLE_NAMES,
    ROLE_NORM_MIX, ROLE_PLAST_CHAN, ROLE_PLAST_TOK, ROLE_Q, ROLE_SINK,
    VECTOR_ROLES, GenomeShape, M1, M_NANO, M_PICO, PhenoConfig,
)
from gnomon.genome.decoder import (
    AXIS_KIND, Decoder, Soma, decode_reference, s_target, vector_offset,
)
from gnomon.utils.stats import effective_rank

SHAPE = GenomeShape()
# A small phenotype so the O(R*C) Python oracle is affordable.
TINY = PhenoConfig(name="tiny", d_model=16, L=3, n_h=2, n_kv=1, d_head=8,
                   d_ff=32, vocab=24, T=64, N_life=8, chunk_size=32, window=32)


def _pair(shape=SHAPE, cfg=TINY, *, seed=0, dtype=torch.float64):
    dec = Decoder(shape, seed=seed).to(dtype)
    soma = Soma(cfg, shape, seed=seed + 1).to(dtype)
    return dec, soma


# ------------------------------------------------------------------ D1, D2

@pytest.mark.parametrize("role", sorted(range(N_ROLES)), ids=lambda r: ROLE_NAMES[r])
def test_D1_matches_the_entrywise_oracle(role):
    """Every role, matrix and vector alike, against the naive triple loop."""
    dec, soma = _pair()
    layer = -1 if role in (ROLE_EMB, ROLE_PLAST_TOK) else 1
    got = dec.decode(role, layer, soma, TINY)
    want = decode_reference(dec, role, layer, soma, TINY)
    assert got.shape == TINY.role_shape(role), ROLE_NAMES[role]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


def test_D2_factorisation_is_exact():
    """SPEC: 03 §3.1. The broadcast must equal explicit concatenation."""
    dec, soma = _pair()
    cache = dec.codes(TINY)
    R, Cn = TINY.role_shape(ROLE_Q)
    u_ctx = dec._u_ctx(1, ROLE_Q, soma, cache)
    u_row = dec._axis("inner", R, soma, cache, column=False) @ dec.W_row.T
    u_col = dec._axis("chan", Cn, soma, cache, column=True) @ dec.W_col.T
    fast = dec._readout(u_row, u_col, u_ctx)
    slow = torch.stack([
        torch.stack([
            (torch.tanh(dec.W_2 @ torch.tanh(
                torch.exp(dec.log_gain) * (u_row[i] + u_col[j] + u_ctx) + dec.c1)
                + dec.c2) @ dec.w_out) + dec.o_0
            for j in range(Cn)]) for i in range(R)])
    torch.testing.assert_close(fast, slow, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------- D3

def test_D3_gradient_reaches_every_genome_tensor():
    """SPEC: 03 §3.8 D3.

    SOMA shipped a routing path whose gradient could be severed entirely with
    452 tests passing, because nothing asserted reachability. This is that
    assertion, per tensor, by name.
    """
    dec, soma = _pair()
    loss = sum(
        dec.decode(r, -1 if r in (ROLE_EMB, ROLE_PLAST_TOK) else l, soma, TINY).pow(2).sum()
        for r in range(N_ROLES) for l in range(TINY.L)
    )
    loss.backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None, f"{name}: no grad"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad"
        assert p.grad.abs().max() > 0, f"{name}: gradient severed (all zero)"


def test_D3_gradient_reaches_every_soma_tensor():
    dec, soma = _pair()
    loss = sum(
        dec.decode(r, -1 if r in (ROLE_EMB, ROLE_PLAST_TOK) else l, soma, TINY).pow(2).sum()
        for r in range(N_ROLES) for l in range(TINY.L)
    )
    loss.backward()
    for name, p in soma.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, f"Z.{name}: severed"


# ---------------------------------------------------------------------- D4

def test_D4_finite_differences():
    """Central differences against autograd, float64, on every genome tensor."""
    dec, soma = _pair()
    g = torch.Generator().manual_seed(7)

    def f():
        return sum(dec.decode(r, 0, soma, TINY).sin().sum()
                   for r in (ROLE_Q, ROLE_NORM_MIX, ROLE_EMB, ROLE_SINK))

    f().backward()
    eps = 1e-6
    for name, p in dec.named_parameters():
        flat = p.data.view(-1)
        idxs = torch.randint(0, flat.numel(), (min(4, flat.numel()),), generator=g)
        for k in idxs.tolist():
            old = flat[k].item()
            flat[k] = old + eps
            hi = f().item()
            flat[k] = old - eps
            lo = f().item()
            flat[k] = old
            num = (hi - lo) / (2 * eps)
            ana = p.grad.view(-1)[k].item()
            assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana)), (
                f"{name}[{k}]: fd {num:.8f} vs autograd {ana:.8f}"
            )


# ----------------------------------------------------------------- D5, D6

def _decoded_rank(dec, soma, cfg, role=ROLE_Q, layer=0):
    """`erank` of the decoded matrix's *spectrum* (§03.4.1 takes singular values)."""
    W = dec.decode(role, layer, soma, cfg).detach().float()
    return effective_rank(torch.linalg.svdvals(W).tolist())


def _one_layer_rank(cfg, seed, gain=None):
    """SPEC: 03 §3.8 D6 -- the degenerate decoder, as a measurement."""
    dec, soma = _pair(cfg=cfg, seed=seed, dtype=torch.float32)
    if gain is not None:
        with torch.no_grad():
            dec.log_gain.fill_(math.log(gain))

    def one_layer(u_row, u_col, u_ctx):
        h1 = torch.tanh(torch.exp(dec.log_gain)
                        * (u_row.unsqueeze(1) + u_col.unsqueeze(0) + u_ctx) + dec.c1)
        return h1 @ dec.w_out + dec.o_0          # readout straight off h1

    dec._readout = one_layer
    return _decoded_rank(dec, soma, cfg)


def _rank_exponent(ranks) -> float:
    """Fit `erank ~ d^alpha` over the three configurations. SPEC: 03 §3.8 D5b."""
    import numpy as np
    d = np.log([min(c.role_shape(ROLE_Q)) for c in (M_PICO, M_NANO, M1)])
    return float(np.polyfit(d, np.log(ranks), 1)[0])


@pytest.mark.parametrize("cfg", [M_PICO, M_NANO, M1], ids=lambda c: c.name)
def test_D5a_decoded_rank_clears_the_absolute_floor(cfg):
    """SPEC: 03 §3.8 D5a. Measured 0.38 / 0.31 / 0.26 of `min(d)`."""
    floor = 0.20 * min(cfg.role_shape(ROLE_Q))
    for seed in range(3):
        dec, soma = _pair(cfg=cfg, seed=seed, dtype=torch.float32)
        r = _decoded_rank(dec, soma, cfg)
        assert r > floor, f"{cfg.name} seed {seed}: erank {r:.2f} <= {floor:.2f}"


def test_D5b_decoded_rank_scales_with_width():
    """SPEC: 03 §3.8 D5b -- the criterion that actually discriminates.

    A fixed fraction of `min(d)` is the wrong shape of test: the decoder's rank
    grows *sublinearly* in width (a strong low-rank prior is the compression
    doing its job, not a defect). What must not happen is a rank that is
    effectively fixed regardless of scale -- SOMA's one-layer decode gave erank
    13.6 at `d=128` and 16.0 at `d=512`, a factor of 1.2 against a fourfold
    width. The two-layer decoder at the default gain grows as `d^0.73`.
    """
    ranks = [sum(_decoded_rank(*_pair(cfg=c, seed=s, dtype=torch.float32), c)
                 for s in range(3)) / 3 for c in (M_PICO, M_NANO, M1)]
    alpha = _rank_exponent(ranks)
    assert alpha > 0.6, f"erank ~ d^{alpha:.2f}; ranks {[f'{r:.1f}' for r in ranks]}"


def test_D6_one_layer_collapses_at_unit_gain():
    """SPEC: 03 §3.1, §3.8 D6. The regime SOMA was in, reproduced.

    At gain 1 -- the natural default, and SOMA's -- `tanh` is near-linear, the
    single-layer decoder is additive in its arguments, and the decoded matrix
    collapses. Measured here: 10.6% / 8.3% / 3.1% of `min(d)` at pico / nano /
    M1, against SOMA's own measurement of 6.76 of 192 = 3.5%.
    """
    fracs = []
    for cfg in (M_PICO, M_NANO, M1):
        r = sum(_one_layer_rank(cfg, s, gain=1.0) for s in range(3)) / 3
        fracs.append(r / min(cfg.role_shape(ROLE_Q)))
    assert fracs[-1] < 0.05, f"M1 one-layer at gain 1 did not collapse: {fracs}"
    assert fracs[0] < 0.15, f"pico one-layer at gain 1 did not collapse: {fracs}"
    # And the collapse deepens with width, which is what makes it a collapse
    # rather than merely a low rank.
    assert fracs[0] > fracs[1] > fracs[2], fracs


def test_D6_positive_control_fails_the_scaling_criterion():
    """The control for D5b: the degenerate decoder must not pass it."""
    ranks = [sum(_one_layer_rank(c, s) for s in range(3)) / 3
             for c in (M_PICO, M_NANO, M1)]
    alpha = _rank_exponent(ranks)
    assert alpha <= 0.6, (
        f"the one-layer control scaled as d^{alpha:.2f} and would pass D5b; "
        f"D5b is then not measuring what it claims to"
    )


def test_both_the_gain_and_the_second_layer_are_needed():
    """SPEC: 03 §3.1. Neither change alone suffices; the spec records both."""
    cfg = M1
    mn = min(cfg.role_shape(ROLE_Q))
    one_g1 = sum(_one_layer_rank(cfg, s, gain=1.0) for s in range(3)) / 3
    one_g4 = sum(_one_layer_rank(cfg, s) for s in range(3)) / 3
    two_g4 = sum(_decoded_rank(*_pair(cfg=cfg, seed=s, dtype=torch.float32), cfg)
                 for s in range(3)) / 3
    assert one_g1 / mn < 0.05          # SOMA's regime
    assert one_g4 / mn < 0.20          # the gain alone is not enough
    assert two_g4 / mn > 0.20          # both together clear the floor


# ---------------------------------------------------------------------- D7

def test_D7_decoded_scale_is_exactly_sigma_times_s_target():
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float64)
    cache = dec.codes(M_PICO)
    for role in range(N_ROLES):
        if role in VECTOR_ROLES:
            continue
        layer = -1 if role == ROLE_EMB else 2
        W = dec.decode(role, layer, soma, M_PICO)
        sig = dec._sigma(dec._u_ctx(layer, role, soma, cache))
        want = float(sig) * s_target(role, M_PICO.role_shape(role), M_PICO)
        got = float(torch.sqrt(W.pow(2).mean()))
        assert got == pytest.approx(want, rel=1e-6), ROLE_NAMES[role]


def test_default_gain_is_the_measured_one():
    """SPEC: 03 §3.6. The default is a measurement; changing it needs a re-sweep."""
    from gnomon.genome.decoder import DEFAULT_GAIN
    assert DEFAULT_GAIN == 4.0
    assert float(torch.exp(Decoder(SHAPE).log_gain)) == pytest.approx(4.0)


def test_decoder_construction_is_reproducible():
    """`nn.init.orthogonal_` draws from the global RNG; W_2 must not."""
    torch.manual_seed(1)
    a = Decoder(SHAPE, seed=3).W_2.clone()
    torch.manual_seed(999)
    b = Decoder(SHAPE, seed=3).W_2.clone()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, Decoder(SHAPE, seed=4).W_2)
    # and it is genuinely orthogonal
    torch.testing.assert_close(a @ a.T, torch.eye(SHAPE.d_h), atol=1e-5, rtol=0)


def test_D7_the_grouped_decode_has_the_same_scale():
    """SPEC: 03 §3.8 D7, over the path the model actually takes.

    `decode` is the single-role reference; `decode_many` is what runs every
    step. An earlier version of D7 checked only the former, and a planted defect
    that scaled `decode_many`'s output by three survived the whole suite
    (`G-MUTATE` M5). Testing the reference path is not testing the real one.
    """
    from gnomon.pheno.model import Gnomon

    m = Gnomon(M_PICO, SHAPE, genome_seed=0, soma_seed=1).double()
    cache = m.decoder.codes(M_PICO)
    for (role, layer), W in m.decode().items():
        if role in VECTOR_ROLES:
            continue
        sig = m.decoder._sigma(m.decoder._u_ctx(layer, role, m.soma, cache))
        want = float(sig) * s_target(role, m.arch.role_shape(M_PICO, role, layer),
                                     M_PICO, m.arch.L)
        got = float(torch.sqrt(W.pow(2).mean()))
        assert got == pytest.approx(want, rel=1e-6), (ROLE_NAMES[role], layer)


def test_D7_sigma_is_one_at_initialisation():
    """SPEC: 03 §3.6. `w_sig = 0`, `sig_0 = 0` gives exactly the innate scale."""
    dec, soma = _pair(cfg=M_PICO)
    cache = dec.codes(M_PICO)
    for role in range(N_ROLES):
        sig = dec._sigma(dec._u_ctx(0, role, soma, cache))
        assert float(sig) == pytest.approx(1.0, abs=1e-12)


def test_gain_roles_decode_near_identity():
    """SPEC: 03 §3.3. An uninformative genome must not break the forward pass."""
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    for role in sorted(GAIN_ROLES):
        v = dec.decode(role, 0, soma, M_PICO)
        assert 0.2 < float(v.mean()) < 1.8, f"{ROLE_NAMES[role]}: mean {float(v.mean())}"
        assert vector_offset(role) == 1.0


# ---------------------------------------------------------------- D8, D9

def test_D8_one_genome_decodes_every_configuration():
    """SPEC: 03 §3.5. The `H log N` claim, exercised rather than asserted."""
    dec = Decoder(SHAPE, seed=0)
    n = dec.n_params()
    for cfg in (M_PICO, M_NANO, M1):
        soma = Soma(cfg, SHAPE, seed=1)
        W = dec.decode(ROLE_Q, 0, soma, cfg)
        assert W.shape == cfg.role_shape(ROLE_Q)
        assert dec.n_params() == n, "the genome changed shape with the phenotype"
    assert n == SHAPE.n_params() == 4483


def test_D9_nested_widths_share_their_codes():
    """SPEC: 02 §2.4 P1, at the level of the decode."""
    dec = Decoder(SHAPE, seed=0)
    small = PhenoConfig(name="s", d_model=64, L=4, n_h=2, n_kv=1, d_head=32,
                        d_ff=128, vocab=64, T=64, N_life=8, chunk_size=32)
    big = PhenoConfig(name="b", d_model=128, L=4, n_h=2, n_kv=1, d_head=32,
                      d_ff=128, vocab=64, T=64, N_life=8, chunk_size=32)
    torch.testing.assert_close(dec.codes(small)["chan"],
                               dec.codes(big)["chan"][:64])


def test_compression_ratios_are_what_the_spec_claims():
    """SPEC: 03 §3.5's table, recomputed rather than quoted."""
    got = {c.name: (c.n_decoded(), c.n_decoded() / SHAPE.n_params())
           for c in (M_PICO, M_NANO, M1)}
    for name, (n, ratio) in got.items():
        assert n > 0 and ratio > 100, (name, n, ratio)
    assert got["m1"][1] > got["m-nano"][1] > got["m-pico"][1], (
        f"compression must improve with scale; got {got}"
    )


# --------------------------------------------------------------- D10, D11

def test_D10_determinism():
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    a = dec.decode(ROLE_FFN_GATE, 1, soma, M_PICO)
    b = dec.decode(ROLE_FFN_GATE, 1, soma, M_PICO)
    assert torch.equal(a, b)


def test_D11_chunked_equals_unchunked_on_a_non_divisible_shape():
    """SPEC: 03 §3.7. Chunk boundaries must be exercised, not merely present."""
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float64)
    full = dec.decode(ROLE_FFN_GATE, 0, soma, M_PICO, chunk_elems=1 << 30)
    for elems in (1 << 20, 7919 * SHAPE.d_h, 3 * 128 * SHAPE.d_h):
        part = dec.decode(ROLE_FFN_GATE, 0, soma, M_PICO, chunk_elems=elems)
        torch.testing.assert_close(part, full, rtol=0, atol=1e-12)


def test_D11_chunking_actually_splits():
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    R, Cn = M_PICO.role_shape(ROLE_FFN_GATE)
    assert R * Cn * SHAPE.d_h > 3 * Cn * SHAPE.d_h, "the chunk test is vacuous"


# ---------------------------------------------------------------------- D12

def test_D12_decoder_holds_only_the_spec_table():
    """SPEC: 03 §3.5 -- these tensors and nothing else."""
    dec = Decoder(SHAPE)
    want = {"W_row", "W_col", "W_tok", "W_ctx", "c1", "W_2", "c2",
            "w_out", "o_0", "w_sig", "sig_0", "log_gain"}
    assert {n for n, _ in dec.named_parameters()} == want
    assert list(dec.buffers()) == []


def test_inner_axes_do_not_collide():
    """The degeneracy the first version of this module had.

    Tiling the channel table onto a wider FFN axis gave units `i` and
    `i + d_model` identical identities and therefore identical decoded rows.
    """
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    W = dec.decode(ROLE_FFN_GATE, 0, soma, M_PICO)
    d = M_PICO.d_model
    assert W.shape[0] > d, "the test needs an FFN wider than the residual stream"
    assert not torch.allclose(W[:d], W[d:2 * d]), "inner identities collided"
    rows = torch.unique(W.round(decimals=6), dim=0)
    assert rows.shape[0] == W.shape[0], "duplicate decoded rows"


def test_inner_axes_carry_no_acquired_identity():
    """SPEC: 03 AXIS_KIND. Internal units are the genome's business alone."""
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    cache = dec.codes(M_PICO)
    feat = dec._axis("inner", M_PICO.d_ff, soma, cache, column=False)
    assert (feat[:, SHAPE.n_c:] == 0).all()
    assert feat.shape == (M_PICO.d_ff, SHAPE.n_c + SHAPE.d_z)


def test_plasticity_rule_decodes_within_its_ranges():
    """SPEC: 10 §10.2, §10.3. The lifetime's update, decoded from identity."""
    from gnomon.config import N_RULE
    from gnomon.lifetime.plasticity import ADAM_POINT

    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    for role in (ROLE_PLAST_CHAN, ROLE_PLAST_TOK):
        c = dec.decode_rule(-1 if role == ROLE_PLAST_TOK else 0, role, soma,
                            M_PICO)
        assert c.shape == M_PICO.role_shape(role)
        assert c.shape[1] == N_RULE
        lo = torch.tensor([-1.0, 0.0, -0.1, -1.0])
        hi = torch.tensor([1.0, 2.0, 0.1, 1.0])
        assert (c >= lo - 1e-5).all() and (c <= hi + 1e-5).all()
        # An uninformative genome must land on Adam, not on a random rule.
        assert abs(float(c[:, 1].mean()) - ADAM_POINT[1]) < 0.25


def test_d_z_zero_is_a_valid_configuration():
    """SPEC: 06 §6.5. The fully-innate left endpoint of the frontier."""
    shape0 = GenomeShape(d_z=0)
    dec = Decoder(shape0, seed=0)
    soma = Soma(M_PICO, shape0, seed=1)
    assert soma.z.numel() == 0 and soma.zeta.numel() == 0
    W = dec.decode(ROLE_Q, 0, soma, M_PICO)
    assert W.shape == M_PICO.role_shape(ROLE_Q) and torch.isfinite(W).all()
    assert dec.n_params() == shape0.n_params()


# ---------------------------------------------------------------- D13, §3.7

def test_D13_the_decode_is_gradient_checkpointed_and_changes_no_value():
    """SPEC: 03 §3.7. "the decode ... is gradient-checkpointed" -- it was not.

    The decode holds `h1` and `h2`, each `d_out x d_in x d_h`, so the retained
    activation is `2 * d_h = 64` times the weight count. At M1's 49.9M weights
    that is 3.19e9 floats -- 12.8 GB in fp32, measured as a 17.5 GB peak on a
    16 GB card at batch 8, which spilled and ran at 4.4 s/step. With
    checkpointing the same step takes 5.65 GB and 1.0 s.

    Chunking alone does not help: every chunk stays in the autograd graph until
    backward, so the peak is the whole decode however it was sliced. That is
    why §3.7 names both and why this test exercises both.

    Recomputation must be *exact*, not merely close -- the decode has no RNG and
    no data-dependent control flow -- so this asserts bit equality of the
    decoded matrix and of the gradient into every tensor of `Gamma` and `Z`.
    """
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float64)
    dec.prepare(M_PICO, max(M_PICO.d_ff, M_PICO.d_qk), M_PICO.L)

    def run(ckpt, chunk):
        dec.checkpoint_decode = ckpt
        for p in list(dec.parameters()) + list(soma.parameters()):
            p.grad = None
        W = dec.decode(ROLE_FFN_GATE, 0, soma, M_PICO, chunk_elems=chunk)
        W.pow(2).sum().backward()
        return (W.detach().clone(),
                {n: p.grad.detach().clone() for n, p in dec.named_parameters()
                 if p.grad is not None},
                {n: p.grad.detach().clone() for n, p in soma.named_parameters()
                 if p.grad is not None})

    for chunk in (1 << 30, 1 << 14):          # unchunked, and many chunks
        Wa, ga, za = run(False, chunk)
        Wb, gb, zb = run(True, chunk)
        assert torch.equal(Wa, Wb), f"chunk {chunk}: the decoded matrix moved"
        assert set(ga) == set(gb), "checkpointing changed which tensors get grad"
        for k in ga:
            assert torch.equal(ga[k], gb[k]), f"chunk {chunk}: d/d{k} moved"
        for k in za:
            assert torch.equal(za[k], zb[k]), f"chunk {chunk}: dL/d{k} moved"
    dec.checkpoint_decode = True


def test_D13_checkpointing_is_on_by_default_and_off_under_capture():
    """SPEC: 03 §3.7. The default is the memory-safe one.

    Checkpointing recomputes inside backward; a captured CUDA graph cannot
    express that. The two are alternatives, not a stack, so the capture backend
    turns it off explicitly rather than silently producing a graph that is
    wrong.
    """
    import inspect

    from gnomon.lifetime.runner import LifetimeRunner

    assert Decoder(SHAPE).checkpoint_decode is True
    src = inspect.getsource(LifetimeRunner._capture)
    assert "checkpoint_decode = False" in src


def test_D13_checkpointing_actually_frees_the_interior():
    """The point of it: `h1` and `h2` are not retained for backward.

    Measured by counting the elements autograd *saves*, via
    `saved_tensors_hooks`, rather than by watching process memory -- a unit test
    that asserted a byte count would be flaky, and this asserts the mechanism.

    The interior is `2 * d_h` times the decoded matrix, so the saving is large
    and the assertion can be a ratio rather than a nudge.
    """
    dec, soma = _pair(cfg=M_PICO, dtype=torch.float32)
    dec.prepare(M_PICO, max(M_PICO.d_ff, M_PICO.d_qk), M_PICO.L)

    def saved_elems(ckpt):
        dec.checkpoint_decode = ckpt
        total = []

        def pack(t):
            total.append(t.numel())
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            dec.decode(ROLE_FFN_GATE, 0, soma, M_PICO,
                       chunk_elems=1 << 30).pow(2).sum().backward()
        return sum(total)

    plain = saved_elems(False)
    ckpt = saved_elems(True)
    dec.checkpoint_decode = True
    out = math.prod(M_PICO.role_shape(ROLE_FFN_GATE))
    print(f"saved: unchecked {plain:,}  checkpointed {ckpt:,}  "
          f"decoded matrix {out:,}")
    assert ckpt < plain / 4, (
        f"checkpointing saved {ckpt:,} elements against {plain:,} unchecked; "
        f"the interior (2*d_h = {2*SHAPE.d_h}x the matrix) is still retained"
    )
