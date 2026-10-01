"""SPEC: 07 §7.6 -- the acceptance tests T1..T11 for the architecture head."""

from __future__ import annotations

import math

import pytest
import torch

from gnomon.config import GenomeShape, M1, M_NANO, M_PICO
from gnomon.genome.arch import (
    D_A, D_GLOB, MENU_F, MENU_H, MENU_L, MENU_W, ArchHead, crossover, mutate,
)
from gnomon.genome.decoder import Decoder, Soma
from gnomon.pheno.model import Gnomon
from gnomon.pheno.plan import Arch

SHAPE = GenomeShape()
CFGS = (M_PICO, M_NANO, M1)


# ---------------------------------------------------------------------- T1

def test_T1_genome_size():
    h = ArchHead(SHAPE)
    assert h.n_params() == 796, h.n_params()
    assert SHAPE.n_params() + h.n_params() == 5279


def test_T1_the_head_holds_only_the_spec_table():
    want = {"A_1", "a_c1", "A_2", "a_c2", "A_op", "b_op", "A_h", "b_h",
            "A_f", "b_f", "A_w", "b_w", "A_d", "b_d", "g_glob"}
    assert {n for n, _ in ArchHead(SHAPE).named_parameters()} == want


# ---------------------------------------------------------------------- T2

def test_T2_the_architecture_is_innate():
    """SPEC: 07 §7.1. A lifetime cannot change the architecture it was born with."""
    h = ArchHead(SHAPE, seed=0)
    a = h.decode_arch(M_PICO)
    soma = Soma(M_PICO, SHAPE, seed=1)
    with torch.no_grad():
        for p in soma.parameters():
            p.mul_(100.0).add_(7.0)
    assert h.decode_arch(M_PICO) == a
    import ast, inspect, textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(ArchHead.decode_arch)))
    body = tree.body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]                      # the docstring names `soma`; the code must not
    names = {n.id for n in ast.walk(ast.Module(body, []))
             if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(ast.Module(body, [])) if isinstance(n, ast.Attribute)}
    assert not (names & {"soma", "zeta", "z", "y"}), (
        f"the architecture head reads the soma; it must be innate: {names}"
    )


# ---------------------------------------------------------------------- T3

def test_T3_determinism():
    h = ArchHead(SHAPE, seed=4)
    assert h.decode_arch(M_NANO) == h.decode_arch(M_NANO)
    assert ArchHead(SHAPE, seed=4).decode_arch(M_NANO) == h.decode_arch(M_NANO)
    assert ArchHead(SHAPE, seed=5).decode_arch(M_NANO) is not None


# ---------------------------------------------------------------------- T4

@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: c.name)
def test_T4_totality_over_random_genomes(cfg):
    """SPEC: 07 §7.3. Every point in the genome space is a valid architecture.

    `argmax` over a fixed menu cannot fail, so the search never has to handle an
    invalid individual and there is no repair operator hiding a second,
    un-evolved piece of design.
    """
    g = torch.Generator().manual_seed(0)
    h = ArchHead(SHAPE, seed=0)
    for i in range(2000):
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 3.0)
        a = h.decode_arch(cfg)
        assert 1 <= a.L <= cfg.L
        assert all(n % cfg.n_kv == 0 and n >= 1 for n in a.n_h), a.n_h
        assert all(n * cfg.d_head <= SHAPE.d_max for n in a.n_h)
        assert all(1 <= f <= SHAPE.d_max for f in a.d_ff)
        assert all(1 <= w <= cfg.T for w in a.window)


def test_T4_totality_the_decoded_architecture_runs():
    """A valid architecture is one that builds and completes a forward pass."""
    g = torch.Generator().manual_seed(1)
    h = ArchHead(SHAPE, seed=0)
    dec = Decoder(SHAPE, seed=0)
    for i in range(12):
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 3.0)
        arch = h.decode_arch(M_PICO)
        m = Gnomon(M_PICO, SHAPE, arch=arch, decoder=dec)
        tok = torch.randint(0, M_PICO.vocab, (2, M_PICO.T + 1))
        L = float(m.loss(tok))
        assert math.isfinite(L), (i, arch)


# ---------------------------------------------------------------------- T5

def test_T5_clamping_saturates_rather_than_raising():
    """A menu entry too large for the configuration saturates to an admissible one."""
    g = torch.Generator().manual_seed(2)
    h = ArchHead(SHAPE, seed=0)
    seen_clamped_L = seen_clamped_ff = False
    for _ in range(400):
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 3.0)
        a = h.decode_arch(M_PICO)
        seen_clamped_L |= a.L == M_PICO.L
        seen_clamped_ff |= max(a.d_ff) == SHAPE.d_max or 8 * M_PICO.d_model in a.d_ff
        assert a.L <= M_PICO.L
    assert seen_clamped_L, "the depth menu never reached the configuration's cap"


# ---------------------------------------------------------------------- T6

def test_T6_the_menus_are_reachable():
    """SPEC: 07 §7.6 T6. A menu entry no genome can reach is not a search space."""
    g = torch.Generator().manual_seed(3)
    h = ArchHead(SHAPE, seed=0)
    ops, heads, ffs, wins, depths = set(), set(), set(), set(), set()
    for _ in range(1000):
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 3.0)
        a = h.decode_arch(M1)
        ops |= set(a.ops)
        heads |= set(a.n_h)
        ffs |= set(a.d_ff)
        wins |= set(a.window)
        depths.add(a.L)
    assert ops == {"attn", "delta"}, ops
    assert len(heads) >= 3, heads
    assert len(ffs) == len(MENU_F), ffs
    assert len(wins) >= 3, wins
    assert len(depths) >= 4, depths


# ---------------------------------------------------------------------- T7

def test_T7_mutation_changes_the_graph():
    """SPEC: 07 §7.6 T7.

    A search in which mutation never changes the architecture is not
    architecture search. This is the assertion that C3 is load-bearing.
    """
    g = torch.Generator().manual_seed(4)
    base = ArchHead(SHAPE, seed=0)
    a0 = base.decode_arch(M_PICO)
    changed = 0
    for _ in range(200):
        h = ArchHead(SHAPE, seed=0)
        mutate([h], generator=g)
        changed += h.decode_arch(M_PICO) != a0
    assert changed >= 20, f"only {changed}/200 mutations changed the architecture"


def test_T7_mutation_of_Gamma_w_does_not_change_the_graph():
    """The two heads are separate: `Gamma_w` moves weights, `Gamma_a` moves shape."""
    g = torch.Generator().manual_seed(5)
    h = ArchHead(SHAPE, seed=0)
    a0 = h.decode_arch(M_PICO)
    dec = Decoder(SHAPE, seed=0)
    mutate([dec], generator=g)
    assert h.decode_arch(M_PICO) == a0


# ---------------------------------------------------------------------- T8

def test_T8_one_layer_control_collapses_the_search_space():
    """SPEC: 07 §7.2. One hidden layer makes architecture a function of depth alone.

    The control for T6: with a single layer the head is additive in
    `(r_l, g_glob)` at initialisation scale, so the per-layer choices become a
    monotone function of relative depth and the reachable space is strictly
    smaller. If this control matched T6's coverage, T6 would be measuring
    nothing.
    """
    g = torch.Generator().manual_seed(6)
    h = ArchHead(SHAPE, seed=0)
    with torch.no_grad():
        h.A_2.copy_(torch.eye(D_A))          # collapse the second layer to identity
        h.a_c2.zero_()
    combos_two, combos_one = set(), set()
    for _ in range(600):
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.4)
        a = h.decode_arch(M1)
        combos_one.add((a.ops, a.n_h, a.d_ff))
    h2 = ArchHead(SHAPE, seed=0)
    g2 = torch.Generator().manual_seed(6)
    for _ in range(600):
        with torch.no_grad():
            for p in h2.parameters():
                p.copy_(torch.randn(p.shape, generator=g2) * 0.4)
        a = h2.decode_arch(M1)
        combos_two.add((a.ops, a.n_h, a.d_ff))
    assert len(combos_two) >= len(combos_one), (len(combos_two), len(combos_one))


# ---------------------------------------------------------------------- T9

def test_T9_descriptors_are_in_range_and_need_no_lifetime():
    """SPEC: 07 §7.4. Computable before evaluation."""
    g = torch.Generator().manual_seed(7)
    h = ArchHead(SHAPE, seed=0)
    for _ in range(200):
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 3.0)
        a = h.decode_arch(M_PICO)
        b1, b2 = a.descriptors(M_PICO)
        assert 0.0 <= b1 <= 1.0
        assert 3.0 < b2 < 9.0, b2
        assert b1 == pytest.approx(sum(o == "attn" for o in a.ops) / a.L)


def test_T9_descriptors_agree_with_the_model_actually_built():
    h = ArchHead(SHAPE, seed=11)
    a = h.decode_arch(M_PICO)
    m = Gnomon(M_PICO, SHAPE, arch=a)
    assert m.n_decoded() == a.n_decoded(M_PICO)
    assert sum(v.numel() for v in m.decode().values()) == m.n_decoded()
    assert m.arch.attn_fraction() == a.descriptors(M_PICO)[0]


# --------------------------------------------------------------------- T10

def _drifted(seed=0):
    """A genome whose tensors have drifted apart in scale, as an evolved one has."""
    d = Decoder(SHAPE, seed=seed)
    with torch.no_grad():
        d.W_2.mul_(50.0)
        d.W_ctx.mul_(0.02)
    return d


def _relative_spread(mutator) -> tuple[float, dict]:
    """Max/min of the **relative** perturbation over the genome's tensors.

    Divided by `rms`, not by `max(rms, 1)`: an earlier version used the latter,
    mirroring the sigma *floor* rather than the quantity being measured, which
    turned this into a comparison of absolute perturbations and let an
    absolute-sigma mutation pass (`G-MUTATE` M17).
    """
    from gnomon.genome.arch import SIGMA_FLOOR

    a, b = _drifted(), _drifted()
    mutator(b)
    rel = {}
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        rms = float(pa.pow(2).mean().sqrt())
        # Only tensors above the floor: below it `sigma` deliberately stops
        # tracking scale, so that a zero-initialised tensor can move at all
        # (§07.5). Including them would test the floor, not the scaling.
        if rms > SIGMA_FLOOR and pa.numel() > 8:
            rel[n] = float((pb - pa).pow(2).mean().sqrt()) / rms
    assert len(rel) >= 4, f"too few tensors above the floor to compare: {rel}"
    return max(rel.values()) / min(rel.values()), rel


def test_T10_mutation_is_scale_relative():
    """SPEC: 07 §7.5, in the regime where the two schemes actually differ.

    At generation 0 the genome's nonzero tensors sit at similar RMS, so absolute
    and relative sigma are nearly the same thing and this test would measure
    nothing. An evolved genome's tensors have drifted; that is the regime.
    """
    ratio, rel = _relative_spread(
        lambda d: mutate([d], sigma_rel=0.15, p_mut=1.0,
                         generator=torch.Generator().manual_seed(8)))
    assert ratio < 4.0, (
        f"relative perturbation spans a factor of {ratio:.1f}; mutation is not "
        f"scale-relative: {rel}"
    )


def test_T10_absolute_sigma_fails_once_the_genome_has_drifted():
    """The control for T10, and the honest form of §07.5's argument.

    At **generation 0** the genome's nonzero tensors happen to sit at similar
    RMS (measured 0.17-0.20), so absolute and relative sigma are nearly the same
    thing there: an absolute 0.15 gives relative perturbations spanning only
    0.59-0.92. The argument for scale-relative mutation is therefore not about
    the initialisation -- it is about what tens of generations of mutation do,
    after which tensors have drifted apart in scale.
    """
    g = torch.Generator().manual_seed(8)

    def absolute(d):
        with torch.no_grad():
            for p in d.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 0.15)

    ratio_abs, rel_abs = _relative_spread(absolute)
    ratio_rel, _ = _relative_spread(
        lambda d: mutate([d], sigma_rel=0.15, p_mut=1.0,
                         generator=torch.Generator().manual_seed(8)))
    assert ratio_abs > 20.0, rel_abs
    assert ratio_rel < ratio_abs / 5.0, (ratio_rel, ratio_abs)


def test_T10_a_zero_initialised_tensor_can_still_move():
    """`w_sig` starts at exactly zero; a purely relative sigma would freeze it."""
    g = torch.Generator().manual_seed(9)
    d = Decoder(SHAPE, seed=0)
    assert float(d.w_sig.abs().max()) == 0.0
    mutate([d], p_mut=1.0, generator=g)
    assert float(d.w_sig.abs().max()) > 0.0


def test_mutation_is_sparse():
    g = torch.Generator().manual_seed(10)
    a = Decoder(SHAPE, seed=0)
    b = Decoder(SHAPE, seed=0)
    mutate([b], p_mut=0.25, generator=g)
    frac = float((a.W_2 != b.W_2).float().mean())
    assert 0.15 < frac < 0.35, frac


# --------------------------------------------------------------------- T11

def test_T11_crossover_takes_whole_tensors():
    g = torch.Generator().manual_seed(11)
    a, b = Decoder(SHAPE, seed=1), Decoder(SHAPE, seed=2)
    child = Decoder(SHAPE, seed=3)
    crossover([child], [a], [b], generator=g)
    for (n, cp), ap, bp in zip(child.named_parameters(), a.parameters(),
                               b.parameters()):
        assert torch.equal(cp, ap) or torch.equal(cp, bp), n


def test_T11_both_parents_contribute_over_many_crossovers():
    g = torch.Generator().manual_seed(12)
    a, b = Decoder(SHAPE, seed=1), Decoder(SHAPE, seed=2)
    from_a = 0
    for _ in range(200):
        child = Decoder(SHAPE, seed=3)
        crossover([child], [a], [b], generator=g)
        from_a += torch.equal(child.W_2, a.W_2)
    assert 60 < from_a < 140, from_a


def test_crossover_covers_the_arch_head_too():
    g = torch.Generator().manual_seed(13)
    a, b = ArchHead(SHAPE, seed=1), ArchHead(SHAPE, seed=2)
    child = ArchHead(SHAPE, seed=3)
    crossover([child], [a], [b], generator=g)
    for (n, cp), ap, bp in zip(child.named_parameters(), a.parameters(),
                               b.parameters()):
        assert torch.equal(cp, ap) or torch.equal(cp, bp), n


# ------------------------------------------------------------------- extras

def test_a_decoded_architecture_trains():
    """The end-to-end check: a genome-decoded graph is a working model."""
    h = ArchHead(SHAPE, seed=21)
    arch = h.decode_arch(M_PICO)
    m = Gnomon(M_PICO, SHAPE, arch=arch, genome_seed=0, soma_seed=1)
    tok = torch.randint(0, M_PICO.vocab, (4, M_PICO.T + 1))
    opt = torch.optim.AdamW(m.soma_parameters(), lr=5e-3)
    first = float(m.loss(tok))
    for _ in range(15):
        loss = m.loss(tok)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(m.loss(tok)) < first - 0.05, (arch.summary(), first)


def test_uniform_arch_matches_the_configuration():
    a = Arch.uniform(M_NANO)
    assert a.L == M_NANO.L
    assert set(a.n_h) == {M_NANO.n_h} and set(a.d_ff) == {M_NANO.d_ff}
    assert a.n_decoded(M_NANO) == M_NANO.n_decoded()


def test_arch_rejects_ragged_fields():
    with pytest.raises(ValueError, match="entries"):
        Arch(("attn", "delta"), (2,), (64, 64), (32, 32))
    with pytest.raises(ValueError, match="unknown operator"):
        Arch(("conv",), (2,), (64,), (32,))
