"""Attention, routing, the Site, Theorem 1, and the optimizer.

SPEC: 05 §5.3
SPEC: 05 §5.4
SPEC: 05 §5.5
SPEC: 05 §5.7
SPEC: 01 §1.4

Gates `G-ATTN`, `G-ROUTE`, `G-THM1`, `G-MUON`, and the A12 control for §01.4.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from soma.config import M1, M_NANO
from soma.germline import Genome
from soma.germline.field import Field, sample_sphere
from soma.model.attention import Attention, causal_mask, document_mask
from soma.model.model import SomaModel
from soma.model.routing import Router
from soma.model.site import PartsLibrary, Site
from soma.optim.muon import Muon, newton_schulz5, wsd_lr

CFG = M_NANO


@pytest.fixture()
def site_fixture():
    g = torch.Generator().manual_seed(3)
    field = Field.initialise(CFG.germline, generator=g)
    parts = PartsLibrary(CFG)
    site = Site(CFG, parts)
    p_u = sample_sphere(CFG.U_max, CFG.m_dim, g)
    g_u = torch.stack(
        [torch.randperm(CFG.N_G, generator=g)[: CFG.k_g] for _ in range(CFG.U_max)]
    )
    active = torch.zeros(CFG.U_max, dtype=torch.bool)
    active[torch.randperm(CFG.U_max, generator=g)[: CFG.U_0]] = True
    return site, field, p_u, g_u, active.float(), active


# ---------------------------------------------------------------------------
# G-ATTN
# ---------------------------------------------------------------------------


def test_attention_is_causal() -> None:
    torch.manual_seed(0)
    at = Attention(CFG)
    x = torch.randn(2, 24, CFG.d_model)
    w = at.attention_weights(x)
    assert float(w.triu(1).abs().max()) < 1e-7


def test_sink_lets_a_head_decline_to_attend() -> None:
    """(5.17): the weights sum to **less than 1**.

    Softmax otherwise forces a distribution, so a head with nothing to retrieve
    improvises a sink on the first token and distorts the residual stream.
    """
    torch.manual_seed(0)
    at = Attention(CFG)
    x = torch.randn(2, 24, CFG.d_model)
    with torch.no_grad():
        at.z_sink.fill_(0.0)
    assert float(at.attention_weights(x).sum(-1).max()) <= 1.0 + 1e-6

    with torch.no_grad():
        at.z_sink.fill_(-40.0)
    assert float(at.attention_weights(x).sum(-1).min()) == pytest.approx(1.0, abs=1e-5)

    with torch.no_grad():
        at.z_sink.fill_(40.0)
    assert float(at.attention_weights(x).sum(-1).max()) < 1e-8


def test_gqa_head_grouping() -> None:
    at = Attention(M1)
    assert at.repeat == M1.n_h // M1.n_kv
    x = torch.randn(2, 16, M1.d_model)
    assert at(x).shape == (2, 16, M1.d_model)


def test_qk_norm_is_gain_free_and_bounds_the_logits() -> None:
    """§05.3: with a learned gain the bound `|s| <= sqrt(d_h)` does not hold."""
    from soma.model.rmsnorm import rms_norm_headwise

    torch.manual_seed(1)
    at = Attention(CFG)
    assert not any("gain" in n for n, _ in at.named_parameters())
    q = torch.randn(2, CFG.n_h, 16, CFG.d_h) * 100.0
    qn = rms_norm_headwise(q)
    assert float(qn.square().mean(-1).max()) == pytest.approx(1.0, abs=1e-4)
    # |q.k| <= ||q|| ||k|| = d_h, so s = q.k/sqrt(d_h) <= sqrt(d_h).
    kn = rms_norm_headwise(torch.randn(2, CFG.n_h, 16, CFG.d_h))
    s = (qn @ kn.transpose(-1, -2)) / (CFG.d_h ** 0.5)
    assert float(s.abs().max()) <= CFG.d_h ** 0.5 + 1e-3


def test_document_mask_blocks_cross_document_attention() -> None:
    """§06.1.1: a token's conditioning depends only on its own document, which
    is what makes the cached `L_ref` valid."""
    torch.manual_seed(2)
    at = Attention(CFG)
    x = torch.randn(2, 32, CFG.d_model)
    doc = torch.zeros(2, 32, dtype=torch.long)
    doc[:, 16:] = 1
    w = at.attention_weights(x, doc_id=doc)
    assert float(w[:, :, 16:, :16].abs().max()) == 0.0
    assert torch.equal(document_mask(doc)[0, 20, :16], torch.zeros(16, dtype=torch.bool))


def test_causal_mask_shape() -> None:
    m = causal_mask(5)
    assert m.shape == (5, 5)
    assert bool(m[4, 0]) and not bool(m[0, 4])


# ---------------------------------------------------------------------------
# G-ROUTE
# ---------------------------------------------------------------------------


def test_routing_selects_only_active_slots(site_fixture) -> None:
    site, field, p_u, g_u, m_u, active = site_fixture
    torch.manual_seed(4)
    x = torch.randn(3, 20, CFG.d_model)
    res = site.router(x, active)
    sel = res.slots[res.slots >= 0]
    assert bool(active[sel].all())


def test_routing_weights_sum_to_one_over_the_routed_set(site_fixture) -> None:
    site, field, p_u, g_u, m_u, active = site_fixture
    torch.manual_seed(5)
    x = torch.randn(3, 20, CFG.d_model)
    res = site.router(x, active)
    s = res.weights.sum(-1)
    assert float((s - 1.0).abs().max()) < 1e-5


def test_empty_candidate_set_gives_the_shared_bypass_alone(site_fixture) -> None:
    """§05.4: `|R_t| = min(k_route, |cand|)`, and `cand` may be empty."""
    site, field, p_u, g_u, m_u, active = site_fixture
    torch.manual_seed(6)
    x = torch.randn(2, 8, CFG.d_model)
    none_active = torch.zeros(CFG.U_max, dtype=torch.bool)
    res = site.router(x, none_active)
    assert bool((res.slots == -1).all())
    assert float(res.weights.abs().max()) == 0.0

    out = site(
        x, field=field, p_u=p_u, g_u=g_u, m_u=torch.zeros(CFG.U_max),
        active=none_active,
    )
    assert float((out.y - site.W_shared(x)).abs().max()) < 1e-6


def test_widening_reaches_k_route_when_possible(site_fixture) -> None:
    """§05.4's widening loop, bounded by `side`."""
    site, field, p_u, g_u, m_u, active = site_fixture
    torch.manual_seed(7)
    x = torch.randn(4, 32, CFG.d_model)
    res = site.router(x, active)
    assert int(res.n_candidates.min()) >= 1
    assert float((res.slots >= 0).float().sum(-1).mean()) > CFG.k_route * 0.9


def test_dying_and_masked_slots_are_excluded(site_fixture) -> None:
    """One predicate governs the row/column mask, `cand` and the softmax."""
    site, field, p_u, g_u, m_u, active = site_fixture
    torch.manual_seed(8)
    x = torch.randn(2, 16, CFG.d_model)
    half = active.clone()
    idx = torch.nonzero(active).squeeze(-1)[: len(torch.nonzero(active)) // 2]
    half[idx] = False
    res = site.router(x, half)
    sel = res.slots[res.slots >= 0]
    assert not bool(half[idx].any())
    assert bool(half[sel].all())


# ---------------------------------------------------------------------------
# G-THM1 and the A12 control
# ---------------------------------------------------------------------------


def _literal_widening(s1, s2, grid, side, k_route):
    """§05.4's algorithm block, transcribed line for line, one token at a time.

    ```
      I1, I2 = topk_{k_route}(s1), topk_{k_route}(s2)
      cand   = { (i1,i2) : i1 in I1, i2 in I2, active[i1,i2] }
      while |cand| < k_route and I1,I2 can widen:
          widen I1 and I2 by one rank each; recompute cand
      R_t = topk_{min(k_route,|cand|)} over cand of s1[i1] + s2[i2]     (5.18)
    ```

    Deliberately slow and deliberately literal: it is the oracle, so it must be
    readable against the spec text without interpretation.
    """
    B, T, _ = s1.shape
    out = torch.full((B, T, k_route), -1, dtype=torch.long)
    o1 = torch.argsort(s1, dim=-1, descending=True)
    o2 = torch.argsort(s2, dim=-1, descending=True)
    for b in range(B):
        for t in range(T):
            w = k_route
            while True:
                I1, I2 = o1[b, t, :w], o2[b, t, :w]
                cand = [(int(i), int(j)) for i in I1 for j in I2
                        if bool(grid[i, j])]
                if len(cand) >= k_route or w >= side:
                    break
                w += 1
            cand.sort(key=lambda p: -(float(s1[b, t, p[0]]) + float(s2[b, t, p[1]])))
            for m, (i, j) in enumerate(cand[:k_route]):
                out[b, t, m] = i * side + j
    return out


@pytest.mark.parametrize("n_active", [240, 128, 96, 64, 32, 8])
def test_routing_matches_the_literal_widening_loop(n_active: int) -> None:
    """§05.4 / audit A-74: the widening rank is a property of the **token**.

    v3.0 chose one `w` for the whole batch from an expected-occupancy bound and
    claimed it reached the same fixed point. Measured at `n_active = 64` — the
    population every M-nano lifetime is seeded with — 85 of 256 tokens were
    routed to a different unit set, and 25 were left with fewer than `k_route`
    units where the loop would have widened and found a full set.

    The occupancy sweep is the point: at the cap the shortcut and the loop
    agree, so a test at one occupancy would have passed on the broken code.
    """
    from soma.model.routing import Router

    cfg = M_NANO
    side, k = cfg.side, cfg.k_route
    torch.manual_seed(11)
    router = Router(cfg)

    active = torch.zeros(side * side, dtype=torch.bool)
    active[torch.randperm(side * side)[:n_active]] = True
    grid = active.view(side, side)

    x = torch.randn(1, 96, cfg.d_model)
    s1, s2 = router.scores(x)
    neg = torch.finfo(s1.dtype).min
    s1 = s1.masked_fill(~grid.any(1), neg)
    s2 = s2.masked_fill(~grid.any(0), neg)

    expected = _literal_widening(s1, s2, grid, side, k)
    got = router(x, active, n_active=n_active).slots

    for t in range(x.shape[1]):
        e = sorted(v for v in expected[0, t].tolist() if v >= 0)
        g = sorted(v for v in got[0, t].tolist() if v >= 0)
        assert e == g, f"token {t} at n_active={n_active}: spec {e} vs code {g}"


def test_site_cost_is_independent_of_population_size(site_fixture) -> None:
    """Theorem 1: `O(N_G d_in + k_route k_g d_out)`, **no dependence on |U|**.

    Measured by timing at three population sizes, not asserted. The tolerance is
    deliberately loose (2x) because the quantity being excluded is *linear* in
    `|U|`: a naive implementation materialising per-unit weights would be ~8x
    slower at 8x the population, far outside it.
    """
    site, field, p_u, g_u, _, _ = site_fixture
    torch.manual_seed(9)
    x = torch.randn(8, 128, CFG.d_model)
    times = []
    for n_live in (32, 128, 240):
        a = torch.zeros(CFG.U_max, dtype=torch.bool)
        a[torch.randperm(CFG.U_max)[:n_live]] = True
        m = a.float()
        for _ in range(3):                      # warm up
            site(x, field=field, p_u=p_u, g_u=g_u, m_u=m, active=a)
        t0 = time.perf_counter()
        for _ in range(10):
            site(x, field=field, p_u=p_u, g_u=g_u, m_u=m, active=a)
        times.append((time.perf_counter() - t0) / 10)
    assert max(times) / min(times) < 2.0, (
        f"Site cost varied {max(times) / min(times):.2f}x across |U_live| in "
        f"32..240; Theorem 1 requires no dependence on |U|"
    )


def test_post_gate_activation_makes_units_distinct(site_fixture) -> None:
    """§01.4 / audit A-8, as a measurement.

    With the activation **before** the gate, `B` and the gate are both linear so
    the per-unit sum equals the aggregate form **exactly** — every unit is a
    linear reweighting of one shared nonlinear feature. With it after, they
    differ, which is what (1.5) claims.
    """
    site, field, p_u, g_u, m_u, active = site_fixture
    torch.manual_seed(10)
    x = torch.randn(2, 24, CFG.d_model)

    def aggregate(pre_gate: bool):
        out = site(
            x, field=field, p_u=p_u, g_u=g_u, m_u=m_u, active=active,
            pre_gate=pre_gate,
        )
        z = x @ site.parts.A.transpose(0, 1)
        psi = site.psi(x)
        safe = out.slots.clamp_min(0)
        genes = g_u[safe]
        phi = field.phi_at(p_u, g_u)[safe]
        psi_sel = torch.gather(
            psi.unsqueeze(2).expand(*psi.shape[:2], out.slots.shape[-1], CFG.N_G),
            3, genes,
        )
        gate = phi * psi_sel
        G = torch.zeros_like(z)
        G.scatter_add_(
            2, genes.reshape(*genes.shape[:2], -1),
            (out.r_ut.unsqueeze(-1) * gate).reshape(*genes.shape[:2], -1),
        )
        agg = (G * torch.relu(z).square()) @ site.parts.B.transpose(0, 1)
        agg = agg + site.W_shared(x)
        return out.y, agg

    y_pre, agg_pre = aggregate(pre_gate=True)
    y_post, agg_post = aggregate(pre_gate=False)

    assert float((y_pre - agg_pre).abs().max()) < 1e-5, (
        "the pre-gate placement must reproduce the aggregate form exactly; if "
        "it does not, the v2 defect is not reproducible and A12 tests nothing"
    )
    assert float((y_post - agg_post).abs().max()) > 1e-3, (
        "the post-gate placement must NOT reduce to the aggregate form; (1.5) "
        "is the claim §01.4 makes and A12 exists to test empirically"
    )


def test_gate_is_k_g_sparse(site_fixture) -> None:
    """The support is `k_g`, which is what Theorem 1's cost rests on."""
    site, field, p_u, g_u, m_u, active = site_fixture
    phi = field.phi_at(p_u, g_u)
    assert phi.shape == (CFG.U_max, CFG.k_g)
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0


# ---------------------------------------------------------------------------
# G-MUON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64), (512, 128), (128, 512), (2048, 192)])
def test_newton_schulz_contracts_the_spectrum(shape) -> None:
    """§05.7: the quintic drives singular values **toward**, not onto, 1."""
    torch.manual_seed(11)
    G = torch.randn(*shape)
    s0 = torch.linalg.svdvals(G / G.norm())
    O = newton_schulz5(G)
    s1 = torch.linalg.svdvals(O)
    assert float(s1.mean()) > 0.7
    assert float(s1.max()) < 1.5
    spread0 = float(s0.max() - s0.min()) / float(s0.mean())
    spread1 = float(s1.max() - s1.min()) / float(s1.mean())
    assert spread1 < spread0


def test_newton_schulz_is_transpose_equivariant() -> None:
    torch.manual_seed(12)
    G = torch.randn(300, 100)
    a = newton_schulz5(G)
    b = newton_schulz5(G.transpose(0, 1).contiguous()).transpose(0, 1)
    assert float((a - b).abs().max()) < 1e-5


def test_muon_refuses_non_matrix_parameters() -> None:
    """§05.7's groups are enumerated **by name**, and the Muon group contains
    only 2-D matrices. Spec v1's "all 2-D soma matrices" rule wrongly captured
    both `p_u` and the embeddings."""
    with pytest.raises(ValueError, match="only 2-D matrices"):
        Muon([torch.randn(64, requires_grad=True)])


def test_muon_step_moves_the_parameter() -> None:
    torch.manual_seed(13)
    p = torch.randn(64, 32, requires_grad=True)
    before = p.detach().clone()
    opt = Muon([p], lr=1e-2, weight_decay=0.0)
    p.square().sum().backward()
    opt.step()
    assert float((p.detach() - before).abs().max()) > 0.0


def test_zeroing_a_freed_slot_clears_optimizer_state() -> None:
    """§05.7: a death zeroes the freed slot's parameter **and** optimizer state;
    a birth leaves the child's state at zero, because inheriting momentum would
    inject a non-zero update into a unit whose mask is 0."""
    from soma.optim.muon import zero_slot_state

    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    assert float(opt.state[p]["exp_avg"].abs().sum()) > 0

    slots = torch.tensor([0, 3, 7])
    zero_slot_state(opt, p, slots)
    assert float(opt.state[p]["exp_avg"][slots].abs().sum()) == 0.0
    assert float(opt.state[p]["exp_avg_sq"][slots].abs().sum()) == 0.0
    assert float(opt.state[p]["exp_avg"][1].abs().sum()) > 0.0


def test_optimizer_groups_are_exhaustive_and_disjoint() -> None:
    """A parameter matching neither group is a specification defect."""
    model = SomaModel(CFG)
    muon, adamw = model.optimizer_groups()
    ids_m = {id(p) for p in muon}
    ids_a = {id(p) for p in adamw}
    assert not (ids_m & ids_a)
    assert len(ids_m | ids_a) == len(list(model.parameters()))


# ---------------------------------------------------------------------------
# Whole-model shape and finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [M_NANO, M1], ids=lambda c: c.name)
def test_model_forward_shapes(cfg) -> None:
    g = torch.Generator().manual_seed(14)
    genome = Genome.initialise(cfg, generator=g)
    model = SomaModel(cfg)
    model.seed_populations(genome, np.random.default_rng(0), g)
    tokens = torch.randint(0, cfg.V, (2, 16), generator=g)
    out = model(tokens, genome=genome)
    assert out.logits.shape == (2, 16, cfg.V)
    assert len(out.site_outputs) == cfg.L
    assert torch.isfinite(out.logits).all()
    assert out.metabolic is not None
    assert torch.isfinite(out.metabolic.total)


def test_dense_baseline_has_no_populations() -> None:
    """§08.4 B1: the §05 stack with the Site replaced by a dense `ReLU^2` FFN."""
    model = SomaModel(CFG, dense_sites=True)
    assert len(model.populations) == 0
    tokens = torch.randint(0, CFG.V, (2, 16))
    out = model(tokens)
    assert out.logits.shape == (2, 16, CFG.V)
    assert out.metabolic is None


def test_block_layout_matches_5_1() -> None:
    """§05.1: M1 3 attention / 9 mixer; M-nano 2 attention / 4 mixer."""
    assert M1.attention_layers == (3, 7, 11)
    assert M_NANO.attention_layers == (2, 5)
    assert len(M1.mixer_layers) == 9
    assert len(M_NANO.mixer_layers) == 4
