"""Acceptance tests G1-G12 of docs/anlage/06. SPEC: 06 §6.7"""

from __future__ import annotations

import torch

from anlage.config import (
    ROLE_EMB, ROLE_GATE, ROLE_NORM_MIX, ROLE_O, ROLE_Q, ROLE_UP, StageConfig,
)
from anlage.genome import DecodeContext, Genome
from anlage.grow import Ramp, grow, layer_placement
from anlage.latents import LatentBank
from anlage.model import Phenotype
from tests_anlage.conftest import make_tables

VOCAB = 64
OLD = StageConfig(0, d_model=16, L=4, n_h=2, n_kv_attn=1, d_ff=24, d_head=8, chunk_size=8)
NEW = StageConfig(1, d_model=24, L=8, n_h=3, n_kv_attn=1, d_ff=32, d_head=8, chunk_size=8,
                  emb_scale=OLD.emb_scale)


def _decoded(shape, cfg, vocab, seed=1, old=None):
    if old is not None:
        import dataclasses
        _, ops = layer_placement(old, cfg)
        cfg = dataclasses.replace(cfg, ops_override=ops)
    bank = LatentBank(shape, cfg, vocab, seed=seed)
    ctx = DecodeContext(cfg, make_tables(shape, cfg, vocab), bank, vocab)
    return {k: v.detach() for k, v in Genome(shape, seed=seed).decode_all(ctx).items()}


def test_G1_theorem_G1_continuity_at_rho_zero(small_shape):
    torch.manual_seed(0)
    m = Phenotype(OLD, VOCAB, seed=0).double()
    toks = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        before = m(toks)
    for mode in ("genome", "random", "copy"):
        dec = _decoded(small_shape, NEW, VOCAB, old=OLD)
        dec = {k: v.double() for k, v in dec.items()}
        g, plan = grow(m, NEW, mode=mode, decoded=dec)
        g = g.double()
        Ramp(g, plan, n_ramp=10, rho_0=0.0)
        with torch.no_grad():
            after = g(toks)
        assert torch.allclose(before, after, atol=1e-8), mode


def test_G2_G3_old_entries_kept_new_entries_decoded(small_shape):
    m = Phenotype(OLD, VOCAB, seed=0)
    dec = _decoded(small_shape, NEW, VOCAB, old=OLD)
    g, plan = grow(m, NEW, mode="genome", decoded=dec)
    for role, l_old in ((ROLE_Q, 0), (ROLE_UP, 1), (ROLE_O, 2)):
        l_new = plan.layer_map.index(l_old)
        old = m.get(role, l_old)
        new = g.get(role, l_new)
        assert torch.equal(new[: old.shape[0], : old.shape[1]], old)
        assert torch.allclose(new[old.shape[0]:], dec[(role, l_new)][old.shape[0]:])
        assert torch.allclose(new[:, old.shape[1]:], dec[(role, l_new)][:, old.shape[1]:])
    E_old, E_new = m.get(ROLE_EMB, -1), g.get(ROLE_EMB, -1)
    assert torch.equal(E_new[:, : OLD.d_model], E_old)
    assert torch.allclose(E_new[:, OLD.d_model:], dec[(ROLE_EMB, -1)][:, OLD.d_model:])
    # a brand-new layer is the decode in full
    l_fresh = plan.layer_map.index(-1)
    assert torch.allclose(g.get(ROLE_GATE, l_fresh), dec[(ROLE_GATE, l_fresh)])


def test_G5_layer_placement():
    import dataclasses
    a = StageConfig(0, d_model=16, L=8, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)
    b = StageConfig(1, d_model=16, L=12, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)
    lm, ops = layer_placement(a, b)
    # existing layers spread in order: round(i * 11 / 7) = 0,2,3,5,6,8,9,11
    assert [lm[i] for i in (0, 2, 3, 5, 6, 8, 9, 11)] == list(range(8))
    assert [lm[i] for i in (1, 4, 7, 10)] == [-1] * 4
    # old attention layers 3 and 7 sit at 5 and 11; the canonical target has 3
    # attention layers, so one free slot -- the one nearest canonical 7 -- is attention
    assert ops[5] == "attn" and ops[11] == "attn" and ops[7] == "attn"
    assert sum(o == "attn" for o in ops) == 3
    for l in range(12):
        if lm[l] >= 0:
            assert a.op(lm[l]) == ops[l]
    order = [lm[l] for l in range(12) if lm[l] >= 0]
    assert order == sorted(order)                       # relative order preserved
    b2 = dataclasses.replace(b, ops_override=ops)
    c = StageConfig(2, d_model=16, L=16, n_h=2, n_kv_attn=1, d_ff=24, d_head=8)
    lm2, ops2 = layer_placement(b2, c)
    assert sorted(x for x in lm2 if x >= 0) == list(range(12))
    assert sum(o == "attn" for o in ops2) == 4


def test_G8_sampled_latents_and_the_mean_arm(small_shape):
    bank = LatentBank(small_shape, OLD, VOCAB, seed=4)
    with torch.no_grad():
        bank.z_res.add_(1.5)
    import dataclasses
    lm, ops = layer_placement(OLD, NEW)
    new_cfg = dataclasses.replace(NEW, ops_override=ops)
    g = torch.Generator().manual_seed(0)
    grown = bank.grown(new_cfg, lm, g, mode="mean")
    mu, _ = bank.moments("RES")
    assert torch.allclose(grown.z_res[OLD.d_model:], mu.expand(NEW.d_model - OLD.d_model, -1))
    assert torch.equal(grown.z_res[: OLD.d_model], bank.z_res)
    zero = bank.grown(new_cfg, lm, g, mode="zero")
    assert torch.all(zero.z_res[OLD.d_model:] == 0)


def test_G10_new_units_are_distinct_and_copies_are_not(small_shape):
    from anlage.regress import RegressConfig, regress
    m = Phenotype(OLD, VOCAB, seed=0)
    # a genome fit to the old phenotype, as §06.4 requires before any growth
    bank = LatentBank(small_shape, OLD, VOCAB, seed=1)
    ctx = DecodeContext(OLD, make_tables(small_shape, OLD, VOCAB), bank, VOCAB)
    G = Genome(small_shape, seed=1)
    regress(G, ctx, {k: v.detach() for k, v in m.weights().items()},
            RegressConfig(n_fit=300, lr=5e-3, report_every=100, plateau_window=1000))
    import dataclasses
    lm, ops = layer_placement(OLD, NEW)
    new_cfg = dataclasses.replace(NEW, ops_override=ops)
    bank1 = bank.grown(new_cfg, lm, torch.Generator().manual_seed(0))
    ctx1 = DecodeContext(new_cfg, make_tables(small_shape, new_cfg, VOCAB), bank1, VOCAB)
    dec = {k: v.detach() for k, v in G.decode_all(ctx1).items()}
    g, plan = grow(m, NEW, mode="genome", decoded=dec)
    l_new = plan.layer_map.index(0)
    rows = torch.nn.functional.normalize(g.get(ROLE_UP, l_new)[OLD.d_ff:], dim=-1)
    cos = (rows @ rows.T).triu(1)
    n = rows.shape[0]
    assert float(cos.sum() / (n * (n - 1) / 2)) < 0.5
    c, _ = grow(m, NEW, mode="copy", decoded=None, noise=0.0)
    crows = c.get(ROLE_UP, l_new)
    # copies of existing rows: each new row equals one old row (before noise)
    old = m.get(ROLE_UP, 0) * (OLD.d_model / NEW.d_model)      # HyperCloning's fan-in split
    for r in crows[OLD.d_ff:]:
        assert any(torch.allclose(r[: OLD.d_model], o, atol=1e-6) for o in old)


def test_G11_gates_removed_after_the_ramp(small_shape):
    m = Phenotype(OLD, VOCAB, seed=0)
    dec = _decoded(small_shape, NEW, VOCAB, old=OLD)
    g, plan = grow(m, NEW, mode="genome", decoded=dec)
    ramp = Ramp(g, plan, n_ramp=5)
    assert g.res_gate is not None and g.layer_gate
    for _ in range(5):
        ramp.step()
    assert ramp.done and g.res_gate is None and not g.layer_gate and not g.inner_gate


def test_G13_shrunk_is_the_inverse_of_spreading(small_shape):
    import dataclasses
    bank = LatentBank(small_shape, OLD, VOCAB, seed=4)
    lm, ops = layer_placement(OLD, NEW)
    new_cfg = dataclasses.replace(NEW, ops_override=ops)
    grown = bank.grown(new_cfg, lm, torch.Generator().manual_seed(0))
    back = grown.shrunk(OLD)
    assert torch.equal(back.z_res, bank.z_res) and torch.equal(back.z_tok, bank.z_tok)
    for i in range(OLD.L):
        assert torch.equal(back.y[i], bank.y[i])
        assert torch.equal(back.z_inner[i], bank.z_inner[i])
        assert torch.equal(back.z_head[i], bank.z_head[i])
        assert torch.equal(back.z_gate[i], bank.z_gate[i])


def test_G6_kv_grouping_survives_head_growth():
    """`g mod n_kv` is unchanged for every existing head because `n_kv` is a
    lineage constant; a lineage that changes it is refused."""
    import dataclasses
    import pytest
    from anlage.config import PILE, TINYSTORIES, Lineage
    for lin in (TINYSTORIES, PILE):
        for a, b in zip(lin.stages, lin.stages[1:]):
            for g in range(a.n_h):
                assert g % b.n_kv_attn == g % a.n_kv_attn
    bad = dataclasses.replace(TINYSTORIES.stages[1], n_kv_attn=1)
    with pytest.raises(ValueError, match="n_kv_attn"):
        Lineage(name="bad", vocab=TINYSTORIES.vocab, T=TINYSTORIES.T,
                stages=(TINYSTORIES.stages[0], bad, TINYSTORIES.stages[2]),
                data_root=TINYSTORIES.data_root)


def test_G7_rebuilt_optimisers_cover_every_grown_parameter_once(small_shape):
    from anlage.recipe import TrainConfig, build_optimizers
    model = Phenotype(OLD, VOCAB, seed=0)
    grown, plan = grow(model, NEW, mode="random", generator=torch.Generator().manual_seed(0))
    muon_p, adam_p = grown.optimizer_groups()
    ids = [id(p) for p in muon_p] + [id(p) for p in adam_p]
    assert len(ids) == len(set(ids)), "a parameter is in two groups"
    all_ids = {id(p) for p in grown.parameters() if p.requires_grad}
    assert set(ids) == all_ids, "a grown parameter is outside both optimiser groups"
    for p_ in muon_p:
        assert p_.dim() == 2, "Muon takes matrices only"
    tc = TrainConfig(tokens=1024, batch_final=8, T=16, lr_provenance="test")
    muon, adam = build_optimizers(grown, tc)
    opt_ids = {id(p_) for o in (muon, adam) for g in o.param_groups for p_ in g["params"]}
    assert opt_ids == all_ids
    for o in (muon, adam):
        for g in o.param_groups:
            for p_ in g["params"]:
                assert p_ not in o.state or len(o.state[p_]) == 0, "state must be zero at birth"


def test_G9_stage_tokens_reproduce_the_flop_split_within_one_batch():
    from anlage.config import PILE, TINYSTORIES
    for lin in (TINYSTORIES, PILE):
        C_total = lin.flops(len(lin.stages) - 1, 100_000_000)
        toks = lin.stage_tokens(C_total)
        assert len(toks) == len(lin.stages)
        for s, (f, T_s) in enumerate(zip(lin.budget_fractions, toks)):
            one_batch = lin.flops(s, 32 * lin.T)
            assert 0 <= f * C_total - lin.flops(s, T_s) < one_batch, (lin.name, s)
        # a lineage cut to two stages keeps the proportions
        two = lin.stage_tokens(C_total, n_stages=2)
        f0, f1 = lin.budget_fractions[:2]
        assert abs(lin.flops(0, two[0]) / lin.flops(1, two[1]) - f0 / f1) < 1e-2


def test_G14_prior_growth_keeps_old_entries_and_gives_new_units_typical_norms():
    from anlage.grow import unit_prior
    model = Phenotype(OLD, VOCAB, seed=0)
    with torch.no_grad():
        for k, w in model.weights().items():
            if w.dim() == 2:
                w.mul_(3.0)                        # a 'trained' scale the innate init does not have
    Lam = unit_prior(model)
    grown, plan = grow(model, NEW, mode="prior", generator=torch.Generator().manual_seed(0), Lambda=Lam)
    for (role, l_new), w_new in grown.weights().items():
        l_old = -1 if l_new == -1 else plan.layer_map[l_new]
        if l_new != -1 and (l_old < 0 or not model.has(role, l_old)):
            continue
        w_old = model.get(role, l_old)
        sl = tuple(slice(0, s_) for s_ in w_old.shape)
        assert torch.equal(w_new[sl], w_old), (role, l_new)
        if w_new.dim() == 2 and w_new.shape[0] > w_old.shape[0]:
            from anlage.config import ROLE_NAMES
            pr = Lam[ROLE_NAMES[role]]
            norms = w_new[w_old.shape[0]:].norm(dim=1)
            assert (norms - pr["row_mu"]).abs().max() <= 4 * pr["row_sd"] + 1e-6, (role, l_new)
    # continuity at rho = 0 holds for the prior mode as for the others (Theorem G1)
    x = torch.randint(0, VOCAB, (2, 12))
    from anlage.grow import Ramp
    Ramp(grown, plan, n_ramp=10, rho_0=0.0)
    with torch.no_grad():
        a = model(x)
        b = grown(x)
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_G15_shrink_inverts_grow_on_the_kept_entries():
    from anlage.grow import shrink
    model = Phenotype(OLD, VOCAB, seed=3)
    grown, plan = grow(model, NEW, mode="random", generator=torch.Generator().manual_seed(1))
    back = shrink(grown, OLD)
    assert back.cfg.L == OLD.L and back.cfg.d_model == OLD.d_model
    for k, w in model.weights().items():
        assert torch.equal(back.get(*k), w), k
    x = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        assert torch.allclose(model(x), back(x), atol=1e-6)
