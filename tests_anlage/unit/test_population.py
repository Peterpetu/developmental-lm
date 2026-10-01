"""Acceptance tests S1-S10 of docs/anlage/07. SPEC: 07 §7.7"""

from __future__ import annotations

import numpy as np
import torch

from anlage.config import GenomeShape, StageConfig
from anlage.data import EvalSet, N_STRATA
from anlage.genome import Genome
from anlage.model import Phenotype
from anlage.population import MUT_RANK, merge, mutate, select_parents
from screen.crn import WindowPool

VOCAB = 64
CFG = StageConfig(0, d_model=16, L=4, n_h=2, n_kv_attn=1, d_ff=24, d_head=8, chunk_size=8)


class _Shard:
    def __init__(self, n):
        g = torch.Generator().manual_seed(0)
        self.tokens = torch.randint(0, VOCAB, (n,), generator=g).numpy().astype(np.uint16)
        self.starts = np.array([0, n // 3, 2 * n // 3])
        self.lengths = np.diff(np.append(self.starts, n))
        self.n_tokens = n


def test_S1_crn_pairing_of_siblings():
    shard = _Shard(20_000)
    pool = WindowPool(shard, n_windows=500, T=16, pool_seed=7, device="cpu")
    a = [w.clone() for w in pool.batches(batch=8, steps=5, seed=3)]
    b = [w.clone() for w in pool.batches(batch=8, steps=5, seed=3)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    c = [w.clone() for w in pool.batches(batch=8, steps=5, seed=4)]
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_S2_mutation_is_low_rank_and_scale_relative():
    shape = GenomeShape(d_h=32, H=2, d_g=16, d_z=4, d_y=4, d_sigma=8)
    G = Genome(shape, seed=0)
    M = mutate(G, 0.1, torch.Generator().manual_seed(1))
    for (n, p), (_, q) in zip(G.named_parameters(), M.named_parameters()):
        d = (q - p).detach()
        if p.dim() >= 2:
            flat = d.reshape(-1, d.shape[-2], d.shape[-1])
            for i in range(flat.shape[0]):
                assert torch.linalg.matrix_rank(flat[i]) <= MUT_RANK
            src = p.detach().reshape(-1, p.shape[-2], p.shape[-1])
            want = 0.1 * src[0].pow(2).mean().sqrt()
            got = flat[0].pow(2).mean().sqrt()
            assert 0.5 * want <= got <= 2.0 * want, n         # rank-4 noise has a loose RMS
        else:
            assert d.abs().sum() > 0


def test_S3_strata_cover_every_token():
    shard = _Shard(5_000)
    rank = torch.randperm(VOCAB)
    ev = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu")
    assert ev.strata.shape == (8, 16)
    assert int(ev.strata.min()) >= 0 and int(ev.strata.max()) < N_STRATA


def test_S4_lexicase_keeps_a_planted_specialist():
    K, C = 4, 6
    rng = np.random.default_rng(0)
    scores = np.full((K, C), 3.0)
    scores[0, :] = 2.9                    # a generalist, best on 5 of 6 by 0.1
    scores[1, 0] = 2.0                    # a specialist: far best on stratum 0
    scores[1, 1:] = 3.5                   # and worst on the rest
    eps = [0.02] * C
    picked = set()
    for _ in range(200):
        picked.update(select_parents(scores.tolist(), eps, 2, rng))
    assert 1 in picked and 0 in picked
    # mean-CE selection never picks the specialist
    assert int(np.argmin(scores.mean(1))) == 0


def test_S5_S6_merge_algebra_and_acceptance():
    torch.manual_seed(0)
    shard = _Shard(20_000)
    rank = torch.randperm(VOCAB)
    score = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu")
    fitw = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu",
                         offset_windows=8).windows
    base_m = Phenotype(CFG, VOCAB, seed=0)
    W0 = {k: v.detach().clone() for k, v in base_m.weights().items()}
    parents = [Phenotype(CFG, VOCAB, seed=1), Phenotype(CFG, VOCAB, seed=2)]
    child, rep = merge(parents, W0, fitw, score, sigma_e=0.05, steps=5, lr=0.0)
    # with lr 0 the coefficients stay at 1/2: the plain average of the parents
    for k, v in child.weights().items():
        avg = 0.5 * parents[0].get(*k) + 0.5 * parents[1].get(*k)
        assert torch.allclose(v, avg, atol=1e-6) or not rep.accepted
    assert rep.chosen in ("merge", "sibling 0", "sibling 1")
    assert rep.accepted == (rep.ce_fit_merge <= rep.ce_fit_best + 0.05)


def test_S8_disjoint_windows():
    shard = _Shard(20_000)
    rank = torch.randperm(VOCAB)
    a = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu")
    b = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu",
                      offset_windows=8)
    ra = {tuple(r.tolist()) for r in a.windows}
    rb = {tuple(r.tolist()) for r in b.windows}
    assert not (ra & rb)


def _sample_windows(model: Phenotype, n: int, T: int, seed: int) -> torch.Tensor:
    """Ancestral samples `(n, T+1)` from `model`: data whose optimum is the model itself."""
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, VOCAB, (n, 1), generator=g)
    with torch.no_grad():
        for _ in range(T):
            logits = model(toks)[:, -1].float()
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
            toks = torch.cat([toks, nxt], 1)
    return toks


def test_S7_coefficient_fit_recovers_planted_orthogonal_task_vectors():
    """Two parents whose task vectors have disjoint support; the fitted `c`
    is within 0.05 of the grid optimum of the same objective."""
    from anlage.config import ROLE_LOGIT_SCALE, ROLE_UP
    from anlage.model import key_name
    torch.manual_seed(0)
    star = Phenotype(CFG, VOCAB, seed=7)
    with torch.no_grad():
        star.W[key_name(ROLE_LOGIT_SCALE, -1)].mul_(6.0)      # a peaked sampler
    fitw = _sample_windows(star, 8, 16, seed=11)
    shard = _Shard(20_000)
    score = EvalSet.build(shard, torch.randperm(VOCAB), n_windows=8, T=16, split="fitness",
                          device="cpu")
    W_star = {k: v.detach().clone() for k, v in star.weights().items()}
    keys = [(ROLE_UP, 0), (ROLE_UP, 1)]
    g = torch.Generator().manual_seed(1)
    taus = [0.5 * W_star[k].std() * torch.randn(W_star[k].shape, generator=g) for k in keys]
    W0 = {k: v.clone() for k, v in W_star.items()}
    for k, tau in zip(keys, taus):
        W0[k] = W0[k] - tau                                     # the base is `star` minus both
    parents = []
    for k, tau in zip(keys, taus):
        W = {kk: v.clone() for kk, v in W0.items()}
        W[k] = W[k] + tau
        parents.append(Phenotype(CFG, VOCAB, weights=W))
    probe = Phenotype(CFG, VOCAB, weights=W0)

    def ce_at(c1, c2):
        with torch.no_grad():
            probe.W[key_name(*keys[0])].copy_(W0[keys[0]] + c1 * taus[0])
            probe.W[key_name(*keys[1])].copy_(W0[keys[1]] + c2 * taus[1])
            return float(probe.loss(fitw)[0])

    coarse = [round(-0.5 + 0.1 * i, 2) for i in range(26)]
    best = min((ce_at(a, b), a, b) for a in coarse for b in coarse)
    fine = [round(x, 3) for x in torch.arange(-0.1, 0.1001, 0.01).tolist()]
    best = min((ce_at(best[1] + da, best[2] + db), best[1] + da, best[2] + db)
               for da in fine for db in fine)
    child, rep = merge(parents, W0, fitw, score, sigma_e=10.0, steps=400, lr=0.02)
    c1 = rep.coefficients[f"{ROLE_UP}_0"][0]
    c2 = rep.coefficients[f"{ROLE_UP}_1"][1]
    assert abs(c1 - best[1]) <= 0.05 and abs(c2 - best[2]) <= 0.05, (c1, c2, best)
    assert abs(ce_at(c1, c2) - best[0]) <= 1e-3


def _population_phase(seed: int):
    from anlage.lifetime import evaluate, train
    from anlage.recipe import TrainConfig
    torch.manual_seed(seed)
    shard = _Shard(20_000)
    rank = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))
    pool = WindowPool(shard, n_windows=64, T=16, pool_seed=seed, device="cpu")
    fit = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu")
    fitw = EvalSet.build(shard, rank, n_windows=8, T=16, split="fitness", device="cpu",
                         offset_windows=8).windows
    base = Phenotype(CFG, VOCAB, seed=seed)
    W0 = {k: v.detach().clone() for k, v in base.weights().items()}
    tc = TrainConfig(tokens=3 * 8 * 16, batch_final=8, T=16, lr_provenance="test")
    sibs, scores = [], []
    for k in range(2):
        m = Phenotype(CFG, VOCAB, weights=W0)
        train(m, pool, fit, tc, order_seed=seed ^ k, compile_mode=None, device="cpu")
        sibs.append(m)
        scores.append(evaluate(m, fit)[1])
    rng = np.random.default_rng(seed)
    parents = select_parents(scores, [0.01] * N_STRATA, 2, rng)
    child, rep = merge([sibs[i] for i in sorted(set(parents))], W0, fitw, fit, 0.05,
                       steps=5, lr=0.05)
    return parents, rep, {k: v.detach().clone() for k, v in child.weights().items()}


def test_S10_population_phase_is_bit_reproducible_from_its_seed():
    a = _population_phase(3)
    b = _population_phase(3)
    assert a[0] == b[0]
    assert a[1].coefficients == b[1].coefficients and a[1].chosen == b[1].chosen
    for k in a[2]:
        assert torch.equal(a[2][k], b[2][k]), k


def test_S9_population_budget_equals_f0_C_total_and_SINGLE_spends_the_same():
    from anlage.config import PILE, TINYSTORIES
    for lin in (TINYSTORIES, PILE):
        C_total = lin.flops(len(lin.stages) - 1, 50_000_000)
        f0 = lin.budget_fractions[0]
        for K in (1, 4, 8):
            T_sib = lin.sibling_tokens(C_total, K)
            spent = K * lin.flops(0, T_sib)
            assert 0 <= f0 * C_total - spent < K * lin.flops(0, 32 * lin.T), (lin.name, K)
        single = lin.flops(0, lin.sibling_tokens(C_total, 1))
        pop = 4 * lin.flops(0, lin.sibling_tokens(C_total, 4))
        assert abs(single - pop) < 4 * lin.flops(0, 1)
