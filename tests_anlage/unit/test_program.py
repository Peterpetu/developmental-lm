"""Acceptance tests M1-M12 of docs/anlage/12. SPEC: 12 §12.10"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anlage.data import (ClassifiedPool, classify_documents, systematic_counts)
from anlage.program import (ALPHA_0, MU_MIN, Program, Surrogate, clip_simplex,
                            dirichlet_neighbourhood, fit_surrogate, project_simplex_floor,
                            quadratic_features, solve_trust_region)

VOCAB = 64


class _Shard:
    """A synthetic shard with documents of varied length and rarity."""

    def __init__(self, n_docs: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        lengths = rng.integers(20, 200, size=n_docs).astype(np.int64)
        toks = []
        for i, L in enumerate(lengths):
            # half the documents draw from the rare half of the vocabulary more often
            p_rare = 0.6 if i % 2 else 0.2
            rare = rng.random(L) < p_rare
            t = np.where(rare, rng.integers(32, 64, size=L), rng.integers(0, 32, size=L))
            toks.append(t)
        self.tokens = np.concatenate(toks).astype(np.uint16)
        self.lengths = lengths
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        self.n_tokens = int(self.tokens.shape[0])


def _rank():
    # tokens 0..31 frequent (rank < 1000 is every token here), so use a rank map that puts
    # the upper half at rank >= 1000
    r = torch.arange(VOCAB)
    r[32:] += 1000
    return r


def test_M1_classes_are_innate_and_reproducible():
    sh = _Shard()
    c1, m1 = classify_documents(sh, _rank())
    c2, m2 = classify_documents(_Shard(), _rank())
    assert np.array_equal(c1, c2) and m1 == m2
    assert m1["n_classes"] == 6 and sum(m1["doc_counts"]) == m1["n_docs"]
    assert len(m1["length_edges"]) == 2 and m1["length_edges"][0] <= m1["length_edges"][1]
    assert min(m1["doc_counts"]) > 0


def test_M2_systematic_rounding_sums_to_B_and_is_unbiased():
    rng = np.random.default_rng(0)
    for n_C in (2, 6, 22):
        mu = rng.dirichlet(np.ones(n_C))
        for B in (1, 8, 32, 33):
            tot = np.zeros(n_C)
            n_draw = 10_000
            for _ in range(n_draw):
                n = systematic_counts(B, mu, float(rng.random()))
                assert int(n.sum()) == B and (n >= 0).all()
                tot += n
            assert np.abs(tot / n_draw - B * mu).max() < 1e-2 * B + 1e-2


def test_M3_M4_crn_across_mixtures_and_the_corpus_mixture():
    sh = _Shard()
    cls, _ = classify_documents(sh, _rank())
    pool = ClassifiedPool(sh, cls, 6, n_windows=2000, T=16, pool_seed=1, device="cpu")
    mu_a = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    mu_b = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5], dtype=torch.float64)
    seqs = {}
    for name, mu in (("a", mu_a), ("b", mu_b)):
        per_class = {c: [] for c in range(6)}
        for w in pool.batches(batch=16, steps=40, seed=3, mu=mu):
            for row in w:
                # recover the class of each sampled window by matching against the pool
                j = int(torch.nonzero((pool.windows == row).all(1))[0])
                per_class[int(pool.window_class[j])].append(j)
        seqs[name] = per_class
    for c in range(6):
        a, b = seqs["a"][c], seqs["b"][c]
        n = min(len(a), len(b))
        assert a[:n] == b[:n], f"class {c}: streams differ under one seed"   # M3
    # M4: at the corpus mixture the sampled class histogram matches the pool's
    mu_c = pool.corpus_mixture
    assert abs(float(mu_c.sum()) - 1.0) < 1e-12
    hist = torch.zeros(6)
    n_tot = 0
    for w in pool.batches(batch=32, steps=400, seed=5, mu=mu_c):
        for row in w:
            j = int(torch.nonzero((pool.windows == row).all(1))[0])
            hist[int(pool.window_class[j])] += 1
            n_tot += 1
    assert (hist / n_tot - mu_c.float()).abs().max() < 1e-2


def test_M5_floor_and_simplex():
    x = torch.tensor([0.9, 0.05, 0.03, 0.01, 0.005, 0.005])
    y = clip_simplex(x, MU_MIN)
    assert float(y.min()) >= MU_MIN - 1e-12 and abs(float(y.sum()) - 1.0) < 1e-12
    assert torch.argmax(y) == 0
    with pytest.raises(ValueError):
        clip_simplex(torch.ones(60) / 60, MU_MIN)
    # the Dirichlet neighbourhood keeps the centre and obeys the floor
    mu_0 = torch.full((6,), 1 / 6, dtype=torch.float64)
    nb = dirichlet_neighbourhood(mu_0, ALPHA_0, 8, seed=1)
    assert nb.shape == (8, 6) and torch.allclose(nb[0], mu_0)
    assert float(nb.min()) >= MU_MIN - 1e-12 and (nb.sum(1) - 1).abs().max() < 1e-12
    assert torch.equal(nb, dirichlet_neighbourhood(mu_0, ALPHA_0, 8, seed=1))
    # projection onto the floored simplex is idempotent and feasible
    v = torch.tensor([0.5, -0.2, 0.9, 0.0, 0.1, 0.3], dtype=torch.float64)
    pr = project_simplex_floor(v, MU_MIN)
    assert abs(float(pr.sum()) - 1.0) < 1e-12 and float(pr.min()) >= MU_MIN - 1e-12
    assert torch.allclose(project_simplex_floor(pr, MU_MIN), pr, atol=1e-12)


def test_M6_identifiability_and_planted_quadratic_recovery():
    n_C, Q = 4, 3
    rng = np.random.default_rng(0)
    beta_true = torch.from_numpy(rng.normal(size=(1 + n_C + n_C * (n_C + 1) // 2, Q)))
    few = torch.from_numpy(rng.dirichlet(np.ones(n_C), size=4 * n_C - 1))
    F_few = quadratic_features(few) @ beta_true
    assert fit_surrogate(few, F_few).linear_only
    many = torch.from_numpy(rng.dirichlet(np.ones(n_C), size=80))
    F_many = quadratic_features(many) @ beta_true
    sur = fit_surrogate(many, F_many, lambda_F=0.0)
    assert not sur.linear_only
    probe = torch.from_numpy(rng.dirichlet(np.ones(n_C), size=10))
    assert torch.allclose(sur.predict(probe), quadratic_features(probe) @ beta_true, atol=1e-6)
    assert sur.loo_rmse < 1e-6


def test_M7_trust_region_matches_the_interior_optimum_or_stops_on_the_boundary():
    n_C = 5
    mu_0 = torch.full((n_C,), 1 / n_C, dtype=torch.float64)
    w = torch.ones(1, dtype=torch.float64)

    def planted(centre):
        # F(mu) = ||mu - centre||^2 as a quadratic surrogate: a = |c|^2, b = -2c, C = I
        d = n_C
        iu = torch.triu_indices(d, d)
        beta = torch.zeros(1 + d + d * (d + 1) // 2, 1, dtype=torch.float64)
        beta[0, 0] = float((centre ** 2).sum())
        beta[1:1 + d, 0] = -2 * centre
        for k in range(iu.shape[1]):
            i, j = int(iu[0, k]), int(iu[1, k])
            beta[1 + d + k, 0] = 1.0 if i == j else 0.0
        return Surrogate(beta=beta, n_classes=d, linear_only=False, lambda_F=0.0, n_samples=0, loo_rmse=0.0)

    inside = mu_0 + torch.tensor([0.05, -0.05, 0.02, -0.02, 0.0], dtype=torch.float64)
    mu, info = solve_trust_region(planted(inside), w, mu_0, tau=0.2)
    assert torch.allclose(mu, inside, atol=1e-6), (mu, inside)
    assert info["simplex_error"] < 1e-12 and info["min_weight"] >= MU_MIN - 1e-12
    far = torch.tensor([0.8, 0.05, 0.05, 0.05, 0.05], dtype=torch.float64)
    mu, info = solve_trust_region(planted(far), w, mu_0, tau=0.2)
    assert info["on_boundary"], info
    assert abs(float((mu - mu_0).norm()) - 0.2) < 1e-6
    assert info["simplex_error"] < 1e-12 and info["min_weight"] >= MU_MIN - 1e-12
    # the boundary point is the closest feasible point to `far` in the ball (convexity)
    direction = (far - mu_0) / (far - mu_0).norm()
    assert float(((mu - mu_0) / (mu - mu_0).norm() - direction).norm()) < 1e-4


def test_M8_program_round_trip_and_scale_freedom(tmp_path):
    mu = torch.stack([torch.full((6,), 1 / 6, dtype=torch.float64)] * 3)
    prog = Program(mu=mu, omega=(0.15, 0.25, 0.60), eta={"lr_mult": 1.0, "a_0": 4.0},
                   s_state={"o": torch.randn(3, 4)}, Lambda={"RES": {"mu": [0.1], "s": [0.2]}},
                   classes={"n_classes": 6})
    prog.save(tmp_path / "g.pt")
    back = Program.load(tmp_path / "g.pt")
    assert prog.equals(back)
    assert prog.n_reals() == 18 + 3 + 2 + 12 + 2
    # the unit prior as `unit_prior` returns it: per role, scalar moments
    prog2 = Program(mu=mu, omega=(0.15, 0.25, 0.60), eta={}, s_state=None,
                    Lambda={"Q": {"row_mu": 1.0, "row_sd": 0.1, "col_mu": 0.9, "col_sd": 0.2}}, classes={})
    assert prog2.n_reals() == 18 + 3 + 4
    with pytest.raises(ValueError):
        Program(mu=mu * 0.5, omega=(0.15, 0.25, 0.60), eta={}, s_state=None, Lambda={}, classes={})


def test_M11_programme_regression_is_bit_reproducible():
    rng = np.random.default_rng(7)
    n_C, Q = 6, 12
    mu_0 = torch.full((n_C,), 1 / n_C, dtype=torch.float64)
    nb = dirichlet_neighbourhood(mu_0, ALPHA_0, 8, seed=3)
    F = torch.from_numpy(rng.normal(size=(8, Q)))
    w = torch.full((Q,), 1 / Q, dtype=torch.float64)
    outs = []
    for _ in range(2):
        sur = fit_surrogate(nb, F)
        mu, info = solve_trust_region(sur, w, mu_0)
        outs.append((sur.beta.clone(), mu.clone(), info))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    assert outs[0][2] == outs[1][2]


def test_M9_s_birth_decodes_vectors_and_leaves_matrices_innate(small_shape, tiny_stage, vocab):
    from anlage.genome import MAT_INDEX, DecodeContext
    from anlage.model import Phenotype
    from anlage.regress import RegressConfig, express_scale_profile, fit_scale_profile
    from tests_anlage.conftest import make_tables
    torch.manual_seed(0)
    trained = Phenotype(tiny_stage, vocab, seed=1)
    with torch.no_grad():
        for (role, l), w in trained.weights().items():
            if w.dim() == 1:
                w.add_(0.5 * torch.randn_like(w))          # a 'trained' scale profile
    tables = make_tables(small_shape, tiny_stage, vocab)
    G, bank, rep = fit_scale_profile(trained, small_shape, tables, rc=RegressConfig(n_fit=300, lr=1e-2))
    assert set(rep.residual) and all(name not in rep.residual for name in ("Q", "UP", "EMB"))
    free = Phenotype(tiny_stage, vocab, seed=5)
    W_free = {k: v.detach().clone() for k, v in free.weights().items()}
    ctx = DecodeContext(tiny_stage, tables, bank, vocab, tied=True)
    W_birth = express_scale_profile(G, ctx, W_free)
    for k, w in W_free.items():
        if k[0] in MAT_INDEX:
            assert torch.equal(W_birth[k], w), k                     # matrices untouched
        else:
            assert not torch.equal(W_birth[k], w), k                 # vectors decoded
            rel = float((W_birth[k] - trained.get(*k)).norm() / trained.get(*k).norm().clamp_min(1e-12))
            assert rel < 0.5, (k, rel)


def test_M13_v3_evaluation_sets_share_no_document_and_one_composition():
    from anlage.data import EvalSet, classify_eval_windows
    a, b = _Shard(300, seed=1), _Shard(300, seed=2)
    rank = _rank()
    sets = [EvalSet.sample_documents([a, b], rank, n_windows=120, T=16, split="x", seed=7, mod=3, residue=r,
                                     device="cpu") for r in range(3)]
    docs = []
    for ev in sets:
        d = set()
        for k, sh in enumerate((a, b)):
            m = ev.sources == k
            base = 0 if k == 0 else len(a.starts)
            d |= {base + int(x) for x in (np.searchsorted(sh.starts, ev.offsets[m], side="right") - 1)}
        docs.append(d)
    assert not (docs[0] & docs[1]) and not (docs[1] & docs[2]) and not (docs[0] & docs[2])
    # composition: classify by document across both shards; every set draws from both shards
    ca, _ = classify_documents(a, rank)
    cb, _ = classify_documents(b, rank, manifest=_)
    for ev in sets:
        assert set(ev.sources.tolist()) == {0, 1}
        wc = classify_eval_windows(ev, [a, b], [ca, cb])
        assert wc.shape[0] == 120 and int(wc.min()) >= 0 and int(wc.max()) < 6
    # determinism
    again = EvalSet.sample_documents([a, b], rank, n_windows=120, T=16, split="x", seed=7, mod=3, residue=0, device="cpu")
    assert again.digest == sets[0].digest
