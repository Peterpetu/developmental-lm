"""SPORE acceptance tests. SPEC: docs/spore/01-model.md, 02-protocol.md (G-BPF, G-DET).

Small dimensions on CPU; every property is a mathematical identity or a planted-signal
recovery, never a smoke test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spore.cortex import Cortex, ResidualScorer, activate, make_candidates
from spore.genome import Genome, GENE_RANGES, crossover, mutate
from spore.hdc import ContextCoder, apply_power, atoms, permutation, permutation_powers
from spore.memory import Memory
from spore.readout import AdamState, GramAccumulator, calibrate, cross_entropy_rows, delta_rule_step, golden_section, softcap

DEV = "cpu"


# ---------------------------------------------------------------- §1.1 atoms
def test_atoms_unit_norm_and_near_orthogonal():
    A = atoms(64, 2048, seed=3, device=DEV)
    assert torch.allclose(A.norm(dim=1), torch.ones(64), atol=1e-5)
    G = A @ A.T
    off = G[~torch.eye(64, dtype=torch.bool)]
    assert abs(float(off.mean())) < 0.01
    assert abs(float(off.std()) - 1 / math.sqrt(2048)) < 0.005          # Plate 1995: sd 1/sqrt(D)


def test_atoms_are_a_deterministic_function_of_the_seed():
    assert torch.equal(atoms(16, 128, seed=5, device=DEV), atoms(16, 128, seed=5, device=DEV))
    assert not torch.equal(atoms(16, 128, seed=5, device=DEV), atoms(16, 128, seed=6, device=DEV))


# ---------------------------------------------------------------- §1.2 binding
def test_permutation_powers_compose():
    perm = permutation(64, seed=1, device=DEV)
    P = permutation_powers(perm, 6)
    x = torch.randn(64)
    r1 = apply_power(x, P, 1)
    r3 = apply_power(x, P, 3)
    # rho^3 = rho(rho(rho(x)))
    y = apply_power(apply_power(r1, P, 1), P, 1)
    assert torch.equal(y, r3)
    assert torch.equal(apply_power(x, P, 0), x)
    assert torch.allclose(r3.norm(), x.norm())                             # orthogonal


def test_ngram_code_identity_and_suffix_overlap():
    V, D = 32, 4096
    A = atoms(V, D, seed=0, device=DEV)
    perm = permutation(D, seed=1, device=DEV)
    coder = ContextCoder(A, perm, orders=(1, 2, 4), gamma_b=0.9, gamma_e=0.7, J_e=8)
    # two windows whose last 4 tokens agree from position 5 on, differ before
    t1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    t2 = torch.tensor([[9, 9, 9, 9, 5, 6, 7, 8]])
    c1, c2 = coder.codes(t1), coder.codes(t2)
    assert torch.allclose(c1["c4"][0, 7], c2["c4"][0, 7], atol=1e-6)     # same last 4 tokens -> same code
    assert abs(float(c1["c4"][0, 7].norm()) - 1) < 1e-5
    # position 6: last-4 = (3,4,5,6) vs (9,9,5,6): shared suffix of length 2 -> inner product 2/4
    ip = float(c1["c4"][0, 5] @ c2["c4"][0, 5])
    assert abs(ip - 0.5) < 0.06
    # order is carried: swapping the two last tokens changes the bigram code
    t3 = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 7]])
    c3 = coder.codes(t3)
    assert abs(float(c1["c2"][0, 7] @ c3["c2"][0, 7])) < 0.1
    # without binding the bigram code is a bag: swapping leaves it unchanged
    coder_nb = ContextCoder(A, perm, orders=(1, 2), gamma_b=0.9, gamma_e=0.7, J_e=4, binding=False)
    assert torch.allclose(coder_nb.codes(t1)["c2"][0, 7], coder_nb.codes(t3)["c2"][0, 7], atol=1e-6)


def test_codes_are_causal():
    V, D = 16, 512
    A = atoms(V, D, seed=0, device=DEV)
    perm = permutation(D, seed=1, device=DEV)
    coder = ContextCoder(A, perm, orders=(1, 3), gamma_b=0.9, gamma_e=0.7, J_e=6)
    t = torch.randint(0, V, (2, 12))
    t2 = t.clone()
    t2[:, 8:] = (t2[:, 8:] + 1) % V
    a, b = coder.codes(t), coder.codes(t2)
    for k in ("c1", "c3", "b", "e"):
        assert torch.allclose(a[k][:, :8], b[k][:, :8], atol=1e-6), k


# ---------------------------------------------------------------- §1.5 memory
def test_memory_stores_and_retrieves_and_does_not_leak():
    V, D, dk = 16, 1024, 256
    A = atoms(V, D, seed=0, device=DEV)
    perm = permutation(D, seed=1, device=DEV)
    coder = ContextCoder(A, perm, orders=(1, 2), gamma_b=0.9, gamma_e=0.7, J_e=4)
    mem = Memory(A, d_k=dk, seed=2, beta=1.0, chunk_size=4)
    # a b c d a b c d : at the second 'c' (t=6) the context (a,b)->? was seen: previous key (a,b) wrote value 'c'
    toks = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    c = coder.codes(toks)
    r = mem.retrieve(c["c2"], c["emb"])                                   # (1, 8, dk)
    m = mem.cleanup(r)                                                    # (1, 8, V)
    # at t=5 (token b, context (a,b)) the memory holds (key(a,b) -> value c) written at t=2; retrieval predicts c
    assert int(m[0, 5].argmax()) == 2
    # at t=1 nothing about (a,b) has been written yet -> no preference for token 2 over others beyond noise
    assert float(m[0, 1].abs().max()) < 0.5 * float(m[0, 5, 2])
    # no leak: changing the token at t=6 must not change the retrieval at t=5
    toks2 = toks.clone()
    toks2[0, 6] = 7
    c2 = coder.codes(toks2)
    r2 = mem.retrieve(c2["c2"], c2["emb"])
    assert torch.allclose(r[0, :6], r2[0, :6], atol=1e-5)


# ---------------------------------------------------------------- §1.4 readout
def test_ridge_recovers_a_linear_map_and_scores_sum_to_one_with_bias():
    torch.manual_seed(0)
    n, F, V = 4000, 24, 6
    Z = torch.randn(n, F)
    Z[:, 0] = 1.0                                                          # the bias column
    Wt = torch.randn(F, V)
    P = torch.softmax(Z @ Wt, dim=1)
    tgt = torch.multinomial(P, 1).squeeze(1)
    g = GramAccumulator(F, V, DEV)
    for i in range(0, n, 500):
        g.add(Z[i:i + 500], tgt[i:i + 500])
    W = g.solve(1e-9)
    S = Z @ W
    assert torch.allclose(S.sum(1), torch.ones(n), atol=1e-3)            # exact: the fit of a constant target
    # at lam -> 0 the ridge is the least-squares solution
    Y = torch.nn.functional.one_hot(tgt, V).float()
    W_ls = torch.linalg.lstsq(Z, Y).solution
    assert torch.allclose(W, W_ls, atol=2e-3)
    # and the scores are the least-squares estimate of P(v | z): closer to P than the marginal is
    marg = Y.mean(0, keepdim=True).expand(n, V)
    assert float((S - P).abs().mean()) < float((marg - P).abs().mean())


def test_gram_grow_keeps_old_block_and_zeroes_new():
    g = GramAccumulator(3, 4, DEV)
    g.add(torch.randn(10, 3), torch.randint(0, 4, (10,)))
    G0, C0 = g.G.clone(), g.C.clone()
    g.grow(5)
    assert torch.equal(g.G[:3, :3], G0) and torch.equal(g.C[:3], C0)
    assert float(g.G[3:].abs().sum()) == 0.0 and float(g.G[:, 3:].abs().sum()) == 0.0
    W = g.solve(0.1)
    assert W.shape == (5, 4) and torch.isfinite(W).all()


def test_delta_rule_gradient_matches_autograd_on_a_tiny_problem():
    torch.manual_seed(1)
    n, F, V = 64, 8, 5
    Z = torch.randn(n, F)
    tgt = torch.randint(0, V, (n,))
    W = torch.randn(F, V) * 0.1
    off = torch.randn(n, V) * 0.1
    # reference gradient through autograd (allowed in a test; never in the organism)
    Wr = W.clone().requires_grad_(True)
    loss = cross_entropy_rows(softcap(Z @ Wr + off), tgt).mean()
    loss.backward()
    adam = AdamState(W, lr=1.0, beta1=0.0, beta2=0.0, eps=0.0)
    W2 = W.clone()
    delta_rule_step(W2, Z, tgt, off, adam, lr_scale=1.0)
    # with beta1 = beta2 = 0, eps = 0, lr = 1: W2 = W - grad / |grad| (sign of the gradient)
    assert torch.allclose(W2, W - torch.sign(Wr.grad), atol=1e-5)
    # and Adam's first moment equals the closed-form gradient
    assert torch.allclose(adam.m, Wr.grad, atol=1e-6)


def test_golden_section_finds_a_minimum():
    x, fx = golden_section(lambda t: (t - 1.7) ** 2 + 3, 0.0, 5.0, tol=1e-4)
    assert abs(x - 1.7) < 1e-3 and abs(fx - 3) < 1e-6


def test_calibration_lowers_ce():
    torch.manual_seed(2)
    n, V = 2000, 7
    P = torch.softmax(torch.randn(n, V) * 2, dim=1)
    tgt = torch.multinomial(P, 1).squeeze(1)
    S = P + 0.02 * torch.randn(n, V)                                     # scores ~ probabilities, as the ridge gives
    ts, tm, ce = calibrate(S.half(), None, tgt, rounds=1)
    ce_raw = float(cross_entropy_rows(softcap(S), tgt).mean())
    assert ce < ce_raw and ts > 1.0


# ---------------------------------------------------------------- §1.6 cortex
def test_candidates_are_sparse_and_seeded():
    W, ph = make_candidates(10, 100, 8, seed=4, device=DEV)
    assert torch.equal(W, make_candidates(10, 100, 8, seed=4, device=DEV)[0])
    assert ((W != 0).sum(1) == 8).all()
    assert torch.allclose(W.abs().sum(1), torch.full((10,), 8 / math.sqrt(8)), atol=1e-5)


def test_wave_selection_recovers_the_planted_useful_unit():
    torch.manual_seed(3)
    n, F_in, V, M = 4000, 32, 4, 50
    X = torch.randn(n, F_in)
    W, ph = make_candidates(M, F_in, 4, seed=1, device=DEV)
    # plant: the target depends on candidate 17's activation
    h17 = torch.relu(X @ W[17])
    logits = torch.zeros(n, V)
    logits[:, 1] = 3.0 * h17
    P = torch.softmax(logits, dim=1)
    tgt = torch.multinomial(P, 1).squeeze(1)
    p_current = torch.full((n, V), 1.0 / V)                                # a readout that knows nothing
    resid = -p_current
    resid[torch.arange(n), tgt] += 1.0
    sc = ResidualScorer(M, V, DEV)
    for i in range(0, n, 1000):
        sc.add(activate(X[i:i + 1000], W, ph, "relu"), resid[i:i + 1000])
    s = sc.scores()
    assert int(s.argmax()) == 17
    cx = Cortex("relu")
    cx.add_wave(W[[17]], ph[[17]])
    h = cx.forward(X)
    assert h.shape == (n, 1) and torch.allclose(h[:, 0], h17)


# ---------------------------------------------------------------- §1.7 genome
def test_genome_validation_and_mutation_stay_in_range():
    with pytest.raises(ValueError):
        Genome(D=3000)
    with pytest.raises(ValueError):
        Genome(gamma_b=0.1)
    rng = np.random.default_rng(0)
    g = Genome()
    for _ in range(300):
        g = mutate(g, rng, p=0.5)
        for name, (lo, hi) in GENE_RANGES.items():
            assert lo <= getattr(g, name) <= hi
        assert set(g.orders) <= set(range(1, 9)) and g.orders
    c = crossover(Genome(), g, rng)
    assert isinstance(c, Genome)
    assert Genome.from_dict(g.as_dict()) == g


def test_genome_digest_is_stable():
    assert Genome().digest() == Genome().digest()
    assert Genome().digest() != Genome(s_A=9).digest()


# ---------------------------------------------------------------- G-BPF
def test_no_organism_tensor_requires_grad_and_backward_is_never_called(monkeypatch):
    called = {"n": 0}
    orig = torch.Tensor.backward

    def spy(self, *a, **k):
        called["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    V, D = 16, 256
    A = atoms(V, D, seed=0, device=DEV)
    perm = permutation(D, seed=1, device=DEV)
    coder = ContextCoder(A, perm, orders=(1, 2), gamma_b=0.9, gamma_e=0.7, J_e=4)
    mem = Memory(A, d_k=64, seed=2, beta=1.0, chunk_size=4)
    toks = torch.randint(0, V, (2, 8))
    c = coder.codes(toks)
    r = mem.retrieve(c["c2"], c["emb"])
    Z = torch.cat([c["c1"], c["c2"], r], -1).reshape(16, -1)
    g = GramAccumulator(Z.shape[1], V, DEV)
    g.add(Z, toks.reshape(-1))
    W = g.solve(0.1)
    adam = AdamState(W, lr=1e-3)
    delta_rule_step(W, Z, toks.reshape(-1), None, adam, 1.0)
    for t in (A, perm, mem.P_k, mem.P_v, Z, W, g.G, g.C):
        assert not t.requires_grad
    assert called["n"] == 0


# ---------------------------------------------------------------- §1.2 variant 2a: the exact order-1 block
def test_onehot_block_is_the_identity_code():
    from spore.life import LifeConfig, Organism
    g = Genome(D=1024, d_k=256, orders=(1, 2), onehot=True)
    cfg = LifeConfig(budget=16 * 8, batch_windows=16, T=8, device=DEV)
    org = Organism(g, cfg, V=32)
    toks = torch.randint(0, 32, (2, 8))
    Z, m, Xc = org.features(toks)
    assert Z.shape == (16, org.F) and org.F == 1 + 32 + 4 * 1024 + 256
    block = Z[:, 1:33]
    assert torch.equal(block.argmax(1), toks.reshape(-1)) and torch.allclose(block.sum(1), torch.ones(16))
    assert Xc.shape[1] == org.F - 1 and Xc.data_ptr() == Z[:, 1:].data_ptr()      # a view, not a copy


# ---------------------------------------------------------------- §1.1 variant 1b: semantic atoms
def test_ppmi_svd_groups_distributionally_similar_tokens():
    from spore.semantic import ppmi_svd, semantic_atoms
    torch.manual_seed(0)
    V = 40
    # two classes of "nouns" (0..19) that are followed by two disjoint classes of "verbs" (20..39)
    N = torch.zeros(V, V)
    for u in range(20):
        verbs = range(20, 30) if u < 10 else range(30, 40)
        for v in verbs:
            N[u, v] = 50 + torch.randint(0, 20, (1,)).item()
    for v in range(20, 40):
        for u in range(20):
            N[v, u] = 5 + torch.randint(0, 3, (1,)).item()        # verbs are followed by any noun
    E, S = ppmi_svd(N, d=4)
    assert E.shape == (V, 8) and (S > 0).all()
    En = E / E.norm(dim=1, keepdim=True)
    within = float((En[:10] @ En[:10].T).mean())
    across = float((En[:10] @ En[10:20].T).mean())
    assert within > across + 0.3
    A = atoms(V, 256, seed=0, device=DEV)
    A2 = semantic_atoms(A, N, d=4, eta=1.0, seed=1)
    assert torch.allclose(A2.norm(dim=1), torch.ones(V), atol=1e-5)
    G = A2 @ A2.T
    assert float(G[:10, :10].mean()) > float(G[:10, 10:20].mean()) + 0.2
    assert torch.equal(semantic_atoms(A, N, d=4, eta=0.0, seed=1), A)


# ---------------------------------------------------------------- §1.2 variant 2b: the reservoir
def test_reservoir_is_causal_bounded_and_at_its_spectral_radius():
    from spore.hdc import Reservoir
    R = Reservoir(64, seed=3, rho=0.9, device=DEV)
    assert abs(float(torch.linalg.eigvals(R.W_res).abs().max()) - 0.9) < 1e-4
    A = atoms(16, 64, seed=0, device=DEV)
    t = torch.randint(0, 16, (2, 12))
    x = R.run(A[t])
    assert x.shape == (2, 12, 64) and float(x.abs().max()) <= 1.0
    t2 = t.clone()
    t2[:, 8:] = (t2[:, 8:] + 1) % 16
    assert torch.allclose(R.run(A[t2])[:, :8], x[:, :8], atol=1e-6)


# ---------------------------------------------------------------- Definition 6b: output-side sharing
def test_delta_rule_with_output_sharing_matches_autograd():
    torch.manual_seed(4)
    n, F, V, e = 32, 6, 9, 3
    Z = torch.randn(n, F)
    tgt = torch.randint(0, V, (n,))
    W = torch.randn(F, V) * 0.1
    U = torch.randn(F, e) * 0.1
    E_out = torch.randn(V, e)
    Wr, Ur = W.clone().requires_grad_(True), U.clone().requires_grad_(True)
    loss = cross_entropy_rows(softcap(Z @ Wr + (Z @ Ur) @ E_out.T), tgt).mean()
    loss.backward()
    aW = AdamState(W, lr=1.0, beta1=0.0, beta2=0.0, eps=0.0)
    aU = AdamState(U, lr=1.0, beta1=0.0, beta2=0.0, eps=0.0)
    W2, U2 = W.clone(), U.clone()
    delta_rule_step(W2, Z, tgt, None, aW, 1.0, U=U2, E_out=E_out, adam_U=aU)
    assert torch.allclose(aW.m, Wr.grad, atol=1e-6) and torch.allclose(aU.m, Ur.grad, atol=1e-6)


def test_ppmi_svd_is_deterministic_given_its_seed():
    from spore.semantic import ppmi_svd
    N = (torch.rand(50, 50) < 0.2).float() * torch.randint(1, 30, (50, 50)).float()
    E1, _ = ppmi_svd(N, d=6, seed=3)
    torch.manual_seed(999)
    torch.rand(10)
    E2, _ = ppmi_svd(N, d=6, seed=3)
    assert torch.equal(E1, E2)


# ---------------------------------------------------------------- Definition 11b: ridge units
def test_ridge_units_predict_their_random_token_subsets():
    from spore.cortex import hash_targets
    torch.manual_seed(5)
    n, F, V, k = 6000, 20, 12, 4
    Z = torch.randn(n, F)
    Z[:, 0] = 1.0
    Wt = torch.randn(F, V) * 1.5
    tgt = torch.multinomial(torch.softmax(Z @ Wt, dim=1), 1).squeeze(1)
    g = GramAccumulator(F, V, DEV)
    g.add(Z, tgt)
    H = hash_targets(V, k, seed=1, device=DEV)
    Wu = g.solve_targets(1e-3, H)                                          # (F, k)
    pred = Z @ Wu                                                          # predicted +-1 subset membership
    truth = H[tgt]                                                         # (n, k)
    corr = torch.stack([torch.corrcoef(torch.stack([pred[:, j], truth[:, j]]))[0, 1] for j in range(k)])
    assert (corr > 0.3).all()
    # and the same units by the direct least-squares route: W_ls Y == W (Y-projection commutes with the solve)
    W_full = g.solve(1e-3)
    assert torch.allclose(W_full @ H, Wu, atol=1e-3)


def test_solve_refuses_a_non_finite_gram():
    g = GramAccumulator(3, 4, DEV)
    g.add(torch.randn(10, 3), torch.randint(0, 4, (10,)))
    g.G[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        g.solve(0.1)
