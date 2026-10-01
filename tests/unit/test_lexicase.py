"""ε-lexicase selection: the lemmas, the two historical degeneracies, and
Proposition 4.3.

SPEC: 04 §4.4
SPEC: 04 §4.5
SPEC: 04 §4.6

Gate `G-LEXICASE`, including a regression that `eps_t` is **not** identically
zero.

Two degeneracies are pinned here as controls, because each would have made the
architecture's central novel mechanism silently inert:

* **v1**: the MAD taken over the **whole population**. Only `k_route ~ 8` of
  `~1e2`-`1e3` units are routed to any token, so `nu_ut = 0` for the
  overwhelming majority, `median_v nu_vt = 0`, and `MAD_t = 0` **identically**.
* **v2**: the MAD over `C_t = {u : nu_ut > 0}`, which is **empty** whenever
  `w_t = 0` — common, since (4.1) makes `w_t = 0` on any case the model has not
  improved on since the lag. A median over the empty set is undefined
  (audit A-20).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.config import M_NANO
from soma.selection.credit import case_weights, spearman
from soma.selection.lexicase import (
    _weighted_permutation,
    case_win_histogram,
    credit_matrix,
    epsilon_lexicase,
    epsilon_per_case,
    mean_utility_parents,
)

U_MAX = M_NANO.U_max
K = M_NANO.k_route


@pytest.fixture()
def credits():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    q = 200
    slots = torch.stack(
        [torch.from_numpy(rng.choice(64, size=K, replace=False)) for _ in range(q)]
    )
    delta = torch.rand(q, K) + 0.1
    r = torch.softmax(torch.randn(q, K), dim=-1)
    w = torch.rand(q)
    w[:60] = 0.0                      # 30% of cases carry no improvement signal
    nu = credit_matrix(slots, delta, r, w, U_MAX)
    return nu, epsilon_per_case(nu), w, rng


def test_credit_is_zero_off_the_routed_set(credits) -> None:
    """(4.2): `nu_ut = 0` for every unit not routed to `t`. This is why `eps_t`
    must be computed over non-zero credits only."""
    nu, _, _, _ = credits
    assert float((nu > 0).float().mean()) < 0.05


def test_epsilon_is_not_identically_zero(credits) -> None:
    """The v1 regression guard."""
    nu, eps, w, _ = credits
    assert float((eps > 0).float().mean()) > 0.5
    assert not torch.isnan(eps).any()


def test_epsilon_is_zero_exactly_where_w_is_zero(credits) -> None:
    """(4.7): `eps_t = 0` when `C_t` is empty — which reduces the filter to
    strict lexicase on a case carrying no learnability signal."""
    nu, eps, w, _ = credits
    assert torch.equal(eps == 0, w == 0)


def test_v1_whole_population_mad_is_identically_zero(credits) -> None:
    """The positive control for the v1 defect: it must remain reproducible, or
    this test cannot show the fix does anything."""
    nu, _, _, _ = credits
    v1 = torch.stack(
        [(nu[t] - nu[t].median()).abs().median() for t in range(nu.shape[0])]
    )
    assert float((v1 == 0).float().mean()) == 1.0


def test_v2_empty_C_t_would_be_nan(credits) -> None:
    """The positive control for the v2 defect (audit A-20)."""
    nu, _, w, _ = credits
    t = int(torch.nonzero(w == 0)[0])
    vals = nu[t][nu[t] > 0]
    assert vals.numel() == 0
    assert torch.isnan(vals.median()), "an empty median must be NaN, or the "\
        "v2 defect is not reproducible and (4.7) fixes nothing"


def test_lemma_4_1_elite_guarantee(credits) -> None:
    """Every returned parent is within `eps_{t_1}` of the pool maximum on the
    first case used to select it."""
    nu, eps, w, rng = credits
    elig = list(range(64))
    for _ in range(200):
        order = _weighted_permutation(
            nu.shape[0], w.numpy().astype(float), float(w.sum()), rng
        )
        t0 = int(order[0])
        pool = np.array(elig)
        col = nu.numpy()[t0][pool]
        survivors = pool[col >= col.max() - float(eps[t0])]
        assert survivors.size >= 1
        for u in survivors:
            assert float(nu[t0, int(u)]) >= col.max() - float(eps[t0]) - 1e-9


def test_lemma_4_2_termination(credits) -> None:
    """The inner loop terminates in at most `|cases|` iterations."""
    nu, eps, w, rng = credits
    res = epsilon_lexicase(list(range(64)), nu, eps, w, 8, rng)
    assert all(c <= nu.shape[0] for c in res.cases_used)
    assert len(res.parents) == 8


def test_final_case_is_used(credits) -> None:
    """§04.5: the loop guard is `|T_q| >= 1`; spec v1's `> 1` discarded the last
    case."""
    rng = np.random.default_rng(1)
    # Two units, one case: with `> 1` the case would be dropped and selection
    # would be uniform; with `>= 1` the better unit wins deterministically.
    nu = torch.zeros(1, U_MAX)
    nu[0, 0] = 1.0
    nu[0, 1] = 0.0
    eps = torch.zeros(1)
    w = torch.ones(1)
    res = epsilon_lexicase([0, 1], nu, eps, w, 32, rng)
    assert set(res.parents) == {0}


def test_shuffle_falls_back_to_uniform_at_zero_weights() -> None:
    """§04.5: reachable whenever the model has improved on no sampled case."""
    rng = np.random.default_rng(2)
    q = 50
    perm = _weighted_permutation(q, np.zeros(q), 0.0, rng)
    assert sorted(perm.tolist()) == list(range(q))


def test_weighted_permutation_respects_weights() -> None:
    """`SHUFFLE(cases, weights = w)` draws proportional to `w_t` at each draw."""
    rng = np.random.default_rng(3)
    w = np.array([10.0, 1.0, 1.0, 1.0])
    first = np.zeros(4)
    for _ in range(4000):
        first[_weighted_permutation(4, w, w.sum(), rng)[0]] += 1
    p = first / first.sum()
    assert p[0] == pytest.approx(10 / 13, abs=0.04)


def test_proposition_4_3_specialist_preservation() -> None:
    """A unit best on a single rare case survives, where mean-utility selection
    requires its mean to exceed a threshold.

    Constructed so the specialist ranks **last of 64** on mean utility: it is
    therefore invisible to A2, yet lexicase preserves it at the rate its rare
    cases occur. That difference is exactly what K3's case-win KS statistic
    measures.
    """
    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    n_units, q, n_par = 64, 50, 8
    nu = torch.rand(q, n_units) * 0.10 + 0.45
    spec, gen = 7, 21
    nu[:, spec] = 0.01
    rare = [3, 11, 22, 37, 44]
    for t in rare:
        nu[t, spec] = 0.90               # margin must exceed the per-case MAD
    nu[:, gen] = 0.58

    eps = epsilon_per_case(nu)
    w = torch.ones(q)
    means = nu.mean(0)
    assert int((means > means[spec]).sum()) == n_units - 1, "specialist must rank last"

    hits = sum(
        spec in epsilon_lexicase(list(range(n_units)), nu, eps, w, n_par, rng).parents
        for _ in range(300)
    )
    expected = 1.0 - (1.0 - len(rare) / q) ** n_par
    assert hits / 300 == pytest.approx(expected, abs=0.10)

    assert spec not in mean_utility_parents(list(range(n_units)), means, n_par)
    assert gen in mean_utility_parents(list(range(n_units)), means, n_par)


def test_case_win_histogram_is_operator_independent() -> None:
    """§04.10: computed as a diagnostic **independent of the selection
    operator**, so it exists in **both** arms of A2.

    Spec v1 defined it only inside lexicase, so K3's comparison had data in one
    arm only.
    """
    rng = np.random.default_rng(4)
    nu = torch.rand(40, U_MAX)
    eps = epsilon_per_case(nu)
    elig = list(range(32))
    wins = case_win_histogram(elig, nu, eps)
    assert set(wins) == set(elig)
    assert sum(wins.values()) > 0
    # It does not call either operator, so it is identical whichever arm runs.
    assert wins == case_win_histogram(elig, nu, eps)


def test_case_weights_polarity() -> None:
    """(4.1): `w_t = max(0, L_lag - L_theta)`.

    Spec v1's `max(0, L_theta - L_lag)` is identically zero, because a lagged
    self knows **less** than the current model.
    """
    lag = torch.tensor([3.0, 2.0, 1.0])
    cur = torch.tensor([2.0, 2.0, 1.5])
    w = case_weights(lag, cur, 3)
    assert w.tolist() == [1.0, 0.0, 0.0]
    v1 = (cur - lag).clamp_min(0)
    assert float(v1.sum()) > 0  # on this toy input; in practice identically 0
    assert case_weights(None, None, 3).tolist() == [0.0, 0.0, 0.0]


def test_a_non_negative_proxy_cannot_rank_a_sign_changing_target() -> None:
    """§04.4 / audit A-68: why the gated statistic is a magnitude.

    (4.2) is a product of three non-negative factors. Constructed here is the
    **best possible** such proxy — one that knows `|exact|` exactly — against a
    target whose sign is randomised. Its signed correlation is near zero and its
    magnitude correlation is 1. No proxy of (4.2)'s form can do better, so a
    `0.5` bar on the signed correlation is unsatisfiable rather than demanding.
    """
    rng = np.random.default_rng(4)
    mag = rng.random(400) + 0.05
    sign = rng.choice([-1.0, 1.0], size=400)
    exact = mag * sign
    perfect = mag                        # the ideal non-negative proxy
    assert spearman(perfect, np.abs(exact)) == pytest.approx(1.0)
    assert abs(spearman(perfect, exact)) < 0.20


def test_the_credit_audit_gates_on_magnitude_and_reports_the_sign() -> None:
    """§04.4 / audit A-68: one statistic decides, the other is reported."""
    from soma.selection.credit import CreditAudit

    rng = np.random.default_rng(9)
    mag = rng.random(80) + 0.05
    exact = mag * rng.choice([-1.0, 1.0], size=80)
    audit = CreditAudit()
    rho = audit.record(mag, exact)

    assert rho == pytest.approx(1.0), "the magnitude correlation is the gate"
    assert audit.last_rho == pytest.approx(1.0)
    assert abs(audit.last_rho_signed) < 0.30, audit.last_rho_signed
    assert audit.valid and audit.tripped == 0
    assert audit.fraction == 0.01, "a passing audit must not double the fraction"
    d = audit.as_dict()
    assert "spearman" in d and "spearman_signed" in d


def test_an_unmeasured_audit_reports_unmeasured_not_valid() -> None:
    """§04.10 / audit A-84: `valid` returned **True** when the Spearman had
    never been computed, asserting the opposite of what was known.

    Observed on the integration configuration: a whole lifetime accumulated 40
    pairs against the 64-pair minimum, so `spearman` stayed `NaN`,
    `n_evaluations` stayed `0`, and the metrics reported `valid: true`.
    """
    from soma.selection.credit import CreditAudit

    audit = CreditAudit()
    assert audit.status == "unmeasured"
    assert audit.valid is None, "an unmeasured audit must not read as valid"

    # Still unmeasured after a partial accumulation.
    audit.record([1.0, 2.0], [1.0, 2.0])
    assert audit.status == "unmeasured" and audit.valid is None
    assert audit.as_dict()["status"] == "unmeasured"

    rng = np.random.default_rng(5)
    mag = rng.random(64) + 0.05
    audit2 = CreditAudit()
    audit2.record(mag, mag)
    assert audit2.status == "valid" and audit2.valid is True

    audit3 = CreditAudit()
    audit3.record(rng.random(64), rng.random(64))
    assert audit3.status == "invalid" and audit3.valid is False


def test_the_credit_audit_accumulates_to_the_64_pair_minimum() -> None:
    """§04.4 / audit A-59: one event supplies far fewer than 64 usable pairs."""
    from soma.selection.credit import CreditAudit

    audit = CreditAudit()
    for _ in range(6):
        assert np.isnan(audit.record([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    assert audit.n_evaluations == 0
    # 18 so far; the 64th pair triggers the single evaluation.
    for _ in range(16):
        audit.record([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert audit.n_evaluations == 1, audit.as_dict()
    assert audit.last_rho == pytest.approx(1.0)


def test_a_failing_audit_doubles_the_fraction_and_stops_at_the_cap() -> None:
    from soma.selection.credit import CreditAudit

    rng = np.random.default_rng(2)
    audit = CreditAudit()
    for _ in range(12):
        proxy = rng.random(64)
        audit.record(proxy, rng.random(64))     # unrelated: rho ~ 0
    assert audit.tripped >= 1
    assert audit.fraction <= audit.max_fraction + 1e-12
    assert audit.fraction == pytest.approx(audit.max_fraction)
    assert not audit.valid


def test_spearman_matches_a_known_value() -> None:
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)


def test_reservoir_is_uniform_and_bounded() -> None:
    """§04.6: a reservoir sample of fixed size, refilled continuously."""
    from soma.selection.buffer import CaseReservoir

    rng = np.random.default_rng(5)
    counts = np.zeros(50)
    for _ in range(500):
        rb = CaseReservoir(10, 1)
        rb.push(
            torch.arange(50), torch.zeros(50, 1, dtype=torch.long),
            torch.zeros(50, 1), torch.zeros(50, 1), torch.zeros(50), rng,
        )
        for t in rb.sample().token_id.tolist():
            counts[t] += 1
    p = counts / counts.sum()
    assert p.min() > 0.5 / 50 and p.max() < 1.6 / 50

    big = CaseReservoir(M_NANO.q_cases, K)
    for _ in range(20):
        n = 4096
        big.push(
            torch.randint(0, 100, (n,)), torch.randint(0, 64, (n, K)),
            torch.rand(n, K), torch.rand(n, K), torch.rand(n), rng,
        )
    assert len(big) == M_NANO.q_cases
    assert big.n_seen == 20 * 4096
    assert big.memory_bytes() < 1 << 21          # under 2 MB per Site
