"""Statistical machinery: Mann-Kendall, BCa bootstrap, TOST, effective rank.

SPEC: 06 §6.4
SPEC: 07 §7.5
SPEC: 08 §8.5
SPEC: 03 §3.4.1

Every test the specification uses to decide something lives here, fully
specified, so that a decision can be reproduced from the logged inputs alone.
Three of them exist because the specification's own audit found that "unchanged"
and "no trend" were being asserted with no test behind them at all:

* `mann_kendall` decides `erank` decline (K8) and certification novelty (K5);
* `bca_bootstrap_slope` decides the generation curve (K1);
* `tost` decides every equivalence claim (K2, K3, K4, K10 and A1-A12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MannKendallResult",
    "mann_kendall",
    "benjamini_hochberg",
    "TostResult",
    "tost",
    "BcaResult",
    "bca_bootstrap",
    "bca_bootstrap_slope",
    "effective_rank",
    "median_absolute_deviation",
    "ks_statistic",
    "pass_at_k",
    "scaled_accuracy",
    "log_slope",
]


# ---------------------------------------------------------------------------
# Mann-Kendall
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MannKendallResult:
    S: int
    n: int
    p_value: float
    method: str
    """`"exact"` or `"normal"` — which null distribution was used."""
    tau: float
    slope: float
    """Theil-Sen slope, in units of `x` per index step."""


def _inversion_distribution(n: int) -> np.ndarray:
    """Exact distribution of the inversion count over permutations of `n` items.

    The generating function is `prod_{k=1..n} (1 + q + ... + q^{k-1})` — the
    Mahonian numbers. Returned as counts indexed by inversion number, summing to
    `n!`.
    """
    poly = np.array([1.0])
    for k in range(1, n + 1):
        block = np.ones(k)
        poly = np.convolve(poly, block)
    return poly


def mann_kendall(
    x: Sequence[float],
    *,
    alternative: str = "increasing",
) -> MannKendallResult:
    """One-sided Mann-Kendall trend test, fully specified per §06.4.

    SPEC: 06 §6.4

    * **original** MK statistic `S = sum_{i<j} sign(x_j - x_i)`;
    * **exact** null distribution when `n <= 10` *and* there are no ties;
    * otherwise the **normal approximation with continuity correction** and the
      **tie-corrected** variance;
    * **one-sided**.

    The two branches are as the specification states them, with one point made
    explicit that the specification leaves implicit: the exact null is the
    distribution of `S` over permutations of *distinct* values, so it is not the
    null distribution when ties are present. With ties at `n <= 10` this
    function therefore uses the tie-corrected normal approximation and reports
    `method = "normal"`. Silently applying the untied exact null to tied data
    would understate the p-value.

    With `J_ring = 8` the normal approximation is invalid for the novelty test,
    so the exact branch is what actually runs there (§06.4).
    """
    if alternative not in ("increasing", "decreasing"):
        raise ValueError("alternative must be 'increasing' or 'decreasing'")
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("mann_kendall expects a 1-D sequence")
    n = a.size
    if n < 3:
        raise ValueError(f"Mann-Kendall needs at least 3 points; got {n}")

    # S and tau
    diff = a[None, :] - a[:, None]
    iu = np.triu_indices(n, k=1)
    signs = np.sign(diff[iu])
    S = int(np.sum(signs))

    # Ties
    _, counts = np.unique(a, return_counts=True)
    ties = counts[counts > 1]
    n_pairs = n * (n - 1) / 2.0
    tie_pairs = float(np.sum(ties * (ties - 1) / 2.0))
    tau = S / math.sqrt((n_pairs - tie_pairs) * n_pairs) if n_pairs > tie_pairs else 0.0

    # Theil-Sen slope, reported alongside because K8's criterion is a slope
    # threshold, not merely a sign (§10 K8).
    idx = np.arange(n, dtype=float)
    di = idx[None, :] - idx[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = diff[iu] / di[iu]
    slope = float(np.median(slopes)) if slopes.size else 0.0

    has_ties = tie_pairs > 0
    if n <= 10 and not has_ties:
        # Exact: S = n(n-1)/2 - 2 * inversions
        counts_inv = _inversion_distribution(n)
        total = counts_inv.sum()
        max_pairs = n * (n - 1) // 2
        s_values = max_pairs - 2 * np.arange(counts_inv.size)
        if alternative == "increasing":
            mask = s_values >= S
        else:
            mask = s_values <= S
        p = float(counts_inv[mask].sum() / total)
        method = "exact"
    else:
        var = (n * (n - 1) * (2 * n + 5)) / 18.0
        var -= float(np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
        if var <= 0:
            return MannKendallResult(S, n, 1.0, "degenerate", tau, slope)
        if S > 0:
            z = (S - 1) / math.sqrt(var)
        elif S < 0:
            z = (S + 1) / math.sqrt(var)
        else:
            z = 0.0
        # Normal survival function without importing scipy, so this module has
        # no heavyweight dependency: Phi(-|z|) via erfc.
        if alternative == "increasing":
            p = 0.5 * math.erfc(z / math.sqrt(2.0))
        else:
            p = 0.5 * math.erfc(-z / math.sqrt(2.0))
        method = "normal"

    return MannKendallResult(S, n, min(1.0, max(0.0, p)), method, tau, slope)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up; returns a boolean rejection mask.

    SPEC: 06 §6.4

    Applied across rows of the certification matrix. Controls the false
    discovery rate at `alpha`, which is the right family-wise notion here: the
    novelty fraction is a *proportion* of rows that trend, so controlling the
    expected proportion of false rejections among them is exactly the guarantee
    the reported statistic needs.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        reject[order[: k + 1]] = True
    return reject


# ---------------------------------------------------------------------------
# TOST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TostResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    """The `1 - 2*alpha` (i.e. 90% at alpha = 0.05) confidence interval."""
    p_lower: float
    p_upper: float
    equivalent: bool
    """Both one-sided tests reject: the difference is within `(-delta, +delta)`."""
    different: bool
    """The ordinary two-sided test rejects."""
    verdict: str
    """One of `"equivalent"`, `"different"`, `"inconclusive"`."""
    n: int
    delta: float


def _student_t_sf(t: float, df: int) -> float:
    """Upper tail of Student's t, via the regularised incomplete beta."""
    if df <= 0:
        return float("nan")
    from scipy import stats as _sp  # local import: only the tests need scipy

    return float(_sp.t.sf(t, df))


def _student_t_ppf(q: float, df: int) -> float:
    from scipy import stats as _sp

    return float(_sp.t.ppf(q, df))


def tost(
    a: Sequence[float],
    b: Sequence[float],
    delta: float,
    *,
    alpha: float = 0.05,
    paired: bool = True,
) -> TostResult:
    """Two one-sided tests for equivalence within `+/- delta`.

    SPEC: 08 §8.5

    "Unchanged" means both one-sided tests reject at `alpha`, i.e. the
    `1 - 2*alpha` confidence interval of the difference lies entirely within
    `(-delta, +delta)`. "Changed" means the ordinary two-sided test rejects.
    Neither rejecting is **inconclusive**, which §08.5 names as an honest third
    outcome rather than folding it into "unchanged".

    `paired = True` compares seed-matched runs, which is the design §08.6
    mandates (every ablation is compared at matched tokens against the same
    seeds), and is materially more powerful than the unpaired form because it
    removes the seed-to-seed variance that dominates at `n_seed = 5`.

    Raises if `delta <= 0`: a zero margin makes the equivalence region empty and
    the test undecidable in that direction. This is not a hypothetical — spec v2
    defined the noise floor over identical seeds, making every
    `delta = 2 * noise_floor` exactly zero (audit A-12), and the failure was
    silent.
    """
    if delta <= 0:
        raise ValueError(
            f"TOST margin must be positive; got {delta!r}. A zero margin makes "
            f"the equivalence region empty and the criterion undecidable "
            f"(see AUDIT A-12)."
        )
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if paired:
        if x.shape != y.shape:
            raise ValueError(
                f"paired TOST needs matched samples; got {x.shape} and {y.shape}"
            )
        d = x - y
        n = d.size
        if n < 2:
            raise ValueError("TOST needs at least 2 paired observations")
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / math.sqrt(n))
        df = n - 1
    else:
        n1, n2 = x.size, y.size
        if n1 < 2 or n2 < 2:
            raise ValueError("TOST needs at least 2 observations per group")
        mean = float(np.mean(x) - np.mean(y))
        v1 = float(np.var(x, ddof=1))
        v2 = float(np.var(y, ddof=1))
        se = math.sqrt(v1 / n1 + v2 / n2)
        # Welch-Satterthwaite
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = int(max(1, round(num / den))) if den > 0 else n1 + n2 - 2
        n = min(n1, n2)

    if se == 0.0:
        # Zero variance: every observation identical. Equivalence holds iff the
        # point estimate is inside the margin; this is reported rather than
        # producing a division by zero, but it also means the noise floor was
        # measured on a degenerate sample (§07.7 A-12).
        equivalent = abs(mean) < delta
        return TostResult(
            mean, mean, mean, 0.0 if equivalent else 1.0, 0.0 if equivalent else 1.0,
            equivalent, not equivalent and mean != 0.0,
            "equivalent" if equivalent else "different", n, delta,
        )

    t_lower = (mean - (-delta)) / se
    t_upper = (mean - delta) / se
    p_lower = _student_t_sf(t_lower, df)      # H0: diff <= -delta
    p_upper = 1.0 - _student_t_sf(t_upper, df)  # H0: diff >= +delta
    equivalent = (p_lower < alpha) and (p_upper < alpha)

    t_two = mean / se
    p_two = 2.0 * _student_t_sf(abs(t_two), df)
    different = p_two < alpha

    crit = _student_t_ppf(1.0 - alpha, df)
    ci_low, ci_high = mean - crit * se, mean + crit * se

    if equivalent and not different:
        verdict = "equivalent"
    elif different and not equivalent:
        verdict = "different"
    elif different and equivalent:
        # Possible with a wide margin: significantly non-zero but practically
        # equivalent. Reported as equivalent, since delta encodes what counts.
        verdict = "equivalent"
    else:
        verdict = "inconclusive"

    return TostResult(
        mean, ci_low, ci_high, p_lower, p_upper, equivalent, different, verdict, n, delta
    )


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BcaResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_one_sided: float
    """P(statistic <= 0) under the bootstrap, for the one-sided `b > 0` test."""
    significant: bool
    n_resamples: int
    n_units: int


def _normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _normal_ppf(p: float) -> float:
    from scipy import stats as _sp

    return float(_sp.norm.ppf(p))


def bca_bootstrap(
    units: Sequence,
    statistic: Callable[[Sequence], float],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    alternative: str = "greater",
) -> BcaResult:
    """Bias-corrected and accelerated bootstrap over `units`.

    SPEC: 07 §7.5

    §07.5 resamples **seeds**, not lifetimes, so `units` is a sequence of
    per-seed records and `statistic` maps a resample of them to the scalar of
    interest. The distinction matters: lifetimes within a seed share a data
    order and an initialisation, so resampling them treats correlated
    observations as independent and understates the interval.

    Spec v1 specified a bootstrap over **3** seeds, which admits 27 distinct
    resamples and a minimum achievable `p = 1/27 = 0.037`; §07.5 requires
    `n_seed >= 5`. This function raises below 5 rather than returning a p-value
    that cannot reach the requested alpha.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    n = len(units)
    if n < 5:
        raise ValueError(
            f"§07.5 requires n_seed >= 5; got {n}. With {n} units the smallest "
            f"attainable one-sided p is {1.0 / (n ** n):.4g} only if every "
            f"resample agrees, and the BCa acceleration is unstable."
        )
    rng = rng if rng is not None else np.random.default_rng(0)

    theta_hat = float(statistic(units))

    idx = rng.integers(0, n, size=(n_resamples, n))
    boots = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        boots[i] = statistic([units[j] for j in idx[i]])

    # Bias correction
    prop = float(np.mean(boots < theta_hat))
    prop = min(max(prop, 1.0 / (2 * n_resamples)), 1.0 - 1.0 / (2 * n_resamples))
    z0 = _normal_ppf(prop)

    # Acceleration from the jackknife
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = statistic([units[j] for j in range(n) if j != i])
    jbar = float(np.mean(jack))
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * (float(np.sum((jbar - jack) ** 2)) ** 1.5)
    accel = num / den if den != 0 else 0.0

    def _adjust(q: float) -> float:
        zq = _normal_ppf(q)
        denom = 1.0 - accel * (z0 + zq)
        if denom == 0:
            return q
        return _normal_cdf(z0 + (z0 + zq) / denom)

    if alternative == "two-sided":
        lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    elif alternative == "greater":
        lo_q, hi_q = alpha, 1.0
    else:
        lo_q, hi_q = 0.0, 1.0 - alpha

    a_lo = _adjust(lo_q) if lo_q > 0 else 0.0
    a_hi = _adjust(hi_q) if hi_q < 1 else 1.0
    ci_low = float(np.quantile(boots, min(max(a_lo, 0.0), 1.0)))
    ci_high = float(np.quantile(boots, min(max(a_hi, 0.0), 1.0)))

    p_one = float(np.mean(boots <= 0.0))
    p_one = max(p_one, 1.0 / (n_resamples + 1))
    if alternative == "less":
        p_one = float(np.mean(boots >= 0.0))
        p_one = max(p_one, 1.0 / (n_resamples + 1))

    if alternative == "greater":
        significant = ci_low > 0.0
    elif alternative == "less":
        significant = ci_high < 0.0
    else:
        significant = ci_low > 0.0 or ci_high < 0.0

    return BcaResult(theta_hat, ci_low, ci_high, p_one, significant, n_resamples, n)


def bca_bootstrap_slope(
    per_seed_curves: Sequence[Sequence[Tuple[int, float]]],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> BcaResult:
    """Test `b > 0` in `y(g) = a + b log(1 + g)`, resampling seeds.

    SPEC: 07 §7.5

    `per_seed_curves[s]` is that seed's `[(g, y), ...]` across generations. The
    statistic is the ordinary-least-squares `b` fitted to the *pooled* points of
    the resampled seeds, which is what §07.5's "fit `y(g) = a + b log(1+g)`"
    means when several seeds are present.
    """

    def stat(curves: Sequence[Sequence[Tuple[int, float]]]) -> float:
        xs: List[float] = []
        ys: List[float] = []
        for curve in curves:
            for g, y in curve:
                xs.append(math.log1p(g))
                ys.append(float(y))
        if len(xs) < 2:
            return 0.0
        X = np.asarray(xs)
        Y = np.asarray(ys)
        xm, ym = X.mean(), Y.mean()
        denom = float(np.sum((X - xm) ** 2))
        if denom == 0:
            return 0.0
        return float(np.sum((X - xm) * (Y - ym)) / denom)

    return bca_bootstrap(
        list(per_seed_curves),
        stat,
        n_resamples=n_resamples,
        alpha=alpha,
        rng=rng,
        alternative="greater",
    )


# ---------------------------------------------------------------------------
# Small statistics used throughout
# ---------------------------------------------------------------------------


def effective_rank(singular_values: Sequence[float], eps: float = 1e-12) -> float:
    """(3.9): `erank = exp(-sum_i p_i log p_i)`, `p = s / ||s||_1`.

    SPEC: 03 §3.4.1

    The Shannon entropy of the normalised spectrum, exponentiated — so a
    spectrum concentrated on one direction gives 1 and a flat spectrum of `k`
    values gives `k`. §03.4 logs it because the peer-reviewed finding is that
    networks losing plasticity show a *falling* effective rank, and K8 is the
    criterion.
    """
    s = np.asarray(singular_values, dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    p = np.clip(p, eps, None)
    return float(np.exp(-np.sum(p * np.log(p))))


def median_absolute_deviation(x: Sequence[float]) -> float:
    """MAD about the median. Returns 0.0 on an empty input.

    SPEC: 04 §4.5

    (4.7) defines `eps_t = 0` when `C_t` is empty, which is common because
    `C_t` is empty whenever `w_t = 0`. Returning 0.0 rather than NaN is what
    makes the ε-lexicase precondition `eps_t >= 0` satisfiable.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (the sup-norm of the ECDF gap).

    SPEC: 08 §8.5

    A2's primary metric is the case-win histogram compared by KS. Computed
    directly rather than via scipy so the value is reproducible without a
    version-dependent tie convention.
    """
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    if x.size == 0 or y.size == 0:
        raise ValueError("KS statistic needs non-empty samples")
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def pass_at_k(n: int, c: int, k: int) -> float:
    """(8.1): `1 - C(n-c, k) / C(n, k)`.

    SPEC: 08 §8.1.2

    The probability that at least one of `k` samples drawn without replacement
    from `n` is correct, given `c` correct. Spec v1 defined `pass@k` as the
    per-sample success rate, which is independent of `k` — under which the
    `k`-curve is flat and §08.7 rule 1's whole argument is unobservable.

    Computed in log space so that `n = 128` does not overflow the intermediate
    binomials.
    """
    if not (0 <= c <= n):
        raise ValueError(f"need 0 <= c <= n; got c={c}, n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n; got k={k}, n={n}")
    if n - c < k:
        return 1.0
    # C(n-c,k)/C(n,k) = prod_{i=0..k-1} (n-c-i)/(n-i)
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)


def scaled_accuracy(acc: float, acc_random: float) -> float:
    """`(acc - acc_random) / (1 - acc_random)` (§08.1).

    SPEC: 08 §8.1

    §08.2 forbids reporting raw accuracy for the regression suite: for parity,
    chance is 0.5, so a raw 0.5 is *no capability at all* and reads as half.
    """
    if not 0.0 <= acc_random < 1.0:
        raise ValueError(f"acc_random must lie in [0,1); got {acc_random!r}")
    return (acc - acc_random) / (1.0 - acc_random)


def log_slope(points: Sequence[Tuple[float, float]]) -> float:
    """OLS slope of `y` against `log(x)` — `early_slope` and `late_slope`.

    SPEC: 08 §8.1

    `points` are `(tokens, CE)`. Entries with `tokens <= 0` are dropped, since
    `log 0` is not a point on the curve.
    """
    xs = np.array([math.log(t) for t, _ in points if t > 0], dtype=float)
    ys = np.array([y for t, y in points if t > 0], dtype=float)
    if xs.size < 2:
        return 0.0
    xm, ym = xs.mean(), ys.mean()
    denom = float(np.sum((xs - xm) ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum((xs - xm) * (ys - ym)) / denom)
