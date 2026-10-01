"""The decomposition every screen reports, and the gates that read it.

One question: **does this mechanism's genome move the loss by more than seed
noise does?** Everything here exists to answer it in a form that cannot be
talked around -- a ratio, a power calculation, and two gates that fire on
numbers rather than on judgement.

Three specific failures in this repository's history are what each piece is for.

``sigma_e`` measured, never inherited
    GNOMON reasoned for weeks with a seed noise of 0.171 nats. That figure is a
    property of GNOMON's saturating decoder; a free-weight transformer at the
    same scale measures **0.0135**. Every conclusion drawn from the wrong number
    was drawn from a noise floor 12.7x too high.

paired contrasts
    Under common random numbers the paired difference has far smaller variance
    than either arm's spread. Using the unpaired sd when the design is paired
    throws away roughly a 3x reduction in the seeds required.

``deterministic_term_check``
    A resource price is noiseless (sigma = 0) while cross-entropy is not. Add
    them and fitness correlates ~1 with the price and ~0.2 with the loss, so the
    search optimises a noiseless function of the genome -- becoming a
    parameter-count minimiser in a language-model costume, with a monotonically
    rising archive score that looks exactly like success.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from gnomon.utils.stats import bca_bootstrap

__all__ = [
    "Contrast", "Decomposition", "pooled_within_sd", "contrast",
    "seeds_for_power", "decompose", "deterministic_term_check",
    "searchworthy",
]

#: Two-sample, two-sided, alpha=0.05, power=0.8:  2 (z_0.975 + z_0.8)^2.
_Z_TWO_SAMPLE = 2.0 * (1.959964 + 0.841621) ** 2      # 15.696
#: The paired form drops the factor of 2 -- one sample of differences.
_Z_PAIRED = (1.959964 + 0.841621) ** 2                # 7.848


@dataclasses.dataclass
class Contrast:
    """One arm against another, paired by seed wherever seeds coincide."""

    name: str
    a: str
    b: str
    delta: float                  # mean(b) - mean(a); positive means `b` is worse
    paired_delta: Optional[float]
    paired_sd: Optional[float]
    n_paired: int
    d_over_sigma_e: float
    seeds_needed: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def line(self) -> str:
        ci = ("" if self.ci_low is None
              else f"  [{self.ci_low:+.4f}, {self.ci_high:+.4f}]")
        return (f"{self.name:<44s} {self.delta:+8.4f}  "
                f"{self.d_over_sigma_e:6.2f} sigma_e  n>={self.seeds_needed:<4d}{ci}")


@dataclasses.dataclass
class Decomposition:
    """The screen's verdict on one mechanism."""

    sigma_e: float
    sigma_e_df: int
    sigma_g: float
    ratio: float
    heritability: float           # sigma_G^2 / (sigma_G^2 + sigma_e^2)
    n_arms: int
    contrasts: list[Contrast]

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    def render(self) -> str:
        out = [
            f"sigma_e (pooled within-arm, {self.sigma_e_df} df) = {self.sigma_e:.4f} nats",
            f"sigma_G (between-arm)                     = {self.sigma_g:.4f} nats",
            f"sigma_G / sigma_e                         = {self.ratio:.2f}",
            f"sigma_G^2 / (sigma_G^2 + sigma_e^2)       = {self.heritability:.3f}",
            "",
            f"{'contrast':<44s} {'delta':>8s}  {'effect':>14s}  {'seeds':<6s}",
        ]
        out += ["  " + c.line() for c in self.contrasts]
        return "\n".join(out)


def pooled_within_sd(table: Mapping[str, Sequence[float]]) -> tuple[float, int]:
    """Pooled within-arm standard deviation and its degrees of freedom.

    Pooled rather than averaged: arms may carry different seed counts, and an
    unweighted mean of per-arm sds would let a 2-seed arm count as much as an
    8-seed one.
    """
    ss, df = 0.0, 0
    for vals in table.values():
        if len(vals) < 2:
            continue
        m = statistics.fmean(vals)
        ss += sum((v - m) ** 2 for v in vals)
        df += len(vals) - 1
    if df == 0:
        raise ValueError("no arm has >= 2 seeds; sigma_e is not estimable")
    return math.sqrt(ss / df), df


def seeds_for_power(delta: float, sigma: float, *, paired: bool) -> int:
    """Seeds per arm for 80% power at alpha=0.05 to resolve `delta`."""
    if delta == 0 or not math.isfinite(delta):
        return 1 << 30
    k = _Z_PAIRED if paired else _Z_TWO_SAMPLE
    return max(1, math.ceil(k * sigma ** 2 / delta ** 2))


def contrast(table: Mapping[str, Sequence[float]],
             seeds: Mapping[str, Sequence[int]],
             a: str, b: str, *, sigma_e: float, name: Optional[str] = None,
             n_resamples: int = 10000) -> Contrast:
    """`b` against `a`, using the paired differences wherever seeds coincide."""
    va, vb = list(table[a]), list(table[b])
    delta = statistics.fmean(vb) - statistics.fmean(va)

    by_a = dict(zip(seeds[a], va))
    by_b = dict(zip(seeds[b], vb))
    shared = sorted(set(by_a) & set(by_b))
    diffs = [by_b[s] - by_a[s] for s in shared]

    lo = hi = None
    if len(diffs) >= 2:
        p_delta = statistics.fmean(diffs)
        p_sd = statistics.stdev(diffs)
        # BCa over seeds: the resampling unit is the seed, which is the unit the
        # design randomises. Resampling *observations* would treat two arms'
        # values at the same seed as independent, which under CRN they are not.
        if len(diffs) >= 3:
            res = bca_bootstrap(diffs, statistics.fmean,
                                n_resamples=n_resamples, alternative="two-sided")
            lo, hi = res.ci_low, res.ci_high
        sigma_for_power = p_sd if p_sd > 0 else sigma_e
        n_needed = seeds_for_power(p_delta, sigma_for_power, paired=True)
    else:
        p_delta = p_sd = None
        n_needed = seeds_for_power(delta, sigma_e, paired=False)

    return Contrast(
        name=name or f"{b} vs {a}", a=a, b=b, delta=delta,
        paired_delta=p_delta, paired_sd=p_sd, n_paired=len(diffs),
        d_over_sigma_e=abs(delta) / sigma_e if sigma_e > 0 else float("inf"),
        seeds_needed=n_needed, ci_low=lo, ci_high=hi)


def decompose(table: Mapping[str, Sequence[float]],
              seeds: Mapping[str, Sequence[int]],
              *, contrasts: Sequence[tuple[str, str, str]] = ()) -> Decomposition:
    """The whole verdict. `contrasts` are `(name, baseline_arm, other_arm)`."""
    sigma_e, df = pooled_within_sd(table)
    means = [statistics.fmean(v) for v in table.values()]
    sigma_g = statistics.stdev(means) if len(means) >= 2 else 0.0
    # Arm means each carry sigma_e^2/n of measurement noise, so the raw spread
    # of means overstates the genetic component. Subtract it, and floor at zero
    # rather than reporting a negative variance as if it meant something.
    n_bar = statistics.fmean([len(v) for v in table.values()])
    var_g = max(sigma_g ** 2 - sigma_e ** 2 / n_bar, 0.0)
    sigma_g_adj = math.sqrt(var_g)
    return Decomposition(
        sigma_e=sigma_e, sigma_e_df=df, sigma_g=sigma_g_adj,
        ratio=sigma_g_adj / sigma_e if sigma_e > 0 else float("inf"),
        heritability=var_g / (var_g + sigma_e ** 2) if sigma_e > 0 else 1.0,
        n_arms=len(table),
        contrasts=[contrast(table, seeds, a, b, sigma_e=sigma_e, name=nm)
                   for nm, a, b in contrasts])


def deterministic_term_check(prices: Sequence[float], lam: float,
                             dec: Decomposition) -> dict:
    """Would a priced fitness be optimising the price instead of the loss?

    `prices` are the *deterministic* term of the fitness (unique parameters,
    cache bytes, FLOPs) for the same arms `dec` was built from, before scaling
    by `lam`. If `Var(lam * price)` exceeds the genetic variance of the loss,
    the search ranks genomes by a noiseless function of themselves and the
    language model is decoration.
    """
    if len(prices) < 2:
        raise ValueError("need >= 2 arms to compare variances")
    var_price = float(np.var(np.asarray(prices, dtype=float) * lam, ddof=1))
    var_g = dec.sigma_g ** 2
    return {
        "lambda": lam,
        "var_lambda_price": var_price,
        "var_genetic_ce": var_g,
        "var_seed_noise": dec.sigma_e ** 2,
        "price_dominates": var_price > var_g,
        "ratio_price_to_genetic": (var_price / var_g) if var_g > 0 else float("inf"),
    }


def searchworthy(dec: Decomposition, *, hill_climbable: Mapping[str, bool],
                 min_sigma: float = 3.0) -> dict:
    """G-SEARCHWORTHY: does this mechanism deserve a search at all?

    An axis qualifies only if it is **both** above `min_sigma` **and** not
    solvable by hill-climbing. The distinction is the whole gate. In this
    project's own S1 measurements the operator axis is 14.8 sigma_e and entirely
    real -- and monotone, so four greedy flips find its optimum from any start.
    A 14.8-sigma effect that a hill-climb solves is a hypothesis with a known
    answer, not a search space, and building MAP-Elites over it would produce a
    third system whose evolutionary machinery contributes nothing.
    """
    rows = []
    for c in dec.contrasts:
        hc = hill_climbable.get(c.name)
        rows.append({
            "contrast": c.name,
            "sigma": c.d_over_sigma_e,
            "above_floor": c.d_over_sigma_e >= min_sigma,
            "hill_climbable": hc,
            "qualifies": bool(c.d_over_sigma_e >= min_sigma and hc is False),
        })
    return {"min_sigma": min_sigma, "axes": rows,
            "passes": any(r["qualifies"] for r in rows)}
