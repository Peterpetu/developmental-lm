"""Does the genome carry the function? The instrument that would have caught SOMA.

SPEC: 01 §1.6
SPEC: 06 §6.1
SPEC: 06 §6.2
SPEC: 06 §6.3
SPEC: 06 §6.4
SPEC: 06 §6.5

Three statistics, for two different questions (§06.1):

``heritability``   **Q1, architectural.** Do different genomes, given full
                   lifetimes, reach different outcomes? Informative at
                   generation 0, before any evolution has happened, which is
                   what makes it the gate. `KG-HERIT`.
``genome_share``   the cheap co-adapted probe that produced §00's table.
                   Confounded; a tripwire, never evidence of success.
``birth_share``    held-out CE with no lifetime learning at all -- the PNAS
                   paper's own headline quantity.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional, Sequence

import numpy as np

from ..utils.stats import BcaResult, bca_bootstrap

__all__ = ["HeritabilityResult", "heritability", "genome_share", "birth_share",
           "share_retrain", "predicted_response", "KG_HERIT_POINT",
           "KG_HERIT_LOWER", "KG_SHARE_FLOOR", "TRUNCATION_INTENSITY"]

# SPEC: 06 §6.2 -- the gate.
KG_HERIT_POINT = 0.25
KG_HERIT_LOWER = 0.10
# SPEC: 01 §1.6, SPEC: 06 §6.3.1 -- `KG-SHARE` fires below this.
KG_SHARE_FLOOR = 0.50
# Selection intensity `i` for truncation at the top quarter: `i = phi(z_p)/p`
# with `p = 1/4`. Falconer & Mackay, 4th ed., Table 11.1.
TRUNCATION_INTENSITY = 1.271


@dataclasses.dataclass
class HeritabilityResult:
    """SPEC: 06 §6.2."""

    h2: float
    ci_low: float
    ci_high: float
    V_G: float
    V_E: float
    sigma_P: float
    n_genomes: int
    n_seeds: int
    means: list[float]

    @property
    def passes(self) -> bool:
        return self.h2 >= KG_HERIT_POINT and self.ci_low >= KG_HERIT_LOWER

    def report(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"h2 = {self.h2:.3f}  95% BCa [{self.ci_low:.3f}, {self.ci_high:.3f}]"
            f"   V_G {self.V_G:.5f}  V_E {self.V_E:.5f}  sigma_P {self.sigma_P:.4f}"
            f"   ({self.n_genomes} genomes x {self.n_seeds} seeds)"
            f"\nKG-HERIT [h2 >= {KG_HERIT_POINT}, lower >= {KG_HERIT_LOWER}]: {verdict}"
            f"\npredicted response R = {predicted_response(self):.4f} nats/generation"
            f"  (breeder's equation, truncation at the top quarter)"
        )


def _h2_from(P: np.ndarray) -> tuple[float, float, float]:
    """`(h2, V_G, V_E)` by one-way random-effects ANOVA. SPEC: 06 §6.2.

    `P` is `(n_genomes, n_seeds)` of held-out CE. `V_G` is clamped at zero:
    `MSB < MSW` is a legitimate finite-sample outcome when the true `V_G` is
    zero, and a negative variance component is not a quantity to report.
    """
    n_G, n_S = P.shape
    if n_G < 2 or n_S < 2:
        raise ValueError(f"need >= 2 genomes and >= 2 seeds, got {n_G} x {n_S}")
    gm = P.mean(axis=1)
    msb = n_S * ((gm - P.mean()) ** 2).sum() / (n_G - 1)
    msw = ((P - gm[:, None]) ** 2).sum() / (n_G * (n_S - 1))
    V_G = max((msb - msw) / n_S, 0.0)
    return (V_G / (V_G + msw) if V_G + msw > 0 else 0.0), float(V_G), float(msw)


def heritability(P: Sequence[Sequence[float]], *, n_resamples: int = 4000,
                 alpha: float = 0.05, seed: int = 0) -> HeritabilityResult:
    """Broad-sense heritability of held-out CE, with a BCa interval.

    SPEC: 06 §6.2. The bootstrap resamples **genomes with their complete seed
    sets**, preserving the nesting: resampling individual lifetimes would treat
    the seeds within a genome as independent observations of different genomes
    and would inflate `V_G` by exactly the quantity being estimated.
    """
    arr = np.asarray(P, dtype=float)
    h2, V_G, V_E = _h2_from(arr)
    units = [arr[g] for g in range(arr.shape[0])]

    def stat(rows: Sequence[np.ndarray]) -> float:
        return _h2_from(np.stack(list(rows)))[0]

    bca: BcaResult = bca_bootstrap(
        units, stat, n_resamples=n_resamples, alpha=alpha,
        rng=np.random.default_rng(seed), alternative="greater",
    )
    return HeritabilityResult(
        h2=h2, ci_low=float(bca.ci_low), ci_high=float(bca.ci_high),
        V_G=V_G, V_E=V_E, sigma_P=float(math.sqrt(V_G + V_E)),
        n_genomes=arr.shape[0], n_seeds=arr.shape[1],
        means=arr.mean(axis=1).tolist(),
    )


def predicted_response(h: HeritabilityResult) -> float:
    """`R = i h2 sigma_P`. SPEC: 06 §6.2 -- registered before the run.

    The breeder's equation is not a diagnostic that correlates with
    evolvability; it is, to first order, its constant of proportionality. §09
    compares this prediction against the observed generation-curve slope.
    """
    return TRUNCATION_INTENSITY * h.h2 * h.sigma_P


def genome_share(L_star: float, L_gamma_rand: Sequence[float],
                 L_z_rand: Sequence[float]) -> dict:
    """The co-adapted randomise-and-measure probe. SPEC: 06 §6.3.

    Confounded by co-adaptation -- `Z*` was trained against `Gamma*`, so
    damaging either partner hurts -- and reported as a tripwire rather than as
    evidence. SOMA's value, computed this way over its own germline/soma split,
    was **0.024**; a collapse toward zero is conclusive, a healthy value is not.
    """
    d_g = float(np.mean(L_gamma_rand)) - L_star
    d_z = float(np.mean(L_z_rand)) - L_star
    tot = d_g + d_z
    return {
        "share": float(d_g / tot) if tot > 0 else float("nan"),
        "delta_gamma": d_g, "delta_z": d_z,
        "L_star": L_star,
        "L_gamma_rand": float(np.mean(L_gamma_rand)),
        "L_z_rand": float(np.mean(L_z_rand)),
        "soma_reference": 0.024,
    }


def birth_share(L_birth: float, L_null: float, L_star: float) -> float:
    """`(L_null - L_birth) / (L_null - L_star)`. SPEC: 06 §6.4.

    The direct analogue of the paper's 94%-at-birth MNIST result. ~0 at
    generation 0 and *should* be: a random genome knows nothing. Its trajectory
    across generations is the quantity to watch.
    """
    den = L_null - L_star
    return float((L_null - L_birth) / den) if den > 0 else float("nan")


def share_retrain(L_star: float, L_random_trained: float,
                  L_birth: float) -> dict:
    """`share_retrain`: the genome's share, with the confound removed.

    SPEC: 01 §1.6, SPEC: 06 §6.3.1. `KG-SHARE` fires below `KG_SHARE_FLOOR`.

    **[NORMATIVE — this statistic was named and never defined.]** §01.6 requires
    it at every certification point and §01.11 lists its symbol, but §06 defined
    only the co-adapted probe of §6.3, and §09.2's kill-criteria table omitted
    `KG-SHARE` altogether. §06.1 explains why: `share_retrain` cannot answer Q1,
    because at generation 0 the evolved genome *is* a random genome and the
    statistic is identically zero by construction -- which is the very defect
    (`BirthScore = 0` by construction) that made SOMA's decisive experiment
    measure nothing. That argument disqualifies it as the **gate on the smoke
    run**; it does not disqualify it as the standing report §01.6 asks for once
    evolution has actually run. Both are now true of it, and §09.2 carries the
    criterion.

    The definition, stated here because nothing else states it. Three losses,
    all held-out CE in nats on the same split, all at the same budget:

    ``L_star``            the evolved genome, after a full lifetime.
    ``L_random_trained``  a **random** genome, after a full lifetime -- the
                          soma re-trained from scratch against it, which is
                          what removes §6.3's co-adaptation confound: the
                          partner is not damaged, it is given a fresh chance.
    ``L_birth``           the evolved genome with **no lifetime at all**
                          (`Z` at its initialisation, `eps = 0`), §6.4's
                          at-birth quantity.

    ::

        Delta_Gamma = L_random_trained - L_star      what the genome is worth
        Delta_Z     = L_birth          - L_star      what the lifetime is worth
        share       = Delta_Gamma / (Delta_Gamma + Delta_Z)

    The asymmetry between the two terms is real and is not a defect in the
    estimator: `Z` is retrained *within* a lifetime and `Gamma` is retrained
    only by *evolution*, over generations, so "re-train the other from scratch"
    can be executed for the soma and cannot be for the germline inside one
    measurement. What the numerator therefore asks is the question that matters
    -- how much of the attainable learning survives replacing the genome, given
    the soma always gets to adapt -- and that is precisely the question SOMA
    answered with 0.024 and never noticed.

    A negative `Delta` means the reference did *better* than the evolved genome,
    which at generation 0 is a coin flip and later is a finding; `share` is then
    reported as-is rather than clamped, because clamping would hide it.
    """
    d_g = float(L_random_trained) - float(L_star)
    d_z = float(L_birth) - float(L_star)
    tot = d_g + d_z
    share = float(d_g / tot) if tot != 0 else float("nan")
    return {
        "share_retrain": share,
        "delta_gamma": d_g,
        "delta_z": d_z,
        "L_star": float(L_star),
        "L_random_trained": float(L_random_trained),
        "L_birth": float(L_birth),
        "floor": KG_SHARE_FLOOR,
        "KG_SHARE": "FIRES" if (share == share and share < KG_SHARE_FLOOR)
                    else "not fired",
    }
