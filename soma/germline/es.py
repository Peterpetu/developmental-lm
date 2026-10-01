"""The phylogeny rate: single-point ES with mirrored sampling on the germline.

SPEC: 02 §2.6
SPEC: 02 §2.6.1
SPEC: 02 §2.6.2

Between lifetimes, the germline's **continuous** components (`mu`, `log_sigma`,
and the unconstrained policy reals) are updated by single-point evolution
strategies with mirrored sampling. The **discrete** components (Inventory
membership and archetype gene sets) are updated by selection and recombination,
not by ES.

**Why ES rather than a gradient, positively.** ES optimises the
perturbation-smoothed objective `E_xi[F(theta + sigma xi)]`, i.e. for parameters
*robust to perturbation* — which is canalization. No gradient method optimises
for this. ES on the germline is not a fallback; its implicit bias is the
property a genome should have (§02.1.3c).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from soma.germline.field import Field
from soma.germline.genome import Genome
from soma.germline.policy import Policy

__all__ = ["EsHyper", "EsState", "rank_normalise", "es_step", "generation_count"]


@dataclass(frozen=True)
class EsHyper:
    """ES hyperparameters (§02.6.1)."""

    P_es: int = 2
    """Population size; must be even and >= 2 (mirrored pairs)."""
    sigma_es: float = 0.02
    alpha_es: float = 0.05
    r_rank: int = 1
    """(2.1) low-rank perturbation rank. The estimator converges to the
    full-rank ES update at `O(1/r_rank)`, so `r_rank = 1` is usable and
    perturbation storage falls from `O(pq)` to `O(r(p+q))` (§02.6.1)."""

    def __post_init__(self) -> None:
        if self.P_es < 2 or self.P_es % 2 != 0:
            raise ValueError(f"P_es must be even and >= 2; got {self.P_es}")
        if self.sigma_es <= 0:
            raise ValueError("sigma_es must be positive")
        if self.r_rank < 1:
            raise ValueError("r_rank must be >= 1")


def rank_normalise(f: Sequence[float]) -> List[float]:
    """Average ranks mapped to `[-0.5, +0.5]` (§02.6.1).

    ```
    r_i <- average rank of f_i in ascending order, ties averaged
    return (r_i - 1)/(P - 1) - 0.5
    ```
    """
    P = len(f)
    if P < 2:
        raise ValueError("rank normalisation needs at least 2 members")
    order = sorted(range(P), key=lambda i: f[i])
    ranks = [0.0] * P
    i = 0
    while i < P:
        j = i
        while j + 1 < P and f[order[j + 1]] == f[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return [(r - 1.0) / (P - 1) - 0.5 for r in ranks]


def centred_difference(
    f_plus: float, f_minus: float, sigma_f: float
) -> Tuple[float, float]:
    """The `P_es = 2` update weights (§02.6.1).

    SPEC: 02 §2.6.1

    With two members `RANK_NORMALISE` returns exactly `{-0.5, +0.5}`, so the
    update reduces to `M <- M +/- (alpha_es/2) E` — a fixed-magnitude step whose
    only information is one sign bit, which is precisely what spec v2 criticised
    v1 for. Rank normalisation discards the magnitude that mirroring exists to
    provide (audit A-39). Therefore at `P_es = 2`:

        f_hat_1 = -f_hat_2 = clip( (f_+ - f_-) / (2 sigma_f), -1, +1 )
    """
    sigma_f = max(float(sigma_f), 1e-12)
    w = (f_plus - f_minus) / (2.0 * sigma_f)
    w = max(-1.0, min(1.0, w))
    return w, -w


@dataclass
class EsState:
    """Mutable ES bookkeeping carried across generations.

    `sigma_f` is a running standard deviation of the fitness over the last 8
    generations, initialised to 1 (§02.6.1). It is what keeps the `P_es = 2`
    estimator scale-free while retaining magnitude.
    """

    hyper: EsHyper = field(default_factory=EsHyper)
    fitness_history: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    G_gen: int = 0
    censored_generations: List[int] = field(default_factory=list)
    """Generations in which every member was right-censored and `SlopeScore`
    was substituted for `CriterionScore` (§07.1, audit A-34)."""

    @property
    def sigma_f(self) -> float:
        if len(self.fitness_history) < 2:
            return 1.0
        vals = list(self.fitness_history)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return max(math.sqrt(var), 1e-6)

    def observe(self, fitnesses: Sequence[float]) -> None:
        for f in fitnesses:
            self.fitness_history.append(float(f))


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------


def _perturbation_for(
    t: torch.Tensor, r_rank: int, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """(2.1)/(2.2): low-rank for 2-D tensors, full-rank Gaussian otherwise.

    SPEC: 02 §2.6.1

    Spec v1 wrote `N(0, I_{p x r})` for *all* tensors, which is undefined for a
    vector. §02.6.1 v2 splits the cases: 1-D germline tensors (`log_sigma`, the
    policy vector) use the full-rank form; the low-rank form applies only to
    2-D tensors, where it is what makes the perturbation storage `O(r(p+q))`.
    """
    if t.dim() == 2:
        p, q = t.shape
        A = torch.randn(p, r_rank, generator=generator, device=t.device, dtype=t.dtype)
        B = torch.randn(q, r_rank, generator=generator, device=t.device, dtype=t.dtype)
        return (A @ B.transpose(0, 1)) / math.sqrt(r_rank)
    return torch.randn(t.shape, generator=generator, device=t.device, dtype=t.dtype)


@dataclass
class Perturbation:
    """One `E_i`, held as the tensors it perturbs."""

    mu: torch.Tensor
    log_sigma: torch.Tensor
    policy: torch.Tensor

    def scaled(self, s: float) -> "Perturbation":
        return Perturbation(self.mu * s, self.log_sigma * s, self.policy * s)


def sample_perturbations(
    genome: Genome, hyper: EsHyper, generator: Optional[torch.Generator] = None
) -> List[Perturbation]:
    """`P_es / 2` mirrored-pair perturbations (§02.6.1).

    The policy perturbation is masked by `Policy.perturbation_mask()`, so a
    pinned constant is **excluded from `E_i`** rather than merely overridden
    afterwards — otherwise the main and ablated arms would differ in their
    germline state even where they agree on every constant (§02.2.3).
    """
    mask = genome.policy.perturbation_mask()
    out: List[Perturbation] = []
    for _ in range(hyper.P_es // 2):
        out.append(
            Perturbation(
                _perturbation_for(genome.field_.mu, hyper.r_rank, generator),
                _perturbation_for(genome.field_.log_sigma, hyper.r_rank, generator),
                _perturbation_for(genome.policy.z, hyper.r_rank, generator) * mask,
            )
        )
    return out


def apply_perturbation(
    genome: Genome, pert: Perturbation, sigma_es: float, sign: float
) -> Genome:
    """Return `Gamma +/- sigma_es * E_i` as a new `Genome` (no aliasing)."""
    g = genome.clone()
    with torch.no_grad():
        g.field_.mu.add_(pert.mu, alpha=sign * sigma_es)
        g.field_.log_sigma.add_(pert.log_sigma, alpha=sign * sigma_es)
        g.policy.z.add_(pert.policy, alpha=sign * sigma_es)
    g.field_.renormalise()
    return g


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


@dataclass
class MemberResult:
    """One evaluated lifetime."""

    fitness: float
    censored: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    archetypes: List[Tuple[int, ...]] = field(default_factory=list)
    mean_utility: float = 0.0
    mean_clade_size: float = 0.0


def es_step(
    genome: Genome,
    evaluate: Callable[[Genome, int, int], MemberResult],
    state: EsState,
    *,
    generator: Optional[torch.Generator] = None,
    member_seed_base: int = 0,
) -> Tuple[Genome, Dict[str, object]]:
    """`ES_STEP` of §02.6.1.

    SPEC: 02 §2.6.1

    ```
    for i in 1 .. P_es/2:                        # mirrored pairs
        E_i ~ (2.1)/(2.2)
        f_{2i-1} <- FITNESS( LIFETIME(Gamma + sigma_es E_i, seed_{2i-1}) )
        f_{2i}   <- FITNESS( LIFETIME(Gamma - sigma_es E_i, seed_{2i}  ) )
    f_hat <- RANK_NORMALISE  (or the centred difference at P_es = 2)
    M <- M + (alpha_es / P_es) * sum_i f_hat_i * E_i^{(+/-)}[M]
    renormalise mu rows to the unit sphere
    postcondition: every policy constant, after its link and rounding, lies in
                   its range
    ```

    `evaluate(genome, member_index, seed)` runs one lifetime and returns its
    `MemberResult`. It is injected rather than imported so that this function is
    testable against an analytic fitness — which is how `G-ES` verifies the
    update direction without running a lifetime at all.

    **Censoring (audit A-34).** If every member is right-censored, the primary
    statistic `CriterionScore` is identically 0 across the generation and carries
    no signal. §07.1 requires the fallback to `SlopeScore`; the caller's
    `MemberResult.fitness` already carries the substituted value, and this
    function records the generation index in `state.censored_generations` so the
    substitution is visible in the analysis.
    """
    hyper = state.hyper
    perts = sample_perturbations(genome, hyper, generator)

    fitnesses: List[float] = []
    results: List[MemberResult] = []
    signs: List[float] = []
    pert_index: List[int] = []

    for i, pert in enumerate(perts):
        for sign in (+1.0, -1.0):
            member = len(fitnesses)
            child = apply_perturbation(genome, pert, hyper.sigma_es, sign)
            res = evaluate(child, member, member_seed_base + member)
            fitnesses.append(float(res.fitness))
            results.append(res)
            signs.append(sign)
            pert_index.append(i)

    if all(r.censored for r in results):
        state.censored_generations.append(state.G_gen)

    if hyper.P_es == 2:
        w_plus, w_minus = centred_difference(fitnesses[0], fitnesses[1], state.sigma_f)
        f_hat = [w_plus, w_minus]
    else:
        f_hat = rank_normalise(fitnesses)

    updated = genome.clone()
    scale = hyper.alpha_es / hyper.P_es
    with torch.no_grad():
        for m, (w, sign, pi) in enumerate(zip(f_hat, signs, pert_index)):
            pert = perts[pi]
            coeff = scale * w * sign
            updated.field_.mu.add_(pert.mu, alpha=coeff)
            updated.field_.log_sigma.add_(pert.log_sigma, alpha=coeff)
            updated.policy.z.add_(pert.policy, alpha=coeff)
    updated.field_.renormalise()

    # Discrete components: **selection** and recombination, not ES (§02.6).
    #
    # **[NORMATIVE — the discrete update selects; v3.1, audit A-86]** v3.0
    # inserted every member's archetypes, losers included, against §02.6's
    # `[NORMATIVE]` statement that "the discrete components ... are updated by
    # selection and recombination". With `P_es = 2` that meant a mirrored pair
    # whose fitnesses were `+1000` and `-1000` contributed **both** their gene
    # sets to the Inventory with equal standing — an inventory of "what
    # appeared", not of "what survived", which is the whole distinction
    # §02.2.4's sampling weight rests on.
    #
    # Selection here is by member fitness against the generation's own mean: a
    # member at or above the mean contributes, one below it does not. With
    # `P_es = 2` that is the mirrored pair's winner, which is the smallest
    # selection this population size admits. Ties contribute — a generation in
    # which both members are identical is not a generation with nothing to
    # inherit.
    if results:
        f_mean = sum(fitnesses) / len(fitnesses)
        survivors = [
            res for res, f in zip(results, fitnesses) if f >= f_mean
        ] or list(results)
    else:
        survivors = []
    for res in survivors:
        for genes in res.archetypes:
            updated.inventory.insert(
                genes,
                mean_utility=res.mean_utility,
                mean_clade_size=res.mean_clade_size,
                generation=genome.G_gen,
            )

    updated.parent_hash = genome.content_hash()
    updated.G_gen = genome.G_gen + 1
    state.G_gen = updated.G_gen
    state.observe(fitnesses)

    # Postcondition (§02.6.1): every constant, after link and rounding, in range.
    # `PolicyConstants.__post_init__` raises on the simplex and on every constant
    # it range-checks, so constructing it *is* most of the check.
    #
    # **[NORMATIVE — v3.1, audit A-91]** It is not all of it. `__post_init__`
    # validates the affine-sigmoid rows and the mutation simplex; the integer
    # constants reached through other links are not covered, and `constants()`
    # accepted `U_cap = 100000` against `U_max = 256` and `sigma_pos = 5.2e21`
    # without complaint. A postcondition that passes on an impossible genome is
    # not a postcondition, and ES is the one place a constant can move without a
    # human looking at it.
    cons = updated.policy.constants()
    # The links that are `softplus`, `round_softplus` or `exp` map onto the
    # **positive reals** and therefore declare no upper bound in
    # `POLICY_SCHEMA`; `__post_init__` range-checks only the affine-sigmoid rows
    # and the simplex. What is unambiguous for the unbounded ones is
    # positivity — a population cap, a parent count or a maturation age of zero
    # is not a small value, it is a broken genome — and that is checked here.
    #
    # Their *upper* bounds are configuration-dependent and are enforced where
    # the configuration is known: `SomaConfig.validate()` requires
    # `U_cap <= U_max`, so a germline that has drifted past a scale's grid is
    # refused when a phenotype is built from it rather than silently accepted.
    # That is the honest division: the germline is scale-free, so it cannot know
    # `U_max`, and inventing a bound here would make it not scale-free.
    for name in ("U_cap", "n_par", "M_mat"):
        v = getattr(cons, name)
        if int(v) < 1:
            raise ValueError(
                f"ES postcondition (§02.6.1): {name} = {v!r} after the update; "
                f"an integer policy constant below 1 is a broken genome, not a "
                f"small value"
            )
    for name in ("sigma_pos", "sigma_mut_0"):
        v = float(getattr(cons, name))
        if not (v > 0.0) or not math.isfinite(v):
            raise ValueError(
                f"ES postcondition (§02.6.1): {name} = {v!r} after the update; "
                f"the `exp` link maps onto the strictly positive reals"
            )

    info: Dict[str, object] = {
        "G_gen": updated.G_gen,
        "fitness": fitnesses,
        "f_hat": f_hat,
        "sigma_f": state.sigma_f,
        "censored": [r.censored for r in results],
        "all_censored": all(r.censored for r in results),
        "parent_hash": updated.parent_hash,
        "content_hash": updated.content_hash(),
        "inventory_size": len(updated.inventory),
    }
    return updated, info


def generation_count(T_run_minutes: float, P_es: int, T_life_minutes: float) -> int:
    """`G_gen = floor( T_run / (P_es * T_life) )` (§02.6.2).

    SPEC: 02 §2.6.2

    §01.9 marks `T_life` `[OPEN]` at both configurations: v2's M1 figure implies
    ~47% of this card's dense bf16 peak from the `6ND` term alone, before
    attention, routing, sparse gathers, `float32` triangular solves, evaluation,
    `erank` SVDs, certification and archive decoding. This function exists so
    that the budget is *computed from a measurement* at gate `G-M0` rather than
    asserted (audit A-37).
    """
    if T_life_minutes <= 0:
        raise ValueError("T_life must be measured and positive (§01.9 [OPEN])")
    return int(T_run_minutes // (P_es * T_life_minutes))
