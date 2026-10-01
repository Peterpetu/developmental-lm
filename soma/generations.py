"""The decisive experiment: germline generations under single-point ES.

SPEC: 07 §7.5
SPEC: 02 §2.6
SPEC: 07 §7.8
SPEC: 09 §9.4

**[NORMATIVE]** The central claim is **not** that a single run beats a baseline;
growth at matched total compute is approximately a wash. The claim is about
capital formation across generations:

> **Adaptation efficiency improves monotonically across germline generations at
> fixed per-lifetime token budget.**

**Protocol.** Run at **M-nano**.

1. `G_gen = 0` starts from a random germline.
2. Each generation runs `P_es = 2` mirrored lifetimes, scores by (7.1), and
   applies `ES_STEP` (§02.6.1). **Single-point ES**; spec v1's §07.5 described
   drawing germlines "from the current population", which is a different
   algorithm from its own §02.6.
3. Record `CriterionScore(g)`, `early_slope(g)`, and `BirthScore(g)`.

**Analysis.** Primary statistic `y(g) = CriterionScore(g)`; secondary
`early_slope(g)`. Fit `y(g) = a + b log(1 + g)`. Test `b > 0` one-sided at
`alpha_sig = 0.05` by **BCa bootstrap over seeds**, resampling **seeds** (not
lifetimes), with `n_seed >= 5`.

**Control.** Ablation **A4** runs the identical protocol with the germline reset
to random each generation. The **difference** between the curves is the
inheritance effect; the absolute curves are confounded by ordinary recipe drift
and are not interpretable alone.

**[NORMATIVE — kill rule, stated as a boolean]** K1 fires iff
`NOT(b_criterion > 0 significant)` **AND** `NOT(b_slope > 0 significant)`, i.e.
the project is killed only when **both** pre-registered tests fail. Spec v1's
§07.5 said "either" and §10 K1 said "or", and the two readings decide the
project's central verdict differently.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.config import SomaConfig
from soma.germline import EsHyper, EsState, Genome, MemberResult, es_step
from soma.germline.es import generation_count
from soma.train import LifetimeConfig, LifetimeResult, lifetime
from soma.utils.stats import bca_bootstrap_slope
from soma.utils.rng import derive_seed

__all__ = [
    "GenerationRecord",
    "GenerationCurve",
    "run_generations",
    "analyse_curves",
    "analyse_arm_difference",
    "K1Verdict",
]


@dataclass
class GenerationRecord:
    """One generation of one seed."""

    seed: int
    g: int
    criterion: float
    slope_score: float
    early_slope: float
    birth: float
    fitness: List[float]
    censored: List[bool]
    all_censored: bool
    wall_seconds: float
    n_c_frac: float = float("nan")
    """§09.4: the canalization constant the **selected** germline carries into
    generation `g + 1`, as a fraction of `N_step`.

    §09.4 predicts it should *decrease* in `g`: at generation `G_gen` the
    germline already encodes a body plan that worked, so less discovery is
    needed. Recorded per generation because the prediction is a trend in `g`,
    and a trend cannot be tested from the final value alone. It is a
    **secondary** confirmation, not an early warning — detecting a trend in a
    single evolving scalar needs the same `G_gen` and seed count as §07.5's
    primary test, which is the v1 claim that section corrects."""
    rho_0: float = float("nan")
    """The other constant §09.4 predicts should evolve downward."""

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed, "g": self.g, "criterion": self.criterion,
            "slope_score": self.slope_score, "early_slope": self.early_slope,
            "birth": self.birth, "fitness": self.fitness,
            "censored": self.censored, "all_censored": self.all_censored,
            "wall_seconds": self.wall_seconds,
            "n_c_frac": self.n_c_frac, "rho_0": self.rho_0,
        }


@dataclass
class GenerationCurve:
    """One seed's trajectory across generations."""

    seed: int
    arm: str
    records: List[GenerationRecord] = field(default_factory=list)

    def criterion_points(self) -> List[Tuple[int, float]]:
        return [(r.g, r.criterion) for r in self.records]

    def slope_points(self) -> List[Tuple[int, float]]:
        return [(r.g, r.slope_score) for r in self.records]

    def canalization_points(self) -> List[Tuple[int, float]]:
        """§09.4's secondary confirmation: `n_c_frac` against generation."""
        return [(r.g, r.n_c_frac) for r in self.records]

    def birth_points(self) -> List[Tuple[int, float]]:
        return [(r.g, r.birth) for r in self.records]

    @property
    def censored_fraction(self) -> float:
        """§07.1: reported, with a `> 0.5` interpretability threshold."""
        if not self.records:
            return 1.0
        return sum(r.all_censored for r in self.records) / len(self.records)


def run_generations(
    cfg: SomaConfig,
    make_lifetime_config: Callable[[str, int], LifetimeConfig],
    *,
    n_generations: int,
    seeds: Sequence[int],
    arm: str = "main",
    hyper: Optional[EsHyper] = None,
    reset_each_generation: bool = False,
    solve_rate_fn=None,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[GenerationCurve]:
    """Run the generation curve for one arm.

    `reset_each_generation` selects ablation **A4**: the germline is reset to
    random each generation, so the arm measures ordinary recipe drift with no
    inheritance. §08.6 additionally requires that a random germline be accepted
    only if it passes §02.5 on the **first** attempt — resampling until
    acceptance would itself be a selection process contaminating the control —
    and that the ES perturbation scale be **matched** across arms, which it is
    because both arms share `hyper`.
    """
    hyper = hyper or EsHyper()
    if len(seeds) < 5:
        raise ValueError(
            f"§07.5 requires n_seed >= 5 for alpha_sig = 0.05; got {len(seeds)}. "
            f"A 3-seed bootstrap admits 27 resamples and a minimum p of 0.037."
        )
    curves: List[GenerationCurve] = []

    for seed in seeds:
        t_seed = time.time()
        state = EsState(hyper)
        gen_torch = torch.Generator().manual_seed(derive_seed(seed, "es_perturb") % (2**31))
        genome = Genome.initialise(
            cfg, generator=torch.Generator().manual_seed(seed), lineage_id=f"{arm}-s{seed}"
        )
        curve = GenerationCurve(seed, arm)

        for g in range(n_generations):
            if reset_each_generation:
                # A4: a fresh random germline each generation, accepted on the
                # first attempt (§08.6). The Inventory is *not* carried over,
                # which is what makes this an inheritance control rather than a
                # perturbation-scale control.
                genome = Genome.initialise(
                    cfg,
                    generator=torch.Generator().manual_seed(seed * 100003 + g),
                    lineage_id=f"{arm}-s{seed}",
                )
                genome.G_gen = g

            t_gen = time.time()
            results: List[LifetimeResult] = []

            def evaluate(child: Genome, member: int, member_seed: int) -> MemberResult:
                lc = make_lifetime_config(f"{arm}-s{seed}-g{g}-m{member}", member_seed)
                res = lifetime(child, lc, member_seed, solve_rate_fn=solve_rate_fn)
                results.append(res)
                return MemberResult(
                    fitness=res.fitness.fitness,
                    censored=res.fitness.censored,
                    metrics=res.fitness.as_dict(),
                    archetypes=res.archetypes,
                    mean_utility=res.mean_utility,
                    mean_clade_size=res.mean_clade_size,
                )

            genome, info = es_step(
                genome, evaluate, state, generator=gen_torch,
                member_seed_base=derive_seed(seed, "member") % 100000 + g * 1000,
            )

            rec = GenerationRecord(
                seed=seed, g=g,
                criterion=float(np.mean([r.fitness.criterion for r in results])),
                slope_score=float(np.mean([r.fitness.slope for r in results])),
                early_slope=float(np.mean([r.curve.early_slope() for r in results])),
                birth=float(np.mean([r.fitness.birth for r in results])),
                fitness=[r.fitness.fitness for r in results],
                censored=[r.fitness.censored for r in results],
                all_censored=bool(info["all_censored"]),
                wall_seconds=time.time() - t_gen,
                n_c_frac=float(genome.constants().n_c_frac),
                rho_0=float(genome.constants().rho_0),
            )
            curve.records.append(rec)
            if verbose:
                print(
                    f"  [{arm} seed={seed}] g={g:3d} crit={rec.criterion:.4f} "
                    f"slope={rec.slope_score:.4f} birth={rec.birth:+.4f} "
                    f"({rec.wall_seconds:.1f}s)"
                )
            if out_dir:
                _append(out_dir, f"{arm}.jsonl", rec.as_dict())

        curves.append(curve)
        if verbose:
            print(f"  [{arm} seed={seed}] done in {time.time() - t_seed:.1f}s "
                  f"censored_fraction={curve.censored_fraction:.2f}")
    return curves


def _append(out_dir: str, name: str, payload: Dict[str, object]) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / name).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


@dataclass
class K1Verdict:
    """§10 K1, stated as an explicit boolean."""

    b_criterion: float
    b_criterion_significant: bool
    b_slope: float
    b_slope_significant: bool
    censored_fraction: float
    interpretable: bool
    fires: bool
    detail: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "b_criterion": self.b_criterion,
            "b_criterion_significant": self.b_criterion_significant,
            "b_slope": self.b_slope,
            "b_slope_significant": self.b_slope_significant,
            "censored_fraction": self.censored_fraction,
            "interpretable": self.interpretable,
            "K1_fires": self.fires,
            **self.detail,
        }


def analyse_arm_difference(
    main: Sequence["GenerationCurve"],
    control: Sequence["GenerationCurve"],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> K1Verdict:
    """The **inheritance effect**: the `main` curve minus the `a4` control,
    paired by seed.

    SPEC: 07 §7.5
    SPEC: 10 §10.1

    **[NORMATIVE — K1 is decided on the difference; v3.1, audit A-82]** This
    module's own contract says it and v3.0 did not do it:

    > The **difference** between the curves is the inheritance effect; the
    > absolute curves are confounded by ordinary recipe drift and **are not
    > interpretable alone**.

    v3.0 called `analyse_curves` on each arm independently and read `K1_fires`
    off `main`. Under that reading, two arms improving **identically** — pure
    recipe drift, an inheritance effect of exactly zero — report success:
    measured on a synthetic pair with the same `0.30 log(1+g)` drift in both
    arms, `b_criterion = +0.3003` and `K1_fires = False` for both, so the
    project would have been declared to have met C1 on an effect of `0.0`. The
    A4 arm — half the M3 GPU budget — was computed and discarded.

    Pairing is **by seed**, not by arm mean. The two arms share seeds by
    construction (`run_generations` runs the same seed list for both), so the
    seed-level difference removes the seed's own recipe draw, which is exactly
    the confound A-28 introduced the control to remove. A seed present in one
    arm and not the other is dropped, and the drop is recorded.
    """
    by_seed_main = {c.seed: c for c in main}
    by_seed_ctrl = {c.seed: c for c in control}
    shared = sorted(set(by_seed_main) & set(by_seed_ctrl))
    dropped = sorted(set(by_seed_main) ^ set(by_seed_ctrl))

    def diff(getter) -> List[List[Tuple[int, float]]]:
        out: List[List[Tuple[int, float]]] = []
        for seed in shared:
            a = dict(getter(by_seed_main[seed]))
            b = dict(getter(by_seed_ctrl[seed]))
            gens = sorted(set(a) & set(b))
            out.append([(g, a[g] - b[g]) for g in gens])
        return out

    if len(shared) < 5:
        # §07.5 fixes `n_seed >= 5`; below it the BCa acceleration is unstable
        # and `bca_bootstrap_slope` refuses. Refusing here too — with the reason
        # — is better than propagating a ValueError out of an analysis: a
        # verdict that cannot be computed is `interpretable=False`, not a crash
        # at the end of a 46-hour run.
        return K1Verdict(
            float("nan"), False, float("nan"), False, 1.0, False, False,
            {"n_seeds": len(shared), "seeds_dropped": dropped,
             "arms_differenced": True,
             "why": f"only {len(shared)} seeds are shared by both arms; "
                    f"§07.5 requires n_seed >= 5"},
        )

    rng = rng if rng is not None else np.random.default_rng(20260828)
    crit = bca_bootstrap_slope(
        diff(lambda c: c.criterion_points()), n_resamples=n_resamples,
        alpha=alpha, rng=rng,
    )
    slope = bca_bootstrap_slope(
        diff(lambda c: c.slope_points()), n_resamples=n_resamples,
        alpha=alpha, rng=rng,
    )
    censored = float(np.mean(
        [by_seed_main[s].censored_fraction for s in shared]
        + [by_seed_ctrl[s].censored_fraction for s in shared]
    )) if shared else 1.0
    interpretable = censored <= 0.5 and len(shared) >= 1
    fires = (not crit.significant) and (not slope.significant)
    return K1Verdict(
        crit.estimate, crit.significant, slope.estimate, slope.significant,
        censored, interpretable, fires,
        {
            "criterion_ci": [crit.ci_low, crit.ci_high],
            "slope_ci": [slope.ci_low, slope.ci_high],
            "criterion_p": crit.p_one_sided,
            "slope_p": slope.p_one_sided,
            "n_seeds": len(shared),
            "n_generations": (len(by_seed_main[shared[0]].records)
                              if shared else 0),
            "arms_differenced": True,
            "seeds_dropped": dropped,
        },
    )


def analyse_curves(
    curves: Sequence[GenerationCurve],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> K1Verdict:
    """Fit `y(g) = a + b log(1+g)` and test `b > 0` by BCa over seeds (§07.5).

    **[NORMATIVE — this is a per-arm statistic, not K1]** K1 is decided on
    `analyse_arm_difference(main, a4)`, because an arm's absolute curve is
    confounded by recipe drift (audit A-82). This function is what produces each
    arm's own curve for reporting and for the difference; it is **not** the
    verdict.

    **[NORMATIVE — K1's boolean]** K1 fires iff **both** pre-registered tests
    fail. That is a conjunction of two failures, not a disjunction: spec v1's
    §07.5 said "either" and §10 K1 said "or", and the two readings decide the
    project's central verdict differently.

    **[NORMATIVE — censoring]** If the censored fraction exceeds `0.5`, `L_star`
    was set too aggressively for the lifetime budget and the generation curve is
    **not interpretable**; the finding is reported and `L_star` is re-derived at
    the measured `D_tok`, with every affected generation re-run (§07.1,
    audit A-34). The verdict records this rather than silently returning a
    number computed from a degenerate statistic.
    """
    rng = rng if rng is not None else np.random.default_rng(20260828)
    crit = bca_bootstrap_slope(
        [c.criterion_points() for c in curves], n_resamples=n_resamples,
        alpha=alpha, rng=rng,
    )
    slope = bca_bootstrap_slope(
        [c.slope_points() for c in curves], n_resamples=n_resamples,
        alpha=alpha, rng=rng,
    )
    censored = float(np.mean([c.censored_fraction for c in curves])) if curves else 1.0
    interpretable = censored <= 0.5
    fires = (not crit.significant) and (not slope.significant)
    return K1Verdict(
        crit.estimate, crit.significant, slope.estimate, slope.significant,
        censored, interpretable, fires,
        {
            "criterion_ci": [crit.ci_low, crit.ci_high],
            "slope_ci": [slope.ci_low, slope.ci_high],
            "criterion_p": crit.p_one_sided,
            "slope_p": slope.p_one_sided,
            "n_seeds": len(curves),
            "n_generations": len(curves[0].records) if curves else 0,
        },
    )


def budget_report(
    T_life_minutes: float, cfg: SomaConfig, *, n_seed: int, n_generations: int,
    P_es: int = 2, T_run_minutes: float = 720.0,
) -> Dict[str, object]:
    """Recompute the M3 budget from the **measured** `T_life` (§07.4.1).

    SPEC: 07 §7.4.1

    §01.9 marks `T_life` `[OPEN]` at both configurations, so `G_gen` and the M3
    budget are derived here rather than asserted. If the measured `T_life`
    exceeds `1.8` min at M-nano the experiment exceeds 12 h and one of `n_seed`,
    `G_gen` or `N_step` must be reduced **before** it starts — with `n_seed >= 5`
    the constraint that may not be reduced, since it is what makes
    `alpha_sig = 0.05` attainable at all.
    """
    arms = 2  # main and the A4 control
    total = arms * n_seed * n_generations * P_es * T_life_minutes
    return {
        "T_life_minutes": T_life_minutes,
        "G_gen_in_run": generation_count(T_run_minutes, P_es, T_life_minutes),
        "total_minutes": total,
        "total_hours": total / 60.0,
        "within_12h": total <= T_run_minutes,
        "n_seed": n_seed,
        "n_generations": n_generations,
        "arms": arms,
        "note": (
            "n_seed >= 5 may not be reduced (§07.5); reduce G_gen or N_step "
            "instead, and record the power consequence in the manifest"
        ),
    }
