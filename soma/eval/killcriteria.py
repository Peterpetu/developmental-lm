"""K1 … K10: §10.1's kill criteria, as mechanical decisions.

SPEC: 10 §10.1
SPEC: 08 §8.5

**[NORMATIVE]** Spec v1's criteria all had the form "if X is unchanged on
**every** metric, the component is decorative". Two things made them unfireable:
no equivalence test existed — a non-significant difference is not evidence of
equivalence — and a conjunction over the dozens of fields in §07.6 will always
have one differing by chance across seeds. §10.1 replaced each with a **small
pre-registered primary metric set** and a **mechanical decision**, and this
module is that mechanism.

**[NORMATIVE — v3.1, audit A-78]** v3.0 had no such module. `scripts/spec_trace.py`
exempted §10.1 with the reason *"kill criteria are evaluated in analysis, mapped
via eval/killcriteria"* — naming a file that did not exist. Two of the ten were
implemented in scattered form (`K1Verdict` in `soma/generations.py`, K8 inline in
`scripts/eval_tinystories.py`); the other eight existed only as prose. A criterion
that is only prose is not pre-registered in any sense that binds: it is decided,
after the fact, by whoever is reading the numbers.

**Every criterion here returns the same object**, so a report can print all ten
with their inputs, and a criterion whose inputs are absent returns
`applicable=False` rather than a default. That distinction is the point: "K3 did
not fire" and "K3 was never evaluated" must not be the same line in a report.

**Order.** §10.1: *"K6 and K9 are checked **first**: both invalidate everything
downstream and both are cheap."* `evaluate_all` returns them first and
`Verdicts.blocking` names any that has fired, so a report cannot bury a mixer
mismatch under nine other rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from soma.utils.stats import mann_kendall, tost

__all__ = [
    "KillVerdict",
    "Verdicts",
    "k1_generation_curve",
    "k2_development",
    "k3_case_selection",
    "k4_connection_cost",
    "k5_open_endedness",
    "k6_mixer_oracle",
    "k7_state_tracking",
    "k8_plasticity",
    "k9_ng_scaling",
    "k10_unit_identity",
    "evaluate_all",
]

ALPHA_SIG = 0.05
"""§10.1: every criterion is decided at this level."""

N_SEED_MIN = 5
"""§07.5: `n_seed >= 5` is what makes `alpha_sig = 0.05` attainable at all — a
3-seed bootstrap admits 27 resamples and a minimum `p` of 0.037."""

DELTA_Q_ROUTE = 0.02
"""§10.1 K4's TOST margin on `Q_route`."""

DELTA_NOVELTY = 0.34
"""§10.1 K5's margin. Set above the `1/7` granularity of the novelty fraction
(§06.4, audit A-51); v2's `0.10` lay inside the gap between two attainable
values, so neither TOST could reject and K5 was undecidable in both
directions (audit A-29)."""

MIXER_TOL = 1e-5
"""§05.2.6's float32 bound; K6 fires on `max |chunk - ref| / (|ref| + 1e-8)`."""

PARITY_BAR = 0.95
"""§08.2's scaled-accuracy bar for `PARITY` at length 256."""

ERANK_SLOPE = -1e-4
"""§10.1 K8: a significant negative trend **and** a slope below this."""


@dataclass
class KillVerdict:
    """One criterion's decision, with everything that produced it."""

    key: str
    fires: bool
    applicable: bool
    """False when the inputs do not exist. A criterion that was never evaluated
    is not a criterion that did not fire, and a report must be able to tell them
    apart."""
    reason: str
    detail: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "K": self.key,
            "fires": self.fires,
            "applicable": self.applicable,
            "reason": self.reason,
            **self.detail,
        }


def _not_applicable(key: str, why: str) -> KillVerdict:
    return KillVerdict(key, False, False, why)


# ---------------------------------------------------------------------------
# K6 and K9 — checked first (§10.1)
# ---------------------------------------------------------------------------


def k6_mixer_oracle(
    max_rel_error: Optional[float], engineer_days_spent: float = 0.0
) -> KillVerdict:
    """`max |chunk - ref| / (|ref| + 1e-8) > 1e-5` on §05.2.6's grid.

    **[NORMATIVE]** *"after 3 engineer-days"* is part of the condition, not
    commentary: a mismatch found and fixed inside three days is a bug, and only
    an unresolved one is a kill. The parameter is explicit so the decision does
    not depend on who is reading the calendar.

    Decision: **halt all downstream work.** No result computed with a wrong
    mixer is admissible.
    """
    if max_rel_error is None:
        return _not_applicable("K6", "no mixer-oracle comparison supplied")
    over = float(max_rel_error) > MIXER_TOL
    fires = over and engineer_days_spent >= 3.0
    return KillVerdict(
        "K6", fires, True,
        "mixer disagrees with the oracle beyond 1e-5 and is unresolved"
        if fires else
        ("mixer disagrees but the 3-engineer-day window has not elapsed"
         if over else "mixer matches the oracle"),
        {"max_rel_error": float(max_rel_error), "tolerance": MIXER_TOL,
         "engineer_days": float(engineer_days_spent)},
    )


def k9_ng_scaling(sweep_report: Optional[Dict[str, object]]) -> KillVerdict:
    """The `CE`-minimising `N_G` grows proportionally to `d_model` (§08.6.1).

    Reads `scripts/run_ng_sweep.py`'s report, which fits
    `log2 N_G*(d) = a + b log2 d` and fires when `b` is closer to 1
    (proportional) than to 0 (scale-free).

    Decision: the bottleneck claim of §09.1 is false and the architecture is a
    hypernetwork with the known ceiling. §10.1 calls this the **most damaging
    single finding**, which is why it is checked first.
    """
    if not sweep_report:
        return _not_applicable("K9", "no N_G sweep report supplied")
    b = sweep_report.get("log2_slope_b")
    if b is None:
        return _not_applicable("K9", "sweep report carries no fitted slope")
    edge = list(sweep_report.get("argmin_at_grid_edge") or [])
    fires = bool(sweep_report.get("K9_fires"))
    reason = "the CE-minimising N_G grows with d_model" if fires else \
        "the CE-minimising N_G does not grow proportionally with d_model"
    if edge:
        reason += (f"; NOTE the argmin sits at a grid edge for d_model in "
                   f"{edge}, so the grid may be what determined it")
    return KillVerdict("K9", fires, True, reason,
                       {"log2_slope_b": float(b), "argmin_at_grid_edge": edge,
                        "argmin_n_g": sweep_report.get("argmin_n_g")})


# ---------------------------------------------------------------------------
# K1 — the constitutional criterion
# ---------------------------------------------------------------------------


def k1_generation_curve(verdict: Optional[Dict[str, object]]) -> KillVerdict:
    """`NOT(b_criterion > 0)` **AND** `NOT(b_slope > 0)` in §07.5's BCa bootstrap.

    **[NORMATIVE]** A conjunction of two *failures*, not a disjunction: spec v1's
    §07.5 read "kill if **both** fail" and §10 K1 read "kill if **either**
    fails", and the two readings decide the project's central verdict
    differently (audit A-72's neighbour, resolved in v3.0).

    Consumes `soma.generations.analyse_curves(...).as_dict()`. If that analysis
    reports the curve **uninterpretable** — §07.5's censoring rule, more than
    half the lifetimes right-censored — K1 is `applicable=False`: a verdict
    computed from a degenerate statistic is not a verdict.
    """
    if not verdict:
        return _not_applicable("K1", "no generation-curve analysis supplied")
    if not verdict.get("arms_differenced"):
        return _not_applicable(
            "K1",
            "K1 is decided on the difference main - a4, paired by seed "
            "(§07.5, audit A-82). An arm's absolute curve is confounded by "
            "ordinary recipe drift and is not interpretable alone; this "
            "verdict came from analyse_curves on one arm.",
        )
    if not verdict.get("interpretable", True):
        return _not_applicable(
            "K1",
            f"censored fraction {verdict.get('censored_fraction')} > 0.5; §07.5 "
            f"makes the generation curve uninterpretable and L_star must be "
            f"re-derived at the measured D_tok",
        )
    n_seeds = int(verdict.get("n_seeds", 0))
    if n_seeds < N_SEED_MIN:
        return _not_applicable(
            "K1", f"{n_seeds} seeds; §07.5 requires n_seed >= {N_SEED_MIN}"
        )
    crit_sig = bool(verdict.get("b_criterion_significant"))
    slope_sig = bool(verdict.get("b_slope_significant"))
    fires = (not crit_sig) and (not slope_sig)
    return KillVerdict(
        "K1", fires, True,
        "neither adaptation-efficiency slope is significantly positive"
        if fires else
        "at least one adaptation-efficiency slope is significantly positive",
        {"b_criterion": verdict.get("b_criterion"),
         "b_criterion_significant": crit_sig,
         "b_slope": verdict.get("b_slope"),
         "b_slope_significant": slope_sig,
         "n_seeds": n_seeds},
    )


# ---------------------------------------------------------------------------
# K2, K3, K4, K10 — ablation equivalence
# ---------------------------------------------------------------------------


def _equivalent_on(
    key: str,
    metrics: Dict[str, tuple],
    margins: Dict[str, float],
    *,
    require_all: bool = True,
) -> tuple:
    """Run TOST per metric. Returns `(all_equivalent, detail)`.

    `metrics[name] = (baseline_seeds, ablated_seeds)`, paired by seed.
    §08.5's "unchanged" is **equivalence**, never a failure to reject.
    """
    detail: Dict[str, object] = {}
    flags: List[bool] = []
    for name, (a, b) in metrics.items():
        delta = margins[name]
        r = tost(a, b, delta, alpha=ALPHA_SIG, paired=True)
        detail[name] = {
            "verdict": r.verdict, "mean_difference": r.mean_difference,
            "ci": [r.ci_low, r.ci_high], "delta": delta, "n": r.n,
        }
        flags.append(r.equivalent)
    ok = all(flags) if require_all else any(flags)
    return ok, detail


def _noise_margin(noise_floor: Optional[float]) -> Optional[float]:
    """§07.7: `delta = 2 x noise_floor(metric)`, the seed-to-seed standard
    deviation over five differently-seeded M0 repeats.

    Not `sd_probe`: §07.7 forbids substituting one for the other, and they
    differ by orders of magnitude."""
    if noise_floor is None or noise_floor <= 0:
        return None
    return 2.0 * float(noise_floor)


def k2_development(
    ce_full: Optional[Sequence[float]] = None,
    ce_a3: Optional[Sequence[float]] = None,
    *,
    noise_floor_ce: Optional[float] = None,
) -> KillVerdict:
    """A3 (selection off) equivalent to full SOMA on `{held-out CE}`.

    Decision: development is decorative. Remove births/deaths; retain selection
    over a fixed population only if A2 still shows a difference.
    """
    delta = _noise_margin(noise_floor_ce)
    if not ce_full or not ce_a3 or delta is None:
        return _not_applicable(
            "K2", "needs paired held-out CE for full/A3 and noise_floor(CE)"
        )
    ok, detail = _equivalent_on("K2", {"held_out_ce": (ce_full, ce_a3)},
                                {"held_out_ce": delta})
    return KillVerdict("K2", ok, True,
                       "A3 is equivalent to full SOMA on held-out CE" if ok
                       else "A3 is not equivalent on held-out CE", detail)


def k3_case_selection(
    metrics: Optional[Dict[str, tuple]] = None,
    *,
    noise_floor_ce: Optional[float] = None,
    noise_floor_ks: Optional[float] = None,
) -> KillVerdict:
    """A2 (mean-utility selection) equivalent on
    `{held-out CE, Q_route, case-win KS}`.

    All three must be equivalent: §10.1 names a **small** metric set precisely so
    that the conjunction is decidable, unlike v1's conjunction over §07.6.

    Decision: case-based selection is decorative. The residue is a
    metabolically-priced MoE and is described as such.
    """
    needed = ("held_out_ce", "Q_route", "case_win_ks")
    if not metrics or any(k not in metrics for k in needed):
        return _not_applicable(
            "K3", f"needs paired {needed} for full and A2"
        )
    d_ce = _noise_margin(noise_floor_ce)
    d_ks = _noise_margin(noise_floor_ks)
    if d_ce is None or d_ks is None:
        return _not_applicable("K3", "needs noise_floor for CE and for the KS statistic")
    margins = {"held_out_ce": d_ce, "Q_route": DELTA_Q_ROUTE, "case_win_ks": d_ks}
    ok, detail = _equivalent_on("K3", metrics, margins)
    return KillVerdict("K3", ok, True,
                       "A2 is equivalent on all three primary metrics" if ok
                       else "A2 differs on at least one primary metric", detail)


def k4_connection_cost(
    q_full: Optional[Sequence[float]] = None,
    q_a1a: Optional[Sequence[float]] = None,
) -> KillVerdict:
    """A1a (`lambda_wire = 0`) equivalent on `{Q_route}` under `delta = 0.02`.

    The margin is absolute and specified, not derived from a noise floor:
    `Q_route` is a modularity in `[-1, 1]` and `0.02` is §10.1's number.

    Decision: the connection-cost mechanism does not operate here; retain the
    term only if it independently improves `CE`.
    """
    if not q_full or not q_a1a:
        return _not_applicable("K4", "needs paired Q_route for full and A1a")
    ok, detail = _equivalent_on("K4", {"Q_route": (q_full, q_a1a)},
                                {"Q_route": DELTA_Q_ROUTE})
    return KillVerdict("K4", ok, True,
                       "A1a is equivalent on Q_route" if ok
                       else "A1a differs on Q_route", detail)


def k10_unit_identity(
    metrics: Optional[Dict[str, tuple]] = None,
    *,
    noise_floor_ce: Optional[float] = None,
    k4: Optional[KillVerdict] = None,
) -> KillVerdict:
    """B3 (random router) equivalent on `{held-out CE, Q_route}` **and** K4 has
    already fired.

    The conjunction with K4 is part of the criterion. B3 equivalence alone says
    routing carries no information *given* a connection cost that may itself be
    inert; K10 is the statement that both are.

    Decision: unit identity carries no information; the population is a fiction
    and the work reduces to a dense model.
    """
    needed = ("held_out_ce", "Q_route")
    if not metrics or any(k not in metrics for k in needed):
        return _not_applicable("K10", f"needs paired {needed} for full and B3")
    if k4 is None or not k4.applicable:
        return _not_applicable("K10", "K4 has not been evaluated; K10 is conditional on it")
    d_ce = _noise_margin(noise_floor_ce)
    if d_ce is None:
        return _not_applicable("K10", "needs noise_floor(CE)")
    ok, detail = _equivalent_on(
        "K10", metrics, {"held_out_ce": d_ce, "Q_route": DELTA_Q_ROUTE}
    )
    fires = ok and k4.fires
    detail["K4_fired"] = k4.fires
    return KillVerdict(
        "K10", fires, True,
        "B3 is equivalent and K4 has fired" if fires else
        ("B3 is equivalent but K4 has not fired" if ok
         else "B3 is not equivalent"),
        detail,
    )


# ---------------------------------------------------------------------------
# K5, K7, K8
# ---------------------------------------------------------------------------


def k5_open_endedness(
    live_minus_ramp: Optional[Sequence[float]] = None,
    live_minus_frozen: Optional[Sequence[float]] = None,
) -> KillVerdict:
    """Live-minus-ramp equivalent to zero under `delta = 0.34`, **or**
    live-minus-frozen equivalent to zero.

    A **disjunction** (§10.1's "or"), and the two arms are not
    interchangeable: the ramp arm is the actual confound — monotone difficulty
    ramping — and K5 is *adjudicated* on `live - ramp`, with `live - frozen`
    retained as a second way for the criterion to fire.

    Decision: the stream is not open-ended. Stop calling it open-ended;
    re-describe as curriculum shaping; reduce `f_oe` and re-evaluate whether it
    earns its compute by the A6 result.
    """
    if not live_minus_ramp and not live_minus_frozen:
        return _not_applicable("K5", "no certification differences supplied")
    detail: Dict[str, object] = {}
    fired: List[str] = []
    for name, xs in (("live_minus_ramp", live_minus_ramp),
                     ("live_minus_frozen", live_minus_frozen)):
        if not xs:
            detail[name] = {"verdict": "absent"}
            continue
        zeros = [0.0] * len(xs)
        r = tost(list(xs), zeros, DELTA_NOVELTY, alpha=ALPHA_SIG, paired=True)
        detail[name] = {"verdict": r.verdict, "mean_difference": r.mean_difference,
                        "ci": [r.ci_low, r.ci_high], "delta": DELTA_NOVELTY, "n": r.n}
        if r.equivalent:
            fired.append(name)
    return KillVerdict(
        "K5", bool(fired), True,
        f"equivalent to zero on {', '.join(fired)}" if fired
        else "neither difference is equivalent to zero",
        detail,
    )


def k7_state_tracking(
    parity_at_256_median: Optional[float] = None,
    curve_by_length: Optional[Dict[int, float]] = None,
) -> KillVerdict:
    """`PARITY` scaled accuracy median-of-3 below `0.95` at length 256, **and**
    the accuracy-vs-length curve is flat over `{40..256}`.

    **[NORMATIVE]** §08.2 evaluates K7 on the **curve**, not a single point, and
    against the **median** of 3 seeds — the cited source's own medians are
    materially weaker than its best-of-3. A low point with a *falling* curve is a
    length-generalisation limit; a low point with a *flat* curve is the claim
    failing, and only the second fires.

    "Flat" is `max - min <= 0.05` over the evaluated lengths, which is the
    smallest spread §08.2's bars can distinguish.
    """
    if parity_at_256_median is None or not curve_by_length:
        return _not_applicable("K7", "needs the median PARITY curve over lengths")
    lengths = sorted(curve_by_length)
    vals = [curve_by_length[k] for k in lengths]
    spread = max(vals) - min(vals)
    flat = spread <= 0.05
    below = float(parity_at_256_median) < PARITY_BAR
    fires = below and flat
    return KillVerdict(
        "K7", fires, True,
        "PARITY is below the bar at 256 and the length curve is flat" if fires
        else ("PARITY is below the bar but the curve is not flat" if below
              else "PARITY meets the bar at 256"),
        {"parity_256_median": float(parity_at_256_median), "bar": PARITY_BAR,
         "lengths": lengths, "scaled_acc": vals, "spread": spread, "flat": flat},
    )


def k8_plasticity(erank_tail: Optional[Sequence[float]] = None) -> KillVerdict:
    """`erank` shows a significant negative Mann–Kendall trend over the final
    50% with slope below `-1e-4 / step`.

    Both halves are required: a significant trend that is negligible in
    magnitude is not a collapse, and a steep slope that is not significant is
    noise. Spec v1 said "does not decline monotonically", which a noisy statistic
    satisfies trivially.
    """
    if not erank_tail or len(erank_tail) < 3:
        return _not_applicable("K8", "fewer than 3 erank points in the final 50%")
    mk = mann_kendall(list(erank_tail), alternative="decreasing")
    fires = bool(mk.p_value < ALPHA_SIG and mk.slope < ERANK_SLOPE)
    return KillVerdict(
        "K8", fires, True,
        "significant negative erank trend with a material slope" if fires
        else "no significant negative erank trend of material slope",
        {"p_value": mk.p_value, "slope": mk.slope, "method": mk.method,
         "n_points": len(erank_tail), "slope_bar": ERANK_SLOPE},
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class Verdicts:
    """All ten, in §10.1's checking order."""

    verdicts: List[KillVerdict]

    @property
    def fired(self) -> List[str]:
        return [v.key for v in self.verdicts if v.fires]

    @property
    def not_evaluated(self) -> List[str]:
        return [v.key for v in self.verdicts if not v.applicable]

    @property
    def blocking(self) -> List[str]:
        """K6 and K9 invalidate everything downstream (§10.1)."""
        return [v.key for v in self.verdicts if v.fires and v.key in ("K6", "K9")]

    def as_dict(self) -> Dict[str, object]:
        return {
            "verdicts": [v.as_dict() for v in self.verdicts],
            "fired": self.fired,
            "not_evaluated": self.not_evaluated,
            "blocking": self.blocking,
        }


def evaluate_all(**inputs) -> Verdicts:
    """Evaluate every criterion from whatever inputs are available.

    §10.1: *"K6 and K9 are checked first: both invalidate everything downstream
    and both are cheap."* They are returned first, and `Verdicts.blocking` names
    either if it fired, so a report cannot bury a mixer mismatch under nine other
    rows.

    A criterion whose inputs are absent is reported `applicable=False` with the
    reason. Nothing defaults to "did not fire".
    """
    k4 = k4_connection_cost(inputs.get("q_full"), inputs.get("q_a1a"))
    out = [
        k6_mixer_oracle(inputs.get("mixer_max_rel_error"),
                        inputs.get("mixer_engineer_days", 0.0)),
        k9_ng_scaling(inputs.get("ng_sweep_report")),
        k1_generation_curve(inputs.get("generation_verdict")),
        k2_development(inputs.get("ce_full"), inputs.get("ce_a3"),
                       noise_floor_ce=inputs.get("noise_floor_ce")),
        k3_case_selection(inputs.get("a2_metrics"),
                          noise_floor_ce=inputs.get("noise_floor_ce"),
                          noise_floor_ks=inputs.get("noise_floor_ks")),
        k4,
        k5_open_endedness(inputs.get("live_minus_ramp"),
                          inputs.get("live_minus_frozen")),
        k7_state_tracking(inputs.get("parity_256_median"),
                          inputs.get("parity_curve")),
        k8_plasticity(inputs.get("erank_tail")),
        k10_unit_identity(inputs.get("b3_metrics"),
                          noise_floor_ce=inputs.get("noise_floor_ce"), k4=k4),
    ]
    return Verdicts(out)
