"""Configuration objects and every derived constant.

Implements the configured dimensions of `docs/spec/01-math-foundations.md` §1.9,
the derived quantities of §1.10, and the canonical policy table of
`02-germline-soma.md` §2.2.3.

SPEC: 01 §1.9
SPEC: 01 §1.10
SPEC: 02 §2.2.3

This module is the single source for every number the rest of the code uses. A
constant that appears anywhere else as a literal is a defect: gate `G-SPECNUM`
checks the values here against the specification tables, and every other module
reads them from a `SomaConfig`.

The germline-scale constants (`N_G`, `k_g`, `m_dim`, `J_max`) are deliberately
*not* per-configuration fields. Spec v2 varied `N_G` between M-nano and M1,
which made the M-nano -> M1 germline inheritance of §02.6.2 shape-impossible
(`mu_g` has shape `N_G x m_dim`); §01.2 fixes them, and `GERMLINE_SCALE` below
is the one place they live.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

__all__ = [
    "GERMLINE_SCALE",
    "GermlineScale",
    "PolicyConstants",
    "SomaConfig",
    "M_NANO",
    "M1",
    "CONFIGS",
    "get_config",
]


# ---------------------------------------------------------------------------
# Germline-scale constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GermlineScale:
    """Constants identical at every configuration.

    SPEC: 01 §1.2

    Only the soma scales. Three consequences, all intended (§1.2): a germline
    transfers between configurations without reshaping; §09.3's claim that
    `|Gamma|` is independent of model scale is true by construction rather than
    by argument; and §08.6.1's `N_G` sweep is interpretable because the default
    holds `N_G` fixed.
    """

    N_G: int = 2048
    """Number of genes; the Site's inner width. There is no separate `d_ff`."""

    k_g: int = 8
    """Genes expressed per unit; `gate(u, c)` is `k_g`-sparse."""

    m_dim: int = 8
    """Dimension of expression space. `p_u, mu_g in S^{m_dim-1}`.

    §1.2: with `m_dim = 2`, 2048 centres on a circle at mean spacing 0.003 rad
    against an RBF spanning ~`sigma_g` rad admit only ~5 resolvable profiles,
    so the expression map would have ~5 effective degrees of freedom whatever
    `N_G` was.
    """

    J_max: int = 256
    """Inventory capacity, in archetypes."""

    n_nbr: int = 8
    """Neighbourhood size for `neighbour(g)` (§02.2.2)."""

    def germline_reals(self, n_policy: int = 19) -> int:
        """`|Gamma|` in reals, per (9.1). Independent of `|U|`, `d_model`, `L`."""
        return self.N_G * (self.m_dim + 1) + self.J_max * self.k_g + n_policy


GERMLINE_SCALE = GermlineScale()


# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConstants:
    """The canonical constant table of §02.2.3, after links and rounding.

    SPEC: 02 §2.2.3

    ES perturbs unconstrained reals; `soma.germline.policy` applies the links.
    This dataclass holds the *linked* values, which is what every consumer
    outside `soma/germline/` needs.

    `phi_ann`, `T_0`, `T_arch`, `T_cert`, `q_cases`, `cap_off`, `epsilon_bif`
    and `tau_death` are derived and live on `SomaConfig` or are computed at run
    time; they are deliberately absent here.
    """

    lambda_wire: float
    lambda_size: float
    lambda_disp: float
    rho_0: float
    n_c_frac: float
    M_mat: int
    sigma_pos: float
    sigma_mut_0: float
    p_point: float
    p_dup: float
    p_del: float
    p_rec: float
    d_rate: float
    k_tok: float
    f_oe: float
    U_cap: int
    n_par: int
    eta_ema: float
    g_growth: float

    def __post_init__(self) -> None:
        s = self.p_point + self.p_dup + self.p_del
        if abs(s - 1.0) > 1e-9:
            raise ValueError(
                f"(p_point, p_dup, p_del) must lie on the simplex; sum = {s!r}"
            )
        # §06.2: values below 0.55 are rejected by config validation, because
        # the cited ratio curve falls *below* the no-selection baseline at 0.5.
        if not 0.55 <= self.k_tok <= 1.0:
            raise ValueError(f"k_tok must lie in [0.55, 1.0]; got {self.k_tok!r}")
        if not 0.0 <= self.f_oe <= 0.5:
            raise ValueError(f"f_oe must lie in [0, 0.5]; got {self.f_oe!r}")
        if not 0.02 <= self.d_rate <= 0.25:
            raise ValueError(f"d_rate must lie in [0.02, 0.25]; got {self.d_rate!r}")
        if not 0.0 <= self.rho_0 <= 0.2:
            raise ValueError(f"rho_0 must lie in [0, 0.2]; got {self.rho_0!r}")
        if not 0.9 <= self.eta_ema <= 0.999:
            raise ValueError(f"eta_ema must lie in [0.9, 0.999]; got {self.eta_ema!r}")
        if not 0.10 <= self.n_c_frac <= 0.50:
            raise ValueError(f"n_c/N_step must lie in [0.10, 0.50]; got {self.n_c_frac!r}")
        # §02.2.3 (A-46): the range is [2, 5] so that the documented default of
        # 4 is strictly *interior*. An affine-sigmoid maps onto the open
        # interval, so a default at an endpoint has no preimage and generation 0
        # cannot be constructed at all.
        if not 2.0 <= self.g_growth <= 5.0:
            raise ValueError(f"g_growth must lie in [2, 5]; got {self.g_growth!r}")
        if not 0.0 <= self.p_rec <= 0.5:
            raise ValueError(f"p_rec must lie in [0, 0.5]; got {self.p_rec!r}")

    @property
    def mutation_simplex(self) -> Tuple[float, float, float]:
        return (self.p_point, self.p_dup, self.p_del)


# Targets from (1.17)-(1.19). Named so the arithmetic reads as the spec does.
E_TARGET = 10
"""Selection events per lifetime (1.17)."""
A_TARGET = 20
"""Archive steps per lifetime (1.18)."""
C_TARGET = 4
"""Certifications per lifetime (1.19)."""

D_MIN = 8
"""Minimum cases per selection event (§04.6)."""

S_REP = 0.5
"""Dispersion kernel width in (1.9)."""

def _d_unif(m_dim: int) -> float:
    """(1.13a): `D_unif(m) = 2^{m-1} Gamma(m/2)^2 / ( sqrt(pi) Gamma(m-1/2) )`.

    The exact mean chord length between two independent uniform points on
    `S^{m-1}`. Duplicated from `soma.model.metabolic.d_unif` (which is the
    documented home) only because `config` may not import `model`; `G-SPECNUM`
    asserts the two agree to machine precision.

    Spec v3.0 tabulated `D_unif(8) = 1.375` as "measured". It is wrong by 1.0%
    (audit A-48), which is why this is computed and not written down.
    """
    return math.exp(
        (m_dim - 1) * math.log(2.0)
        + 2.0 * math.lgamma(m_dim / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma(m_dim - 0.5)
    )


D_UNIF_8 = _d_unif(8)
"""1.389328 — the (1.13) reference at `m_dim = 8`."""

NOCOLLAPSE_FRACTION = 0.60
"""(1.13): `mean ||p_u - p_v|| >= 0.60 * D_unif(m_dim)`."""

SIGMA_G_INIT = 1.2
"""§01.2. Gives measured `median[phi] = 0.4995` at `m_dim = 8`.

Spec v2 used `0.6` with the comment "so a typical phi is near 0.5"; the measured
median there is 0.062 and `N_eff` falls to 0.79, below §01.5's abort guard.
"""


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class SomaConfig:
    """One configuration of the architecture.

    SPEC: 01 §1.9
    SPEC: 05 §5.0

    Every field is a value from §01.9's table; every property below is a derived
    quantity with its defining equation cited. `validate()` asserts the shape
    constraints that §05.0 states, and is called from `__post_init__` so an
    invalid configuration cannot be constructed at all.
    """

    name: str

    # --- block structure -------------------------------------------------
    L: int
    d_model: int
    attn_period: int
    """Attention at `i mod attn_period == attn_period - 1` (§05.1)."""

    # --- attention -------------------------------------------------------
    n_h: int
    n_kv: int

    # --- mixer -----------------------------------------------------------
    n_m: int
    d_k: int
    d_v: int
    e_v: int
    C_chunk: int

    # --- data ------------------------------------------------------------
    T_ctx: int
    V: int
    D_tok: int
    B_tok: int

    # --- population ------------------------------------------------------
    k_route: int
    U_max: int
    side: int
    U_0: int
    d_key: int

    # --- optimisation ----------------------------------------------------
    lr_max: float

    # --- policy ----------------------------------------------------------
    policy: PolicyConstants

    # --- germline scale (shared) -----------------------------------------
    germline: GermlineScale = GERMLINE_SCALE

    # --- growth ----------------------------------------------------------
    growth_schedule_fracs: Tuple[float, ...] = (0.10, 0.20, 0.35)
    """§03.5.1 default for 3 events. Ablation A8 compares {1, 3, 8}."""

    # --- evaluation band --------------------------------------------------
    eval_lengths: Tuple[int, ...] = (40, 64, 96, 128, 192, 256)
    """§08.2. M-nano uses a reduced band; see `validate` and A-32."""

    # --- reference horizon ------------------------------------------------
    N_step_ref_override: int | None = None
    """§07.3: `CriterionScore` is measured against a fixed reference.

    `None` means `N_step_ref = N_step`, which §01.9 sets at both configurations.
    """

    # --- ablation ---------------------------------------------------------
    ablation_pin: Mapping[str, Any] = field(default_factory=dict)
    """§02.2.3: bypasses the germline parameterisation for the named constants
    and excludes them from `E_i`, so endpoint values are reachable and the main
    arm cannot drift into the ablated condition."""

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Derived: schedule
    # ------------------------------------------------------------------

    @property
    def N_step(self) -> int:
        """`ceil(D_tok / B_tok)` (§05.8).

        `D_tok` and `B_tok` are configured and `N_step` derived; spec v1 derived
        `B_tok` from `D_tok` while `D_tok = B_tok * N_step` was itself unknown.
        """
        return math.ceil(self.D_tok / self.B_tok)

    @property
    def N_step_ref(self) -> int:
        """§07.3. Fixed reference horizon for `CriterionScore`."""
        return self.N_step if self.N_step_ref_override is None else self.N_step_ref_override

    @property
    def n_c(self) -> float:
        """Canalization constant, `n_c_frac * N_step` (§01.10)."""
        return self.policy.n_c_frac * self.N_step

    @property
    def T_0(self) -> int:
        """(1.17): `ceil( n_c * ln(1 + N_step/n_c) / E_target )`.

        Derived from a target selection-event count so that the event count is a
        design constant rather than an accident of the budget. Inverting (1.16).
        """
        n_c = self.n_c
        return math.ceil(n_c * math.log1p(self.N_step / n_c) / E_TARGET)

    @property
    def T_arch(self) -> int:
        """(1.18): `max(1, floor(N_step / A_target))`."""
        return max(1, self.N_step // A_TARGET)

    @property
    def T_cert(self) -> int:
        """(1.19): `max(1, floor(N_step / C_target))`."""
        return max(1, self.N_step // C_TARGET)

    @property
    def phi_ann(self) -> int:
        """Mask anneal length (§03.3): `min(5000, max(50, floor(0.02 N_step)))`."""
        return min(5000, max(50, int(0.02 * self.N_step)))

    @property
    def n_warm(self) -> int:
        """§05.8: `0.01 N_step`."""
        return max(1, int(0.01 * self.N_step))

    @property
    def n_dec(self) -> int:
        """§05.8: `0.86 N_step`."""
        return int(0.86 * self.N_step)

    @property
    def T_log(self) -> int:
        """§07.6."""
        return 50

    @property
    def T_val(self) -> int:
        """§07.1: `max(50, floor(N_step / 100))`, so `n_crit` has >=100-point
        resolution."""
        return max(50, self.N_step // 100)

    @property
    def T_ref(self) -> int:
        """§06.1.2: `theta_lag` update period, in steps."""
        return 32

    @property
    def h_lag(self) -> float:
        """§06.1.2: target half-life of `theta_lag`, in steps."""
        return 0.05 * self.N_step

    @property
    def kappa_lag(self) -> float:
        """(6.2): `0.5 ** (T_ref / h_lag)`, stated per *update*, not per step."""
        return 0.5 ** (self.T_ref / self.h_lag)

    @property
    def growth_schedule(self) -> Tuple[int, ...]:
        """§03.5.1: explicit steps at which `GROW` fires."""
        return tuple(int(f * self.N_step) for f in self.growth_schedule_fracs)

    # ------------------------------------------------------------------
    # Derived: selection
    # ------------------------------------------------------------------

    @property
    def q_cases(self) -> int:
        """§01.10 / §04.6: `max(D_min, round(d_rate * B_tok))`.

        **Proportional** to `d_rate`. Spec v2 used `ceil(D_min / d_rate)`, which
        is inversely proportional, so §04.5.1's cost governor amplified the cost
        it was meant to bound.
        """
        return max(D_MIN, round(self.policy.d_rate * self.B_tok))

    @property
    def n_birth_max(self) -> int:
        """`floor(rho_0 * U_cap)` — the largest birth cohort the cap admits."""
        return int(self.policy.rho_0 * self.policy.U_cap)

    @property
    def cap_off(self) -> int:
        """§04.6: `max(2, ceil(0.5 * n_birth_max))`, so the cap is *reachable*.

        Spec v2 fixed `cap_off = 6` at M1 against a per-event birth count that
        could not reach it, so §04.10's falsifier and §07.6's hit-rate log were
        identically zero by arithmetic rather than by population health.
        """
        return max(2, math.ceil(0.5 * self.n_birth_max))

    # ------------------------------------------------------------------
    # Derived: dimensions
    # ------------------------------------------------------------------

    @property
    def d_h(self) -> int:
        return self.d_model // self.n_h

    @property
    def D_qkv(self) -> int:
        """§05.2.4: `n_m * (2 d_k + d_v)`."""
        return self.n_m * (2 * self.d_k + self.d_v)

    @property
    def N_G(self) -> int:
        return self.germline.N_G

    @property
    def k_g(self) -> int:
        return self.germline.k_g

    @property
    def m_dim(self) -> int:
        return self.germline.m_dim

    @property
    def J_max(self) -> int:
        return self.germline.J_max

    @property
    def attention_layers(self) -> Tuple[int, ...]:
        """§05.1: block indices carrying attention rather than the mixer."""
        return tuple(
            i for i in range(self.L) if i % self.attn_period == self.attn_period - 1
        )

    @property
    def mixer_layers(self) -> Tuple[int, ...]:
        attn = set(self.attention_layers)
        return tuple(i for i in range(self.L) if i not in attn)

    @property
    def L_rand(self) -> float:
        """§07.1: `ln V`, a shape-independent constant."""
        return math.log(self.V)

    @property
    def nocollapse_threshold(self) -> float:
        """(1.13): `0.60 * D_unif(m_dim)`."""
        return NOCOLLAPSE_FRACTION * _d_unif(self.m_dim)

    # ------------------------------------------------------------------
    # Derived: realised event counts
    # ------------------------------------------------------------------

    def realised_selection_events(self) -> int:
        """Count `SELECT` firings under the counter of (1.15).

        (1.16) is the continuum solution and gives the *target*; the realised
        count follows the integer recursion `n_{j+1} = n_j + ceil(T_sel(n_j))`
        and may differ by one. §01.7 forbids any section from assuming the
        target is attained exactly, so this is computed, never assumed.
        """
        n_next = self.T_0
        warm = max(self.T_0, self.policy.M_mat)
        count = 0
        n = 0
        while n < self.N_step:
            if n >= n_next and n > 0 and n >= warm:
                count += 1
                n_next = n + math.ceil(self.T_sel(n))
            n += 1
        return count

    def realised_periodic_events(self, period: int) -> int:
        """Firings of a fixed-period hook under (1.15) over one lifetime."""
        if period <= 0:
            raise ValueError("period must be positive")
        count = 0
        n_next = period
        for n in range(self.N_step):
            if n >= n_next and n > 0:
                count += 1
                n_next = n + period
        return count

    def T_sel(self, n: int) -> float:
        """(3.11): `T_0 * (1 + n / n_c)`."""
        return self.T_0 * (1.0 + n / self.n_c)

    def rho(self, n: int) -> float:
        """(3.10): `rho_0 * exp(-n / n_c)`."""
        return self.policy.rho_0 * math.exp(-n / self.n_c)

    def b_str(self, n: int) -> float:
        """(3.12): structural churn per step."""
        return self.rho(n) / self.T_sel(n)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Assert every constraint §05.0 and §01.9 state.

        SPEC: 05 §5.0
        """
        g = self.germline
        errs = []

        def req(cond: bool, msg: str) -> None:
            if not cond:
                errs.append(msg)

        req(self.d_model % self.n_h == 0, f"d_model {self.d_model} % n_h {self.n_h} != 0")
        req(self.n_h % self.n_kv == 0, f"n_h {self.n_h} % n_kv {self.n_kv} != 0")
        req(
            self.n_m * self.d_v == self.d_model * self.e_v,
            f"n_m*d_v = {self.n_m * self.d_v} != d_model*e_v = {self.d_model * self.e_v}",
        )
        req(self.U_max == self.side ** 2, f"U_max {self.U_max} != side^2 {self.side ** 2}")
        req(self.U_0 <= self.U_max, f"|U|_0 {self.U_0} > U_max {self.U_max}")
        req(
            self.policy.U_cap <= self.U_max,
            f"U_cap {self.policy.U_cap} > U_max {self.U_max}",
        )
        req(self.d_key % 2 == 0, f"d_key {self.d_key} must be even (product-key split)")
        req(_is_pow2(self.C_chunk), f"C_chunk {self.C_chunk} must be a power of two")
        req(_is_pow2(self.B_tok), f"B_tok {self.B_tok} must be a power of two")
        req(self.k_route <= self.U_0, f"k_route {self.k_route} > |U|_0 {self.U_0}")
        req(g.k_g <= g.N_G, "k_g > N_G")
        req(g.n_nbr < g.N_G, "n_nbr >= N_G")
        req(2 <= self.attn_period <= self.L, "attn_period out of range")
        req(len(self.attention_layers) >= 1, "no attention layer in the stack")

        # §03.3 (A-41): a unit born at one event must finish its anneal, or its
        # death, before the next. v1's phi_ann = 5000 against T_sel = 2000 is
        # exactly the configuration §03.4's dying-unit exclusion was added to
        # contain; asserting the inequality prevents it recurring.
        req(
            self.phi_ann < self.T_0,
            f"phi_ann {self.phi_ann} >= T_0 {self.T_0}: the mask anneal outlives "
            f"the selection interval",
        )

        # §01.9: T_ctx must hold the longest evaluated prompt plus its answer.
        req(
            max(self.eval_lengths) * 2 + 2 <= self.T_ctx * 2,
            "eval band exceeds what T_ctx can serialise",
        )

        req(self.n_warm < self.n_dec < self.N_step, "warmup/decay schedule degenerate")
        req(self.T_arch >= 1 and self.T_cert >= 1, "archive/certification period < 1")
        req(self.q_cases >= D_MIN, f"q_cases {self.q_cases} < D_min {D_MIN}")

        unknown = set(self.ablation_pin) - _PINNABLE
        req(not unknown, f"ablation.pin names unknown constants: {sorted(unknown)}")

        if errs:
            raise ValueError(
                f"invalid SomaConfig {self.name!r}:\n  " + "\n  ".join(errs)
            )

    def with_pins(self, **pins: Any) -> "SomaConfig":
        """Return a copy with `ablation.pin` entries applied (§02.2.3).

        The pin bypasses the germline parameterisation entirely, which is what
        makes endpoint values (`lambda_wire = 0`, `k_tok = 1.0`, `f_oe = 0`,
        `p_rec = 0`, `rho_0 = 0`, `U_cap = U_max`) reachable at all: the links
        of §02.2.3 cannot express them.
        """
        unknown = set(pins) - _PINNABLE
        if unknown:
            raise ValueError(f"cannot pin unknown constants: {sorted(unknown)}")
        pol_fields = {f for f in PolicyConstants.__dataclass_fields__}
        pol_pins = {k: v for k, v in pins.items() if k in pol_fields}
        cfg_pins = {k: v for k, v in pins.items() if k not in pol_fields}
        policy = replace(self.policy, **pol_pins) if pol_pins else self.policy
        merged = dict(self.ablation_pin)
        merged.update(pins)
        return replace(self, policy=policy, ablation_pin=merged, **cfg_pins)

    def summary(self) -> Dict[str, Any]:
        """Every configured and derived number, for the run manifest."""
        return {
            "name": self.name,
            "L": self.L,
            "d_model": self.d_model,
            "N_G": self.N_G,
            "k_g": self.k_g,
            "m_dim": self.m_dim,
            "J_max": self.J_max,
            "n_h": self.n_h,
            "n_kv": self.n_kv,
            "d_h": self.d_h,
            "n_m": self.n_m,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "e_v": self.e_v,
            "C_chunk": self.C_chunk,
            "T_ctx": self.T_ctx,
            "V": self.V,
            "D_tok": self.D_tok,
            "B_tok": self.B_tok,
            "N_step": self.N_step,
            "N_step_ref": self.N_step_ref,
            "k_route": self.k_route,
            "U_max": self.U_max,
            "side": self.side,
            "U_0": self.U_0,
            "U_cap": self.policy.U_cap,
            "d_key": self.d_key,
            "lr_max": self.lr_max,
            "n_c": self.n_c,
            "T_0": self.T_0,
            "T_arch": self.T_arch,
            "T_cert": self.T_cert,
            "phi_ann": self.phi_ann,
            "q_cases": self.q_cases,
            "cap_off": self.cap_off,
            "n_warm": self.n_warm,
            "n_dec": self.n_dec,
            "T_val": self.T_val,
            "T_log": self.T_log,
            "kappa_lag": self.kappa_lag,
            "growth_schedule": list(self.growth_schedule),
            "attention_layers": list(self.attention_layers),
            "L_rand": self.L_rand,
            "germline_reals": self.germline.germline_reals(),
            "realised_selection_events": self.realised_selection_events(),
            "realised_archive_steps": self.realised_periodic_events(self.T_arch),
            "realised_certifications": self.realised_periodic_events(self.T_cert),
            "ablation_pin": dict(self.ablation_pin),
        }


_PINNABLE = frozenset(
    set(PolicyConstants.__dataclass_fields__)
    | {"growth_schedule_fracs", "eval_lengths"}
)
"""Constants an ablation may pin (§02.2.3, §08.6)."""


# ---------------------------------------------------------------------------
# The two configurations of §01.9
# ---------------------------------------------------------------------------

_POLICY_NANO = PolicyConstants(
    lambda_wire=1e-3,
    lambda_size=1e-6,
    lambda_disp=1e-3,
    rho_0=0.05,
    n_c_frac=0.25,
    M_mat=100,
    sigma_pos=0.10,
    sigma_mut_0=0.05,
    p_point=0.5,
    p_dup=0.3,
    p_del=0.2,
    p_rec=0.10,
    d_rate=0.10,
    k_tok=0.65,
    f_oe=0.25,
    U_cap=240,
    n_par=8,
    eta_ema=0.99,
    g_growth=4.0,
)

_POLICY_M1 = PolicyConstants(
    lambda_wire=1e-3,
    lambda_size=1e-6,
    lambda_disp=1e-3,
    rho_0=0.02,
    n_c_frac=0.25,
    M_mat=400,
    sigma_pos=0.10,
    sigma_mut_0=0.05,
    p_point=0.5,
    p_dup=0.3,
    p_del=0.2,
    p_rec=0.10,
    d_rate=0.10,
    k_tok=0.65,
    f_oe=0.25,
    U_cap=960,
    n_par=24,
    eta_ema=0.99,
    g_growth=4.0,
)


M_NANO = SomaConfig(
    name="m-nano",
    L=6,
    d_model=192,
    attn_period=3,
    n_h=4,
    n_kv=2,
    n_m=3,
    d_k=64,
    d_v=64,
    e_v=1,
    C_chunk=32,
    T_ctx=128,
    V=4096,
    D_tok=int(4e7),
    B_tok=16384,
    k_route=4,
    U_max=256,
    side=16,
    U_0=64,
    d_key=32,
    lr_max=3e-3,
    policy=_POLICY_NANO,
    # A-32: T_ctx = 128 cannot hold a length-256 prompt plus its answer, so the
    # 40..256 band of §08.2 is not computable here. §08.6 assigns M-nano the
    # reduced band with pass bars unchanged, and adjudicates K7 at M1.
    eval_lengths=(40, 64, 96),
)

M1 = SomaConfig(
    name="m1",
    L=12,
    d_model=512,
    attn_period=4,
    n_h=8,
    n_kv=2,
    n_m=4,
    d_k=128,
    d_v=128,
    e_v=1,
    C_chunk=64,
    T_ctx=512,
    V=8192,
    D_tok=int(8e8),
    B_tok=65536,
    k_route=8,
    U_max=1024,
    side=32,
    U_0=256,
    d_key=64,
    lr_max=1e-3,
    policy=_POLICY_M1,
    eval_lengths=(40, 64, 96, 128, 192, 256),
)

CONFIGS: Dict[str, SomaConfig] = {"m-nano": M_NANO, "m1": M1}


def get_config(name: str) -> SomaConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"unknown configuration {name!r}; known: {sorted(CONFIGS)}"
        ) from None


def m0_config(base: SomaConfig) -> SomaConfig:
    """The M0 / B1 baseline at `base`'s scale (§07.4.2, §08.4).

    SPEC: 07 §7.4.2

    M0 is the §05 stack with the Site replaced by a dense `ReLU^2` FFN of inner
    width `N_G`, and with no germline, units, selection or stream. It is
    represented as a `SomaConfig` with every SOMA mechanism pinned off, so the
    two arms cannot diverge in any dimension the config controls.

    §07.1 (A-33): M-nano requires its *own* M0, because M0's `L_star`,
    `slope_ref` and `theta_m0` are all measured at `V = 8192, T_ctx = 512` and
    are incomparable at `V = 4096, T_ctx = 128`.
    """
    return replace(
        base,
        name=f"m0-{base.name}",
        policy=replace(
            base.policy,
            rho_0=0.0,
            f_oe=0.0,
            k_tok=1.0,
            lambda_wire=0.0,
            lambda_disp=0.0,
            lambda_size=0.0,
        ),
        ablation_pin={
            "rho_0": 0.0,
            "f_oe": 0.0,
            "k_tok": 1.0,
            "lambda_wire": 0.0,
            "lambda_disp": 0.0,
            "lambda_size": 0.0,
        },
        growth_schedule_fracs=(),
    )
