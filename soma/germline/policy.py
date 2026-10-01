"""`Policy` — germline constants as unconstrained reals, with their links.

SPEC: 02 §2.2.3

Every constant is stored as an **unconstrained real** and mapped by a fixed
link; ES perturbs the unconstrained reals. Spec v1 mandated `sigmoid`/`softplus`
only, which cannot express a simplex, an integer, or an interval other than
`(0,1)` — so its own postcondition ("every policy constant lies in its range")
was false for the majority of the table.

The `ablation.pin` path bypasses this parameterisation entirely for the named
constants and excludes them from `E_i`, which is what makes endpoint values
(`lambda_wire = 0`, `k_tok = 1.0`, `f_oe = 0`, `p_rec = 0`, `rho_0 = 0`,
`U_cap = U_max`) reachable at all: no link can attain its own interval endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from soma.config import PolicyConstants
from soma.germline.registry import GermlineRegistry
from soma.utils.shapes import check_shape

__all__ = ["LinkSpec", "POLICY_SCHEMA", "Policy", "softplus", "affine_sigmoid"]


# ---------------------------------------------------------------------------
# Links (§02.2.3)
# ---------------------------------------------------------------------------


def softplus(z: float) -> float:
    """`log(1 + e^z)`, computed stably for large `|z|`."""
    if z > 30.0:
        return z
    if z < -30.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def inv_softplus(y: float) -> float:
    if y <= 0.0:
        raise ValueError(f"softplus is strictly positive; cannot invert {y!r}")
    if y > 30.0:
        return y
    return math.log(math.expm1(y))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def affine_sigmoid(z: float, lo: float, hi: float) -> float:
    """`lo + (hi - lo) * sigmoid(z)` — an open interval `(lo, hi)`."""
    return lo + (hi - lo) * _sigmoid(z)


def inv_affine_sigmoid(y: float, lo: float, hi: float) -> float:
    if not lo < y < hi:
        raise ValueError(
            f"affine-sigmoid maps onto the open interval ({lo}, {hi}); "
            f"{y!r} is not interior. Endpoint values are reachable only through "
            f"ablation.pin (§02.2.3)."
        )
    u = (y - lo) / (hi - lo)
    return math.log(u / (1.0 - u))


def _softmax(zs: Sequence[float]) -> Tuple[float, ...]:
    m = max(zs)
    es = [math.exp(z - m) for z in zs]
    s = sum(es)
    return tuple(e / s for e in es)


def _inv_softmax(ps: Sequence[float]) -> Tuple[float, ...]:
    # log p is a valid preimage up to an additive constant; centred for
    # determinism so a round trip is idempotent.
    logs = [math.log(max(p, 1e-12)) for p in ps]
    mean = sum(logs) / len(logs)
    return tuple(l - mean for l in logs)


@dataclass(frozen=True)
class LinkSpec:
    """One row of the §02.2.3 table."""

    name: str
    link: str
    """`softplus` | `affine_sigmoid` | `exp` | `round_softplus` | `softmax`"""
    lo: Optional[float] = None
    hi: Optional[float] = None
    group: Optional[str] = None
    """Simplex membership, for `softmax` rows."""

    def forward(self, z: float) -> float:
        if self.link == "softplus":
            return softplus(z)
        if self.link == "exp":
            return math.exp(z)
        if self.link == "affine_sigmoid":
            assert self.lo is not None and self.hi is not None
            return affine_sigmoid(z, self.lo, self.hi)
        if self.link == "round_softplus":
            return float(max(1, round(softplus(z))))
        raise ValueError(f"{self.name}: link {self.link!r} is not scalar-valued")

    def inverse(self, y: float) -> float:
        if self.link == "softplus":
            return inv_softplus(y)
        if self.link == "exp":
            if y <= 0:
                raise ValueError(f"{self.name}: exp link needs y > 0; got {y!r}")
            return math.log(y)
        if self.link == "affine_sigmoid":
            assert self.lo is not None and self.hi is not None
            return inv_affine_sigmoid(y, self.lo, self.hi)
        if self.link == "round_softplus":
            return inv_softplus(max(1.0, float(y)))
        raise ValueError(f"{self.name}: link {self.link!r} is not scalar-valued")

    @property
    def is_integer(self) -> bool:
        return self.link == "round_softplus"


POLICY_SCHEMA: Tuple[LinkSpec, ...] = (
    LinkSpec("lambda_wire", "softplus"),
    LinkSpec("lambda_size", "softplus"),
    LinkSpec("lambda_disp", "softplus"),
    LinkSpec("rho_0", "affine_sigmoid", 0.0, 0.2),
    LinkSpec("n_c_frac", "affine_sigmoid", 0.10, 0.50),
    LinkSpec("M_mat", "round_softplus"),
    LinkSpec("sigma_pos", "exp"),
    LinkSpec("sigma_mut_0", "exp"),
    LinkSpec("p_point", "softmax", group="mut"),
    LinkSpec("p_dup", "softmax", group="mut"),
    LinkSpec("p_del", "softmax", group="mut"),
    LinkSpec("p_rec", "affine_sigmoid", 0.0, 0.5),
    LinkSpec("d_rate", "affine_sigmoid", 0.02, 0.25),
    LinkSpec("k_tok", "affine_sigmoid", 0.55, 1.0),
    LinkSpec("f_oe", "affine_sigmoid", 0.0, 0.5),
    LinkSpec("U_cap", "round_softplus"),
    LinkSpec("n_par", "round_softplus"),
    LinkSpec("eta_ema", "affine_sigmoid", 0.9, 0.999),
    LinkSpec("g_growth", "affine_sigmoid", 2.0, 5.0),
)
"""The canonical table of §02.2.3, in a fixed order.

The order is the storage order of the unconstrained vector, so it is part of the
serialisation format and may not be permuted without a schema-version bump.

`lambda_disp` is new in spec v3: (1.9)'s dispersion term is what prevents the
positional collapse the sphere constraint does not (audit A-9), so it is a real
germline constant and belongs in the ES search space, not a hard-coded literal.
"""

_BY_NAME: Dict[str, LinkSpec] = {s.name: s for s in POLICY_SCHEMA}
_SIMPLEX_NAMES = frozenset(s.name for s in POLICY_SCHEMA if s.link == "softmax")
"""Pinning any member of a simplex group pins the whole group, since a softmax
preimage is only defined for a complete, normalised set."""
N_POLICY = len(POLICY_SCHEMA)


@dataclass
class Policy:
    """The germline policy vector and its decoding.

    SPEC: 02 §2.2.3

    `z` is `(n_policy,)` of unconstrained reals — the object ES perturbs.
    `constants()` applies the links and returns the `PolicyConstants` every
    other module consumes.
    """

    z: torch.Tensor
    pinned: Mapping[str, Any]
    """`ablation.pin` entries: bypass the link and are excluded from `E_i`."""

    def __post_init__(self) -> None:
        check_shape(self.z, [N_POLICY], name="policy")
        if self.z.requires_grad:
            raise ValueError("policy is germline; gradients are prohibited (§02.1)")
        unknown = set(self.pinned) - set(_BY_NAME)
        if unknown:
            raise ValueError(f"pinned names not in the policy table: {sorted(unknown)}")

    # ------------------------------------------------------------------

    @classmethod
    def from_constants(
        cls,
        c: PolicyConstants,
        *,
        pinned: Optional[Mapping[str, Any]] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Policy":
        """Invert the links to obtain the unconstrained vector for `c`.

        **[NORMATIVE — a pinned constant is not inverted]** §02.2.3: an
        `ablation.pin` **bypasses the germline parameterisation entirely**. A
        pinned value is typically an interval *endpoint* (`lambda_wire = 0`,
        `k_tok = 1.0`, `f_oe = 0`, `p_rec = 0`, `rho_0 = 0`), which no link can
        represent — inverting it raises. Its slot in `z` is therefore filled
        with `0.0`, a value the pin overrides on every read and which
        `perturbation_mask` excludes from `E_i`, so it never influences
        anything.
        """
        pinned = dict(pinned or {})
        values: List[float] = []
        simplex = _softmax_preimages(c) if not (set(pinned) & _SIMPLEX_NAMES) else None
        for spec in POLICY_SCHEMA:
            if spec.name in pinned:
                values.append(0.0)
            elif spec.link == "softmax":
                if simplex is None:
                    values.append(0.0)
                else:
                    values.append(simplex[spec.name])
            else:
                values.append(spec.inverse(float(getattr(c, spec.name))))
        z = torch.tensor(values, device=torch.device(device), dtype=dtype)
        return cls(z, pinned)

    def register(self, registry: GermlineRegistry, prefix: str = "") -> None:
        registry.register(f"{prefix}policy", self.z)

    # ------------------------------------------------------------------

    def constants(self) -> PolicyConstants:
        """Apply the links, then the pins, then round. §02.2.3.

        The postcondition of `ES_STEP` holds **after rounding** (§02.2.3), which
        is why rounding happens here rather than at the call site: an integer
        constant that is in range before rounding and out of range after it
        would satisfy a postcondition checked in the wrong place.
        """
        raw: Dict[str, float] = {}
        groups: Dict[str, List[str]] = {}
        for i, spec in enumerate(POLICY_SCHEMA):
            if spec.link == "softmax":
                groups.setdefault(spec.group or "", []).append(spec.name)
            else:
                raw[spec.name] = spec.forward(float(self.z[i]))
        for _, names in groups.items():
            zs = [float(self.z[_index_of(n)]) for n in names]
            for n, p in zip(names, _softmax(zs)):
                raw[n] = p

        for name, value in self.pinned.items():
            raw[name] = float(value)

        for spec in POLICY_SCHEMA:
            if spec.is_integer:
                raw[spec.name] = int(round(raw[spec.name]))

        return PolicyConstants(
            lambda_wire=raw["lambda_wire"],
            lambda_size=raw["lambda_size"],
            lambda_disp=raw["lambda_disp"],
            rho_0=raw["rho_0"],
            n_c_frac=raw["n_c_frac"],
            M_mat=int(raw["M_mat"]),
            sigma_pos=raw["sigma_pos"],
            sigma_mut_0=raw["sigma_mut_0"],
            p_point=raw["p_point"],
            p_dup=raw["p_dup"],
            p_del=raw["p_del"],
            p_rec=raw["p_rec"],
            d_rate=raw["d_rate"],
            k_tok=raw["k_tok"],
            f_oe=raw["f_oe"],
            U_cap=int(raw["U_cap"]),
            n_par=int(raw["n_par"]),
            eta_ema=raw["eta_ema"],
            g_growth=raw["g_growth"],
        )

    def perturbation_mask(self) -> torch.Tensor:
        """1 where ES may perturb, 0 for pinned constants (§02.2.3).

        A pinned constant is **excluded from `E_i`**, not merely overridden after
        the fact: otherwise the main arm's ES could drift the unconstrained real
        while the pin held the value fixed, and the two arms would differ in
        their germline state even though they agreed on every constant.
        """
        mask = torch.ones_like(self.z)
        for name in self.pinned:
            mask[_index_of(name)] = 0.0
        return mask

    def clone(self) -> "Policy":
        return Policy(self.z.clone(), dict(self.pinned))

    def to(self, device: torch.device | str) -> "Policy":
        return Policy(self.z.to(device), dict(self.pinned))

    def as_dict(self) -> Dict[str, Any]:
        c = self.constants()
        return {s.name: getattr(c, s.name) for s in POLICY_SCHEMA}


def _index_of(name: str) -> int:
    for i, spec in enumerate(POLICY_SCHEMA):
        if spec.name == name:
            return i
    raise KeyError(name)


def _softmax_preimages(c: PolicyConstants) -> Dict[str, float]:
    """Preimages for every simplex group, centred so the round trip is stable."""
    out: Dict[str, float] = {}
    groups: Dict[str, List[str]] = {}
    for spec in POLICY_SCHEMA:
        if spec.link == "softmax":
            groups.setdefault(spec.group or "", []).append(spec.name)
    for _, names in groups.items():
        ps = [float(getattr(c, n)) for n in names]
        total = sum(ps)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"simplex group {names} sums to {total!r}, not 1")
        for n, z in zip(names, _inv_softmax(ps)):
            out[n] = z
    return out
