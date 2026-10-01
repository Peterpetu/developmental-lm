"""Criticality: the depth-wise finite-time Lyapunov exponents of the active stack and the
homeostatic rule.

SPEC: 07 §7.1  (one tangent propagated through the whole active stack per window and
                perturbation; per-sub-block exponents from consecutive tangent norms; the
                relative exponents subtract the residual stream's own growth)
SPEC: 07 §7.2  (Lambda = sum of the per-block exponents = the stack's exponent, exactly)
SPEC: 07 §7.3  (zeta <- clamp(zeta - kappa clip(lambda_rel, -1, 1), -2, 2))
SPEC: 07 §7.4  (cost: one JVP per (window, delta); never in the step loop)
SPEC: 07 §7.6  (K-CRIT is conditional on the profile leaving the band)

Second pass after the adversarial review (11 §P-C.30): the first form drew an
independent isotropic perturbation per sub-block, whose sum is not the exponent of the
composed map. The tangent is now propagated: with `delta_0 = delta` at the residual
input and `delta_k = J_k delta_{k-1}` through the sub-blocks, `lambda_k = ½ log(||delta_k||² /
||delta_{k-1}||²)` and `sum_k lambda_k = ½ log(||delta_last||² / ||delta_0||²)` telescopes.
Because the residual stream grows with depth and the output is RMS-normalised, the neutral
point for a *relative* perturbation is not `lambda = 0` but the stream's own growth rate;
`lambda_k^{rel} = lambda_k - log(RMS(h_k) / RMS(h_{k-1}))` is the quantity the rule and the
band use. The JVP is forward-mode (`torch.func.jvp`) with the exact double-backward trick as
the fallback, in fp32 with autocast disabled (K5).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from gnomon.utils.shapes import check_shape
from .backbone import Backbone
from .morphology import SwitchPattern

__all__ = [
    "N_PROBE_WINDOWS", "N_DELTA", "KAPPA", "LAMBDA_CLIP", "ZETA_CLAMP", "BAND", "SE_GATE", "TINY_NORM",
    "exponent_of_map", "stack_exponents", "probe", "zeta_update", "trigger", "needs_control",
]

N_PROBE_WINDOWS = 16
N_DELTA = 4
KAPPA = 0.5
LAMBDA_CLIP = 1.0
ZETA_CLAMP = 2.0
TINY_NORM = 1e-30
"""A floor under every squared norm the exponents take a logarithm of."""
BAND = 0.5
"""SPEC: 07 §7.6 -- the band is on the **per-block** relative exponent: `Lambda / K_a` and
`max_k |lambda_k^rel|`, not on `Lambda` itself. `Lambda` is a *sum* over the `K_a = 2 x active
layers` sub-blocks, so a fixed band on it is a band on depth as much as on criticality: the
same per-block behaviour that gives `Lambda = -0.51` at the 8-layer primordium gives
`-1.02` at the 16-layer adult, and growth alone would fire the trigger (11 §P-C.33)."""
SE_GATE = 2.0
"""SPEC: 07 §7.2, §7.6 -- no verdict is read off a displacement smaller than `SE_GATE` standard
errors. The gate is part of the trigger, not a footnote to it."""


JVP_FALLBACK = "does not support forward"
"""The only failure the double-backward fallback may be used for: an operator without a
forward-mode rule. Any other RuntimeError -- a dtype mismatch, an out-of-memory, a bad shape --
is a defect and must propagate, because the fallback runs with `strict=False` and returns
**zeros** for an output it thinks is independent of the input, which would enter the profile as
`log 0` rather than as a failure (11 §P-C.33)."""


def _jvp(f: Callable, H: torch.Tensor, delta: torch.Tensor):
    """`(f(H), J_f(H) delta)`, exact: forward-mode first, the double-backward trick as the
    fallback for operators with no forward-mode rule, and nothing else."""
    try:
        return torch.func.jvp(f, (H,), (delta,))
    except NotImplementedError:
        pass
    except RuntimeError as exc:
        if JVP_FALLBACK not in str(exc):
            raise
    return torch.autograd.functional.jvp(f, H, delta, create_graph=False, strict=True)


def exponent_of_map(f: Callable[[torch.Tensor], torch.Tensor], H: torch.Tensor, *, n_delta: int = N_DELTA,
                    generator: Optional[torch.Generator] = None) -> float:
    """`½ E_delta log(||J delta||² / ||delta||²)` of one map on one window `H (1, T, d)` (K2)."""
    check_shape(H, [1, None, None], name="H")
    with torch.autocast(device_type=H.device.type, enabled=False):
        H32 = H.detach().float()
        vals = []
        for _ in range(n_delta):
            delta = torch.randn(H32.shape, generator=generator, dtype=torch.float32).to(H32.device)
            _, Jd = _jvp(f, H32, delta)
            if Jd.dtype != torch.float32:
                raise TypeError(f"the probe must run in fp32, got {Jd.dtype}")
            vals.append(0.5 * (torch.log(Jd.square().sum()) - torch.log(delta.square().sum())))
        return float(torch.stack(vals).mean())


def stack_exponents(backbone: Backbone, h0: torch.Tensor, state: Optional[dict], *, pattern: SwitchPattern,
                    gains: Optional[dict], zeta: Optional[torch.Tensor], n_delta: int = N_DELTA,
                    generator: Optional[torch.Generator] = None) -> dict:
    """The per-sub-block exponents of one window from the propagated tangent. SPEC: 07 (7.1)-(7.2).

    `h0 (1, T, d)` is the residual stream entering layer 0 (after `W_in`), `state` the
    entering delta states of that window (held fixed). Returns `{"keys": [(kind, l), ...],
    "abs": [...], "rel": [...]}` with the per-draw mean of `lambda_k` and `lambda_k^{rel}`.
    """
    check_shape(h0, [1, None, backbone.cfg.d], name="h0")
    cfg = backbone.cfg
    keys = []
    for l in pattern.active_layers():
        keys.append(("mix", l))
        keys.append(("ffn", l))
    gains32 = None if gains is None else {k: v.detach().float() for k, v in gains.items()}
    zeta32 = None if zeta is None else zeta.detach().float()
    state32 = None if state is None else {l: s.detach().float() for l, s in state.items()}

    def f(H):
        outs = []
        h = H
        for kind, l in keys:
            S = None if (kind == "ffn" or state32 is None) else state32.get(l)
            h = backbone.sub_block(kind, l, h, S=S, pattern=pattern, gains=gains32, zeta=zeta32)
            outs.append(h)
        return tuple(outs)

    with torch.autocast(device_type=h0.device.type, enabled=False):
        H = h0.detach().float()
        acc_abs = torch.zeros(len(keys))
        acc_rel = torch.zeros(len(keys))
        for _ in range(n_delta):
            delta = torch.randn(H.shape, generator=generator, dtype=torch.float32).to(H.device)
            outs, tangents = _jvp(f, H, delta)
            prev_h, prev_t = H, delta
            for k, (h_k, t_k) in enumerate(zip(outs, tangents)):
                if t_k.dtype != torch.float32:
                    raise TypeError(f"the probe must run in fp32, got {t_k.dtype}")
                # the floors keep a degenerate window (a dead tangent, an all-zero stream) from
                # writing +-inf into the profile, which R3 runs with the zeta rule off and so
                # would record silently rather than raise (11 §P-C.33)
                lam = 0.5 * (torch.log(t_k.square().sum().clamp_min(TINY_NORM))
                             - torch.log(prev_t.square().sum().clamp_min(TINY_NORM)))
                growth = 0.5 * (torch.log(h_k.square().mean().clamp_min(TINY_NORM))
                                - torch.log(prev_h.square().mean().clamp_min(TINY_NORM)))
                acc_abs[k] += float(lam)
                acc_rel[k] += float(lam - growth)
                prev_h, prev_t = h_k, t_k
    return {"keys": keys, "abs": (acc_abs / n_delta).tolist(), "rel": (acc_rel / n_delta).tolist()}


def probe(backbone: Backbone, x: torch.Tensor, *, states: Optional[list] = None,
          pattern: Optional[SwitchPattern] = None, gains: Optional[dict] = None,
          zeta: Optional[torch.Tensor] = None, n_delta: int = N_DELTA,
          generator: Optional[torch.Generator] = None) -> dict:
    """The profile on the probe batch `x (n_pr, T, d_x)` (perception features of the episode's
    windows) with their entering states `states[i]` (a dict per window, or None for a fresh
    state): `{"mix": {l: lambda_rel}, "ffn": {...}, "mix_abs", "ffn_abs", "Lambda", "Lambda_abs",
    "Lambda_se"}`, the relative form being what the rule and the band use. SPEC: 07 §7.1-7.2."""
    cfg = backbone.cfg
    pattern = pattern or SwitchPattern.adult(cfg)
    was = backbone.training
    backbone.eval()
    try:
        n_pr = x.shape[0]
        per_window = []
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            gains32 = None if gains is None else {k: v.detach().float() for k, v in gains.items()}
            h0_all = torch.nn.functional.linear(x.detach().float(), backbone._eff("IN_W", gains32).float(),
                                                backbone.W["IN_b"].float())
        for i in range(n_pr):
            st = None if states is None else states[i]
            per_window.append(stack_exponents(backbone, h0_all[i:i + 1], st, pattern=pattern, gains=gains,
                                              zeta=zeta, n_delta=n_delta, generator=generator))
        keys = per_window[0]["keys"]
        rel_all = torch.tensor([w["rel"] for w in per_window])          # (n_pr, K_a)
        abs_all = torch.tensor([w["abs"] for w in per_window])
        abs_m, rel_m = abs_all.mean(0), rel_all.mean(0)
        rel_se = (rel_all.std(0, unbiased=True) / math.sqrt(n_pr)) if n_pr > 1 else torch.full_like(rel_m, float("nan"))
        lam_tot = rel_all.sum(1)
        out = {"mix": {}, "ffn": {}, "mix_abs": {}, "ffn_abs": {}, "mix_se": {}, "ffn_se": {}}
        for k, (kind, l) in enumerate(keys):
            out[kind][l] = float(rel_m[k])
            out[kind + "_abs"][l] = float(abs_m[k])
            out[kind + "_se"][l] = float(rel_se[k])
        out["n_blocks"] = len(keys)
        out["Lambda"] = float(rel_m.sum())
        out["Lambda_abs"] = float(abs_m.sum())
        out["Lambda_se"] = float(lam_tot.std(unbiased=True) / math.sqrt(n_pr)) if n_pr > 1 else float("nan")
        out["lambda_block"] = out["Lambda"] / max(1, len(keys))        # the depth-intensive form
        out["lambda_block_se"] = out["Lambda_se"] / max(1, len(keys))
        out["n_probe_windows"] = n_pr
        return out
    finally:
        backbone.train(was)


def zeta_update(zeta: torch.Tensor, profile: dict, *, kappa: float = KAPPA, clip: float = LAMBDA_CLIP,
                clamp: float = ZETA_CLAMP) -> torch.Tensor:
    """(7.4) on every probed sub-block's *relative* exponent; unprobed entries unchanged; a
    non-finite exponent raises rather than moving the gain."""
    z = zeta.detach().clone().float()
    for kind, idx in (("mix", 0), ("ffn", 1)):
        for l, lam in profile[kind].items():
            if not math.isfinite(float(lam)):
                raise FloatingPointError(f"non-finite exponent at {kind} {l}")
            j = 2 * int(l) + idx
            # only the entries the probe reached move, and only they are clamped: 07 §7.3 says
            # sub-blocks the probe did not reach keep their `zeta` (11 §P-C.33)
            z[j] = float(min(clamp, max(-clamp, float(z[j]) - kappa * max(-clip, min(clip, float(lam))))))
    return z


def trigger(profile: dict, band: float = BAND, se_gate: float = SE_GATE) -> dict:
    """Whether one consolidation's profile fires the K-CRIT trigger of 07 §7.6, and why.

    Both limbs are on **per-block** relative exponents, and both are gated by the standard
    error: a displacement past the band smaller than `se_gate` standard errors is not a
    displacement. Returns the two limbs, the displacements in units of their own standard
    error, and the verdict, so that a record can be read without recomputing it.
    """
    n = int(profile.get("n_blocks") or (len(profile.get("mix", {})) + len(profile.get("ffn", {}))) or 1)
    lam_block = profile.get("lambda_block")
    if lam_block is None:
        lam_block = float(profile.get("Lambda", 0.0)) / n
    se_block = profile.get("lambda_block_se")
    if se_block is None:
        se = float(profile.get("Lambda_se", float("nan")))
        se_block = se / n if se == se else float("nan")
    blocks = {(kind, l): float(v) for kind in ("mix", "ffn") for l, v in profile.get(kind, {}).items()}
    ses = {(kind, l): float(v) for kind in ("mix", "ffn") for l, v in profile.get(kind + "_se", {}).items()}
    worst, worst_key = 0.0, None
    for key, v in blocks.items():
        if abs(v) > worst:
            worst, worst_key = abs(v), key

    def excess(value: float, se: float) -> float:
        """The displacement past the band, in units of its own standard error."""
        d = abs(value) - band
        if d <= 0:
            return 0.0
        if not (se == se) or se <= 0:
            return float("inf")                       # no error bar: the displacement stands
        return d / se

    e_stack = excess(float(lam_block), float(se_block))
    e_block = excess(worst, float(ses.get(worst_key, float("nan")))) if worst_key else 0.0
    fired = e_stack > se_gate or e_block > se_gate
    return {"fires": bool(fired), "lambda_block": float(lam_block), "lambda_block_se": float(se_block),
            "worst_block": worst, "worst_block_key": None if worst_key is None else list(worst_key),
            "stack_excess_se": e_stack, "block_excess_se": e_block, "band": band, "se_gate": se_gate,
            "n_blocks": n, "Lambda": float(profile.get("Lambda", float("nan")))}


def needs_control(profiles: list[dict], band: float = BAND, se_gate: float = SE_GATE) -> bool:
    """K-CRIT is evaluated only if some consolidation's profile fires `trigger` (07 §7.6)."""
    return any(trigger(p, band, se_gate)["fires"] for p in profiles)
