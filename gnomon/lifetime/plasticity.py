"""The decoded plasticity rule: the lifetime's update, chosen by the genome.

SPEC: 01 §1.9
SPEC: 10 §10.2
SPEC: 10 §10.3

```
    ghat = g / (sqrt(v) + eps)                     scale-free error
    mhat = m / (sqrt(v) + eps)                     scale-free eligibility trace
    nu   = tanh( (Lbar - L) / (sigma_L + eps) )    the neuromodulator
    dz_i = - lr * [ A_i ghat_i + B_i mhat_i + C_i z_i + D_i nu mhat_i ]
```

`A, B, C, D` are decoded per identity from `Gamma`. **AdamW is the point
`A=0, B=1, C=0, D=0`**, so the rule the author would otherwise have imposed lies
inside the space the genome searches rather than outside it; §10.6's P1 proves
the identity numerically. Every coefficient and `nu` is detached: a lifetime must
not be able to minimise its loss by switching its own plasticity off.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

__all__ = ["N_RULE", "Neuromodulator", "PlasticState", "DecodedRule",
           "coefficients_from", "ADAM_POINT"]

# SPEC: 10 §10.3 -- (A, B, C, D). Fixed: it is a column count in a decoded role.
N_RULE = 4

# SPEC: 10 §10.3 -- the ranges. `B` is centred on 1 so an uninformative genome
# performs Adam exactly and evolution has to earn any departure from it.
A_SCALE, B_SCALE, C_SCALE, D_SCALE = 1.0, 1.0, 0.1, 1.0

#: The coefficients that make §10.2's rule identical to AdamW.
ADAM_POINT = (0.0, 1.0, 0.0, 0.0)


def coefficients_from(raw: torch.Tensor) -> torch.Tensor:
    """`(n, 4)` decoded reals -> `(n, 4)` rule coefficients. SPEC: 10 §10.3."""
    t = torch.tanh(raw)
    return torch.stack([
        A_SCALE * t[:, 0],
        1.0 + B_SCALE * t[:, 1],
        C_SCALE * t[:, 2],
        D_SCALE * t[:, 3],
    ], dim=-1)


class Neuromodulator:
    """`nu = tanh((Lbar - L) / sigma_L)`: is the model beating its recent self?

    SPEC: 10 §10.2. A reward-prediction error, global across channels, detached,
    and bounded in `(-1, 1)` so a single anomalous batch cannot dominate an
    update. Gating plasticity on one is the defining feature of three-factor
    learning rules (Frémaux & Gerstner 2016).
    """

    def __init__(self, rho: float = 0.99, eps: float = 1e-8):
        self.rho, self.eps = rho, eps
        self.mean: Optional[float] = None
        self.var = 0.0
        self.n = 0

    def update(self, loss: float) -> float:
        self.n += 1
        if self.mean is None:
            self.mean = loss
            return 0.0
        dev = loss - self.mean
        self.var = self.rho * self.var + (1.0 - self.rho) * dev * dev
        self.mean = self.rho * self.mean + (1.0 - self.rho) * loss
        sd = math.sqrt(self.var)
        # Before the baseline has any spread there is no prediction to err
        # against, so `nu` is exactly zero rather than an artefact of `eps`.
        if sd < 1e-6 or self.n < 8:
            return 0.0
        return math.tanh(dev * -1.0 / (sd + self.eps))

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "n": self.n}

    def load(self, st: dict) -> "Neuromodulator":
        """Restore. SPEC: 05 §5.7 L8 -- a resumed lifetime must match exactly.

        `nu` is a function of the running baseline, so a resume that restarted
        the baseline would take a different update on its very first step and
        the two trajectories would separate immediately.
        """
        self.mean, self.var, self.n = st["mean"], st["var"], st["n"]
        return self


@dataclasses.dataclass
class PlasticState:
    """The two traces the rule keeps, per acquired-identity tensor."""

    m: torch.Tensor
    v: torch.Tensor

    @staticmethod
    def zeros_like(p: torch.Tensor) -> "PlasticState":
        return PlasticState(torch.zeros_like(p), torch.zeros_like(p))


class DecodedRule:
    """Applies §10.2's update. Holds only traces; the rule itself is decoded.

    Deliberately *not* a `torch.optim.Optimizer` subclass: the coefficients
    change every step because they are decoded from a `Gamma` and a `Z` that
    both move, and an Optimizer's `param_groups` are the wrong home for
    something that is an output of the model rather than a setting of the run.
    """

    def __init__(self, params: dict[str, torch.Tensor], *,
                 beta_1: float = 0.9, beta_2: float = 0.95, eps: float = 1e-8,
                 rho: float = 0.99, weight_decay: float = 0.0,
                 max_identity_rms: Optional[float] = None):
        """`weight_decay` is §10.2's `C` term with a floor. SPEC: 05 §5.4.

        §05.4 specifies decoupled weight decay **0**, and `R-ADAM` pins `C = 0`
        (`ADAM_POINT`), so in the default arm *nothing* bounds an identity's
        norm. That is not a conservative choice, it is an unbounded one, and it
        is what destroyed the 40,000-step run: `z` reached `inf`, `zeta` grew
        188,401x, the decoder's first layer went from 32% responsive to 0.1%,
        and the decoded `erank` collapsed 38-fold to 0.0065 of `min(d)`.

        The mechanism is specific to learning *identities behind a saturating
        map*, which is what makes it GNOMON's problem and not the PNAS paper's.
        Adam's update is `lr * m_hat / (sqrt(v_hat) + eps)`, which is `O(lr)`
        **regardless of gradient magnitude**. As `||z||` grows, `tanh` saturates
        and `dL/dz -> 0` -- but Adam keeps taking full-size steps, so `z` drifts
        further out, saturates harder, and the loop closes. A vanishing gradient
        is normally self-limiting; under Adam it is self-reinforcing.

        Two bounds are offered because they are different claims:

        ``weight_decay``       a restoring force proportional to the identity.
                               Implemented by adding to the decoded `C`, which
                               makes the relationship exact: weight decay *is*
                               §10.2's homeostasis term with a floor under it.
        ``max_identity_rms``   a hard cap: rescale any identity row whose RMS
                               exceeds it. Bounds the worst case rather than
                               the expectation, and leaves everything inside
                               the ball untouched.
        """
        self.params = params
        self.beta_1, self.beta_2, self.eps = beta_1, beta_2, eps
        self.weight_decay = float(weight_decay)
        self.max_identity_rms = max_identity_rms
        self.state = {k: PlasticState.zeros_like(p) for k, p in params.items()}
        self.nu = Neuromodulator(rho=rho)
        self.t = 0
        self.last_nu = 0.0
        self.last_terms: dict[str, float] = {}

    @torch.no_grad()
    def step(self, coeffs: dict[str, torch.Tensor], loss: float,
             lr: float) -> None:
        """One update. `coeffs[k]` is `(n_k, N_RULE)`, detached, from the decode."""
        self.t += 1
        nu = self.nu.update(loss)
        self.last_nu = nu
        b1, b2 = self.beta_1, self.beta_2
        # Bias correction, exactly as Adam's: without it the first steps are
        # scaled by `(1 - b1^t)/sqrt(1 - b2^t)` and the rule would not reduce to
        # AdamW at `ADAM_POINT`, which is the property P1 exists to guarantee.
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t

        for name, p in self.params.items():
            if p.numel() == 0 or p.grad is None:
                continue
            st = self.state[name]
            g = p.grad
            st.m.mul_(b1).add_(g, alpha=1.0 - b1)
            st.v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            denom = (st.v / c2).sqrt_().add_(self.eps)
            ghat = g / denom
            mhat = (st.m / c1) / denom

            c = coeffs[name]
            A, B, C, D = (c[:, 0:1], c[:, 1:2], c[:, 2:3], c[:, 3:4])
            # SPEC: 05 §5.4 -- `weight_decay` enters as a floor under the
            # decoded `C`, so the two are one mechanism rather than two.
            upd = (A * ghat + B * mhat + (C + self.weight_decay) * p
                   + D * (nu * mhat))
            p.add_(upd, alpha=-lr)
            if self.max_identity_rms is not None and p.dim() == 2:
                # Per row: an identity is a vector, and it is that vector's
                # length that drives the decoder into saturation.
                rms = p.pow(2).mean(dim=1, keepdim=True).sqrt()
                over = (rms > self.max_identity_rms).squeeze(1)
                if bool(over.any()):
                    p[over] *= (self.max_identity_rms / rms[over])
            self.last_terms[name] = float(upd.abs().mean())

    def state_dict(self) -> dict:
        """Everything a resume needs. SPEC: 05 §5.7 L8.

        The two traces *and* the step count: bias correction divides by
        `1 - beta^t`, so a resume that restarted `t` would take a step scaled by
        a factor of up to `1/(1-beta_1)` and the run would visibly kink.
        """
        return {
            "m": {k: v.m.detach().cpu().clone() for k, v in self.state.items()},
            "v": {k: v.v.detach().cpu().clone() for k, v in self.state.items()},
            "t": self.t, "nu": self.nu.state(), "last_nu": self.last_nu,
        }

    def load_state_dict(self, st: dict) -> "DecodedRule":
        for k, ps in self.state.items():
            ps.m.copy_(st["m"][k].to(ps.m.device, ps.m.dtype))
            ps.v.copy_(st["v"][k].to(ps.v.device, ps.v.dtype))
        self.t = int(st["t"])
        self.nu.load(st["nu"])
        self.last_nu = float(st["last_nu"])
        return self

    def contributions(self, coeffs: dict[str, torch.Tensor]) -> dict[str, float]:
        """Mean |coefficient| per term, for the `A-RULE` report. SPEC: 10 §10.6.

        Reported because §10.1's defect was never *measured*: if the genome
        settles at `A=0, B=1, C=0, D=0` then the decoded rule is AdamW wearing a
        hat, and that must be visible rather than inferred.
        """
        out: dict[str, float] = {}
        for name, c in coeffs.items():
            if c.numel() == 0:
                continue
            for j, term in enumerate("ABCD"):
                out[f"{name}.{term}"] = float(c[:, j].mean())
        out["nu"] = self.last_nu
        return out
