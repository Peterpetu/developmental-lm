"""Consolidation: window summaries, the reverse-time consolidator, the reward, the
genome step, the estimators, the ES arm, salience and rehearsal.

SPEC: 06 §6.1   (window summaries r_w, (6.1))
SPEC: 06 §6.2   (the reverse read, the proposal, salience, (6.2)-(6.5))
SPEC: 06 §6.3   (the reward, (6.6)-(6.7))
SPEC: 06 §6.4   (the genome step: trust region, decay toward the prior, the bound, (6.8))
SPEC: 06 §6.5   (estimator 1: one REINFORCE step with an EMA baseline, (6.9))
SPEC: 06 §6.6   (estimator 2: the null probe, best-of-G, the elite regression, (6.10))
SPEC: 06 §6.8   (the evolution-strategies arm)
SPEC: 06 §6.9   (salience targets, the salience head, the rehearsal buffer, (6.11))

Second pass after the adversarial review (11 §P-C.28): the forward read-out is the
last-reached output; the proposal's scale is bounded so that a norm-tau step is the
typical draw and the trust region rarely binds; the elite regression trains the mixer
by one step on the current episode and the heads on cached read-outs; the ES step has
an explicit length; the rehearsal buffer is a uniform sample under the uniform arm and
falls back to uniform on a degenerate salience; the salience head alone is fit to the
salience targets. The orchestration (which candidate is kept, the null probe, the
boundary order update-then-propose) lives in `tempo/life.py`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anlage import ops as aops
from gnomon.utils.shapes import check_shape

__all__ = [
    "TAU_FACTOR", "LAMBDA_DEC", "ETA_2", "LAMBDA_KL", "G_PROPOSALS", "BASELINE_DECAY",
    "P_REHEARSAL", "N_PROBE_SET", "SIGMA_MIN", "SIGMA_BIAS", "ACCEPTED_MAX", "N_PROBE_STEPS",
    "WindowSummaries", "env_loss", "Consolidator", "trust_region", "genome_step", "kl_diag",
    "L_check", "reward", "Reinforce", "RestEM", "es_proposal", "salience_targets", "salience_fit",
    "Rehearsal",
]

TAU_FACTOR = 0.2
"""SPEC: 06 (6.8) -- `tau = 0.2 sqrt(d_g)`, the per-coordinate scale of a norm-tau step."""
LAMBDA_DEC = 0.01
ETA_2 = 1e-4
LAMBDA_KL = 0.1
G_PROPOSALS = 4
N_PROBE_STEPS = 100
"""SPEC: 06 §6.6 -- `n_probe = N / 10` in the protocol; the constant is the default at `N = 1000`."""
BASELINE_DECAY = 0.9
P_REHEARSAL = 0.05
N_PROBE_SET = 512
SIGMA_MIN = 0.02
"""SPEC: 06 (6.4) -- the proposal's per-coordinate scale is clamped to `[0.02, 0.2]` (the
upper bound is `tau / sqrt(d_g)`: a draw of typical norm `tau`; the lower bound prevents
the collapse of the elite regression, 11 §P-C.28)."""
SIGMA_BIAS = 3.892
"""SPEC: 06 (6.4) -- `sigmoid(3.892) = 0.98`, so at birth `sigma ~ 0.191`, near the top of
`[0.02, 0.2]` and inside it: the head starts where its gradient is alive."""
ACCEPTED_MAX = 8
"""SPEC: 06 §6.6 -- the accepted set holds the last eight pairs (their read-outs go stale)."""


# =============================================================================
# Window summaries. SPEC: 06 §6.1
# =============================================================================

def env_loss(returns: torch.Tensor, R_scale: float) -> torch.Tensor:
    """`ell_w = -return / R_scale` for environment windows (6.1)."""
    if R_scale <= 0:
        raise ValueError("R_scale > 0")
    return -returns.float() / R_scale


class WindowSummaries:
    """The per-window records of one episode, in arrival order, with identities."""

    def __init__(self, d: int, d_x: int, N: int):
        self.d, self.d_x, self.N = d, d_x, int(N)
        self.h_bar: list[torch.Tensor] = []
        self.ell: list[torch.Tensor] = []
        self.mu_bar: list[torch.Tensor] = []
        self.step: list[torch.Tensor] = []
        self.ident: list[tuple] = []
        self.stream: list[str] = []
        self.rehearsed: list[bool] = []
        self.is_return: list[bool] = []      # the row's ell_w is -return/R_scale, not a likelihood
        self.t0: list[float] = []            # the wall-clock second the window began (11 §P-D.12)

    def append(self, h_bar: torch.Tensor, ell: torch.Tensor, mu_bar: torch.Tensor, step: int,
               idents: list[tuple], streams: list[str], rehearsed: Optional[list[bool]] = None,
               is_return: Optional[list[bool]] = None, t0: Optional[list[float]] = None) -> None:
        n = h_bar.shape[0]
        check_shape(h_bar, [n, self.d], name="h_bar")
        check_shape(ell, [n], name="ell")
        check_shape(mu_bar, [n, self.d_x], name="mu_bar")
        if len(idents) != n or len(streams) != n:
            raise ValueError("one identity and one stream per window")
        self.h_bar.append(h_bar.detach().to(torch.bfloat16).cpu())
        self.ell.append(ell.detach().float().cpu())
        self.mu_bar.append(mu_bar.detach().to(torch.bfloat16).cpu())
        self.step.append(torch.full((n,), int(step), dtype=torch.long))
        self.ident.extend(idents)
        self.stream.extend(streams)
        self.rehearsed.extend(rehearsed if rehearsed is not None else [False] * n)
        self.is_return.extend(is_return if is_return is not None else [False] * n)
        self.t0.extend(t0 if t0 is not None else [0.0] * n)

    @property
    def n_windows(self) -> int:
        return len(self.ident)

    def inputs(self, device="cpu") -> torch.Tensor:
        """`[h_bar; ell; mu_bar; n/N]` `(n_w, d + 1 + d_x + 1)` fp32 in arrival order."""
        if not self.ident:
            raise ValueError("no summaries")
        h = torch.cat(self.h_bar).float()
        e = torch.cat(self.ell)[:, None]
        m = torch.cat(self.mu_bar).float()
        s = (torch.cat(self.step).float() / self.N)[:, None]
        return torch.cat([h, e, m, s], dim=1).to(device)

    def losses(self) -> torch.Tensor:
        return torch.cat(self.ell)

    def steps(self) -> torch.Tensor:
        return torch.cat(self.step)

    def rescale(self, stream: str, factor: float) -> None:
        """Multiply the stored `ell_w` of one stream (when its scale is set after the fact)."""
        self.rescale_rows(stream, factor, only_returns=False)

    def rescale_rows(self, stream: str, factor: float, *, only_returns: bool) -> None:
        """As `rescale`, but with `only_returns` the rows whose `ell_w` is a *return* rather
        than a likelihood. A stream can carry both -- a task window's supervised row is in nats
        and its rollout rows are returns -- and `R_scale` applies to the returns alone
        (06 §6.1, 11 §P-C.36)."""
        st = np.asarray([s == stream for s in self.stream])
        if only_returns:
            st = st & np.asarray(self.is_return)
        off = 0
        for e in self.ell:
            n = e.shape[0]
            sel = torch.from_numpy(st[off:off + n])
            e[sel] = e[sel] * factor
            off += n

    def mean_loss(self, *, last_frac: Optional[float] = None, stream: Optional[str] = None,
                  exclude_rehearsed: bool = True) -> float:
        """The mean of `ell_w`, optionally over the last `last_frac` of steps and one stream,
        excluding rehearsed windows by default (re-seen windows are not learning progress)."""
        ell, st = self.losses(), self.steps()
        keep = torch.ones(ell.shape[0], dtype=torch.bool)
        if last_frac is not None:
            keep &= st >= self.N - self.last_steps(last_frac)
        if stream is not None:
            keep &= torch.tensor([s == stream for s in self.stream])
        if exclude_rehearsed and any(self.rehearsed):
            keep &= ~torch.tensor(self.rehearsed)
        if not keep.any():
            raise ValueError("no windows selected")
        return float(ell[keep].mean())

    def last_steps(self, frac: float) -> int:
        """"The last `frac` of steps" is the last `max(1, round(frac N))` steps."""
        return max(1, int(round(float(frac) * self.N)))

    def streams_present(self) -> list[str]:
        return sorted(set(self.stream))

    def state_dict(self) -> dict:
        return {"d": self.d, "d_x": self.d_x, "N": self.N, "h_bar": self.h_bar, "ell": self.ell,
                "mu_bar": self.mu_bar, "step": self.step, "ident": self.ident, "stream": self.stream,
                "rehearsed": self.rehearsed, "is_return": self.is_return, "t0": self.t0}

    @classmethod
    def from_state_dict(cls, s: dict) -> "WindowSummaries":
        """The summaries live on the CPU (they are `N B` window records, not a computation);
        `torch.load(map_location=cuda)` moves every tensor in a checkpoint to the device, so
        they are brought back explicitly or the next `append` concatenates across devices."""
        out = cls(s["d"], s["d_x"], s["N"])
        out.h_bar = [t.cpu() for t in s["h_bar"]]
        out.ell = [t.cpu() for t in s["ell"]]
        out.mu_bar = [t.cpu() for t in s["mu_bar"]]
        out.step = [t.cpu() for t in s["step"]]
        out.ident, out.stream = list(s["ident"]), list(s["stream"])
        out.rehearsed = list(s.get("rehearsed", [False] * len(out.ident)))
        out.is_return = list(s.get("is_return", [False] * len(out.ident)))
        out.t0 = list(s.get("t0", [0.0] * len(out.ident)))
        return out


# =============================================================================
# The consolidator. SPEC: 06 §6.2
# =============================================================================

class Consolidator(nn.Module):
    """`psi = {W_r, Mixer_C, W_R, W_mu, W_sigma, w_s}`. SPEC: 06 (6.2)-(6.5)."""

    def __init__(self, d: int, d_x: int, d_g: int, *, d_c: int = 256, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int = 1024, chunk_size: int = 64, seed: int = 0):
        super().__init__()
        if d_c % n_heads:
            raise ValueError("d_c must be a multiple of n_heads")
        self.d, self.d_x, self.d_g, self.d_c = d, d_x, d_g, d_c
        self.n_layers, self.n_heads, self.d_head, self.chunk_size = n_layers, n_heads, d_c // n_heads, chunk_size
        self.log_sigma_max = math.log(TAU_FACTOR)                 # tau / sqrt(d_g) per coordinate
        self.log_sigma_min = math.log(SIGMA_MIN)
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        def mat(shape):
            return nn.Parameter(torch.randn(*shape, generator=g) * shape[1] ** -0.5)

        self.W_r = mat((d_c, d + 1 + d_x + 1))
        self.M = nn.ParameterDict()
        for l in range(n_layers):
            for r in ("Q", "K", "V", "OG"):
                self.M[f"L{l}_{r}"] = mat((d_c, d_c))
            self.M[f"L{l}_O"] = mat((d_c, d_c))
            self.M[f"L{l}_AB"] = mat((2 * n_heads, d_c))
            for r in ("CQ", "CK", "CV"):
                taps = torch.zeros(d_c, 4)
                taps[:, 0] = 1.0
                self.M[f"L{l}_{r}"] = nn.Parameter(taps)
            self.M[f"L{l}_GATE"] = mat((d_ff, d_c))
            self.M[f"L{l}_UP"] = mat((d_ff, d_c))
            self.M[f"L{l}_DOWN"] = mat((d_c, d_ff))
            self.M[f"L{l}_NMIX"] = nn.Parameter(torch.full((d_c,), (2.0 * n_layers) ** -0.5))
            self.M[f"L{l}_NFFN"] = nn.Parameter(torch.full((d_c,), (2.0 * n_layers) ** -0.5))
        self.W_R = mat((d_c, 2 * d_c + d_g))
        self.W_mu = nn.Parameter(torch.zeros(d_g, d_c))
        self.W_sigma = nn.Parameter(torch.zeros(d_g, d_c))
        self.b_sigma = nn.Parameter(torch.full((d_g,), SIGMA_BIAS))
        # 11 §P-D.8: the proposal scale is NOT learned. With `G` candidates and one elite per
        # boundary this is CEM at elite fraction 1/G, and CEM takes its scale from the spread of
        # the elite SET -- with a single elite there is no spread and nothing to estimate. Fitted
        # anyway, against a mean over-parameterised against the accepted set by 32x at the TS
        # shape, it collapses to SIGMA_MIN within one boundary and the policy goes deterministic
        # (measured: sigma 0.191 -> 0.020, the null probe winning 4 of 10 episodes). The head is
        # kept so the state dict is unchanged, and frozen at an initialisation that already gives
        # sigma = 0.1909, which is 06 (6.4)'s `tau / sqrt(d_g)` = 0.2: a typical draw of norm
        # `tau`, the trust region the step may take and so the right scale to explore it with.
        self.W_sigma.requires_grad_(False)
        self.b_sigma.requires_grad_(False)
        self.w_s = nn.Parameter(torch.zeros(d_c))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def mixer_parameters(self) -> list[nn.Parameter]:
        return [self.W_r] + list(self.M.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """The proposal's read-out and its **mean** head -- the parameters an estimator may fit.

        The scale head is deliberately absent (11 §P-D.8). An earlier correction added
        `b_sigma` here so that the elite regression could concentrate the policy, which it then
        did to the point of collapse: with one elite per boundary there is no spread to estimate
        a scale from, and the fit drives `sigma` to its floor within a single update. The scale
        is now a constant of the design rather than a quantity to be inferred from four samples.
        """
        return [self.W_R, self.W_mu]

    def scale_is_frozen(self) -> bool:
        """G-SIGMA: the scale head must not be trainable (11 §P-D.8)."""
        return not (self.W_sigma.requires_grad or self.b_sigma.requires_grad)

    # ------------------------------------------------------------------ mixer
    def mixer(self, r: torch.Tensor) -> torch.Tensor:
        """Four gated-delta layers (Peri-LN, as the backbone) over `(1, n, d_c)`; states reset."""
        x = r
        for l in range(self.n_layers):
            M = self.M
            h = aops.rms_norm_gf(x)
            y, _ = aops.delta_mixer(h, M[f"L{l}_Q"], M[f"L{l}_K"], M[f"L{l}_V"], M[f"L{l}_O"], M[f"L{l}_AB"],
                                    M[f"L{l}_OG"], M[f"L{l}_CQ"], M[f"L{l}_CK"], M[f"L{l}_CV"],
                                    n_h=self.n_heads, d_head=self.d_head, chunk_size=self.chunk_size)
            x = x + M[f"L{l}_NMIX"] * aops.rms_norm_gf(y)
            h = aops.rms_norm_gf(x)
            f = aops.swiglu(h, M[f"L{l}_GATE"], M[f"L{l}_UP"], M[f"L{l}_DOWN"])
            x = x + M[f"L{l}_NFFN"] * aops.rms_norm_gf(f)
        return aops.rms_norm_gf(x)

    def features(self, r_in: torch.Tensor, *, reverse: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """`(feat (2 d_c,), c (n_w, d_c))`: the read-out features `[c_last_reached; mean_w c_w]`
        and every window's output in the *original* order. SPEC: 06 (6.2)-(6.3).

        Reversed (the mechanism): the mixer reads windows last-to-first and the read-out is
        the first window's output, reached last. Forward (the K-REV control): first-to-last,
        the read-out the last window's output, reached last. Both: "final read-out + mean".
        """
        check_shape(r_in, [None, self.d + 1 + self.d_x + 1], name="r_in")
        r = F.linear(r_in.to(self.W_r.dtype), self.W_r)                    # (n_w, d_c)
        seq = r.flip(0) if reverse else r
        c_seq = self.mixer(seq[None])[0]
        c = c_seq.flip(0) if reverse else c_seq
        last_reached = c_seq[-1]
        return torch.cat([last_reached, c.mean(0)]), c

    def readout(self, feat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """`R = W_R [feat; g]` (6.3)."""
        check_shape(g, [self.d_g], name="g")
        return F.linear(torch.cat([feat, g.to(feat.dtype)]), self.W_R)

    def read(self, r_in: torch.Tensor, g: torch.Tensor, *, reverse: bool = True
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(R (d_c,), c (n_w, d_c))`."""
        feat, c = self.features(r_in, reverse=reverse)
        return self.readout(feat, g), c

    # ------------------------------------------------------------------ proposal
    def propose(self, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mu, log sigma)`, `sigma` in `(0.02, 0.2)` per coordinate. SPEC: 06 (6.4).

        The scale is a **squashed** affine map, not a clamped one:

            log sigma = log s_min + (log s_max - log s_min) * sigmoid(W_sigma R + b_sigma)

        A `clamp` was the first two forms of this head and it killed the head twice. With
        `W_sigma = 0` at birth the pre-activation is `0`, which is *above* `log 0.2`; `clamp`
        has zero gradient outside its bounds, so `W_sigma.grad` was exactly zero from every
        estimator and `sigma` sat at the upper bound for the whole life -- the scale head was
        dead by construction, again, in the very form written to revive it (11 §P-C.36). The
        sigmoid is differentiable everywhere, so the head can move in both directions from any
        starting point, and `b_sigma` is initialised so that birth is near the top of the range
        (`sigma ~ 0.19`), which is what (6.4) wants: a first draw of norm about `tau`.
        """
        mu = F.linear(R, self.W_mu)
        z = torch.sigmoid(F.linear(R, self.W_sigma) + self.b_sigma)
        log_sigma = self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * z
        return mu, log_sigma

    @staticmethod
    def sample(mu: torch.Tensor, log_sigma: torch.Tensor, generator: Optional[torch.Generator] = None
               ) -> torch.Tensor:
        xi = torch.randn(mu.shape, generator=generator, dtype=torch.float32).to(mu.device)
        return (mu + torch.exp(log_sigma) * xi).detach()

    @staticmethod
    def log_prob(dg: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        z = (dg.float() - mu.float()) * torch.exp(-log_sigma.float())
        return -(0.5 * z.square().sum() + log_sigma.float().sum() + 0.5 * dg.numel() * math.log(2 * math.pi))

    def salience(self, c: torch.Tensor) -> torch.Tensor:
        """`s_w = sigmoid(w_s^T c_w)` (6.5)."""
        return torch.sigmoid(F.linear(c, self.w_s[None]))[:, 0]


# =============================================================================
# The genome step and the reward. SPEC: 06 §6.3-6.4
# =============================================================================

def trust_region(dg: torch.Tensor, tau: float) -> torch.Tensor:
    """`Pi_tau(dg) = dg min(1, tau / ||dg||)`."""
    n = dg.norm()
    return dg * torch.clamp(tau / n.clamp_min(1e-12), max=1.0)


def genome_step(g: torch.Tensor, dg: torch.Tensor, *, tau: float, lambda_dec: float = LAMBDA_DEC,
                bound: float, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(6.8): `g_0 + P_bound[(1 - lambda_dec)(g + Pi_tau(dg) - g_0)]`: the trust-region step,
    the decay toward the prior, and the bound `||g - g_0|| <= bound`, all about `g_0`."""
    check_shape(dg, list(g.shape), name="dg")
    g0 = torch.zeros_like(g) if prior is None else prior.to(g)
    delta = (1.0 - lambda_dec) * (g + trust_region(dg, tau) - g0)
    n = delta.norm()
    return g0 + delta * torch.clamp(bound / n.clamp_min(1e-12), max=1.0)


def kl_diag(mu: torch.Tensor, ls: torch.Tensor, mu_old: torch.Tensor, ls_old: torch.Tensor) -> torch.Tensor:
    """`KL(N(mu, sigma^2) || N(mu_old, sigma_old^2))`, diagonal, closed form, summed."""
    var, var_old = torch.exp(2 * ls), torch.exp(2 * ls_old)
    return (ls_old - ls + (var + (mu - mu_old).square()) / (2 * var_old) - 0.5).sum()


def L_check(scores: dict[str, float], scales: dict[str, float]) -> float:
    """(6.6): the uniform mean over the streams present of each stream's fresh-state fitness
    score divided by its scale `s_m` (the standard deviation of that score's windows at the
    stream's first evaluation, floored), so that streams with incommensurable losses enter in
    comparable units (11 §P-C.29)."""
    if not scores:
        raise ValueError("no scores")
    return float(np.mean([scores[m] / max(scales.get(m, 1.0), 1e-6) for m in scores]))


def reward(L_check_e: float, L_check_next: float, F_next: float, F_ref: float) -> float:
    """(6.7): `(L_e - L_{e+1}) F_ref / F_{e+1}`: standardised improvement per primordium-episode of compute."""
    if F_next <= 0 or F_ref <= 0:
        raise ValueError("FLOPs must be positive")
    return (float(L_check_e) - float(L_check_next)) * float(F_ref) / float(F_next)


# =============================================================================
# Estimators. SPEC: 06 §6.5-6.6, §6.8
# =============================================================================

@dataclasses.dataclass
class Reinforce:
    """Estimator 1: one REINFORCE step per boundary with an EMA baseline, applied *before*
    the boundary's proposal is drawn so that every proposal is a sample of the policy whose
    gradient scores it (on-policy). The KL clause of the plan's (6.9) has zero gradient at a
    single step and is not applied (11 §P-C.18). SPEC: 06 (6.9)."""

    baseline: float = 0.0
    decay: float = BASELINE_DECAY

    def update(self, cons: Consolidator, opt: torch.optim.Optimizer, r_in: torch.Tensor, g: torch.Tensor,
               dg: torch.Tensor, r_e: float, *, reverse: bool = True) -> dict:
        adv = float(r_e) - self.baseline
        opt.zero_grad()
        R, _ = cons.read(r_in, g, reverse=reverse)
        mu, ls = cons.propose(R)
        lp = cons.log_prob(dg, mu, ls)
        loss = -adv * lp
        loss.backward()
        opt.step()
        self.baseline = self.decay * self.baseline + (1.0 - self.decay) * float(r_e)
        return {"log_prob": float(lp), "advantage": adv, "baseline": self.baseline}

    def state_dict(self) -> dict:
        return {"baseline": self.baseline}

    def load_state_dict(self, d: dict) -> None:
        self.baseline = float(d["baseline"])


@dataclasses.dataclass
class RestEM:
    """Estimator 2's policy update (06 §6.6): the accepted set `D` of the last `ACCEPTED_MAX`
    pairs `(feat, g, dg*)` and the elite regression: the mixer takes one gradient step through
    the current episode's read; the heads `W_R, W_mu, W_sigma` are regressed for `steps`
    on the cached features of every accepted pair with a KL penalty to the pre-update
    policy. This is the cross-entropy method's elite regression (elite fraction 1/G) on a
    short buffer, not ReST-EM, which refits from the base model on fresh samples."""

    accepted: list = dataclasses.field(default_factory=list)      # [(feat (2 d_c,), g (d_g,), dg (d_g,))]
    lambda_kl: float = LAMBDA_KL
    steps: int = 200
    lr: float = 1e-3

    @staticmethod
    def select(rewards: list[float]) -> int:
        return int(np.argmax(np.asarray(rewards, dtype=np.float64)))

    def _heads_fit(self, cons: Consolidator, pairs: list, olds: list) -> dict:
        opt = torch.optim.Adam(cons.head_parameters(), lr=self.lr)
        info = {}
        for _ in range(self.steps):
            opt.zero_grad()
            nll = torch.zeros((), device=cons.W_R.device)
            kl = torch.zeros((), device=cons.W_R.device)
            for (feat, g, dg), (mo, lo) in zip(pairs, olds):
                mu, ls = cons.propose(cons.readout(feat, g))
                nll = nll - cons.log_prob(dg, mu, ls)
                kl = kl + kl_diag(mu, ls, mo, lo)
            loss = nll + self.lambda_kl * kl
            loss.backward()
            opt.step()
            info = {"nll": float(nll), "kl": float(kl), "n_accepted": len(pairs)}
        return info

    def update(self, cons: Consolidator, r_in: torch.Tensor, g: torch.Tensor, dg_star: torch.Tensor,
               *, reverse: bool = True) -> dict:
        # one step through the mixer on the current pair (the only pair whose summaries exist)
        opt_m = torch.optim.Adam(cons.mixer_parameters() + cons.head_parameters(), lr=self.lr)
        opt_m.zero_grad()
        feat, _ = cons.features(r_in, reverse=reverse)
        mu, ls = cons.propose(cons.readout(feat, g))
        (-cons.log_prob(dg_star, mu, ls)).backward()
        opt_m.step()
        with torch.no_grad():
            feat, _ = cons.features(r_in, reverse=reverse)
        self.accepted.append((feat.detach().clone(), g.detach().clone(), dg_star.detach().clone()))
        self.accepted = self.accepted[-ACCEPTED_MAX:]
        with torch.no_grad():
            olds = [cons.propose(cons.readout(f, gg)) for f, gg, _ in self.accepted]
            olds = [(m.detach(), l.detach()) for m, l in olds]
        return self._heads_fit(cons, self.accepted, olds)

    def update_grpo(self, cons: Consolidator, r_in: torch.Tensor, g: torch.Tensor, cands: list[torch.Tensor],
                    rewards: list[float], *, reverse: bool = True, clip_lo: float = 0.2, clip_hi: float = 0.28,
                    lr: float = 1e-3) -> dict:
        """The registered ablation (arm (g)): the group-normalised advantage over the `G`
        candidates with the clipped ratio (02 (2.12)); one on-policy step (`rho = 1`, so the
        clip is inert at the first pass); no accepted set. SPEC: 06 §6.6."""
        r = torch.tensor(rewards, dtype=torch.float32)
        std = r.std(unbiased=False)
        if r.numel() < 2 or std < 1e-8:
            return {"skipped": True}
        adv = (r - r.mean()) / (std + 1e-6)
        with torch.no_grad():
            R0, _ = cons.read(r_in, g, reverse=reverse)
            mu_old, ls_old = cons.propose(R0)
            lp_old = torch.stack([cons.log_prob(dg, mu_old, ls_old) for dg in cands])
        opt = torch.optim.Adam([p for p in cons.parameters() if p.requires_grad], lr=lr)
        opt.zero_grad()
        R, _ = cons.read(r_in, g, reverse=reverse)
        mu, ls = cons.propose(R)
        lp = torch.stack([cons.log_prob(dg, mu, ls) for dg in cands])
        rho = torch.exp(lp - lp_old)
        term = torch.minimum(rho * adv, rho.clamp(1 - clip_lo, 1 + clip_hi) * adv)
        loss = -term.mean()
        loss.backward()
        opt.step()
        return {"loss": float(loss), "advantages": adv.tolist()}

    def log_likelihood(self, cons: Consolidator) -> float:
        with torch.no_grad():
            return float(sum(cons.log_prob(dg, *cons.propose(cons.readout(f, g))) for f, g, dg in self.accepted))

    def state_dict(self) -> dict:
        return {"accepted": [(f.cpu(), g.cpu(), dg.cpu()) for f, g, dg in self.accepted]}

    def load_state_dict(self, d: dict, device=None) -> None:
        """The accepted pairs are read back by `cons.readout`, so they must sit where the
        consolidator does; a checkpoint saved on the CPU and loaded on CUDA (or the reverse)
        would otherwise mix devices at the first regression step."""
        self.accepted = [(f, g, dg) if device is None else (f.to(device), g.to(device), dg.to(device))
                         for f, g, dg in d["accepted"]]


def es_proposal(eps: torch.Tensor, r_plus: torch.Tensor, r_minus: torch.Tensor, *, sigma_es: float,
                tau: float) -> torch.Tensor:
    """(6.8) arm (f): mirrored ES on `g` (Salimans et al. 2017, antithetic pairs). The
    estimator `g_hat = (1 / (2 n sigma_es)) sum_i (r_i^+ - r_i^-) eps_i` gives the direction;
    the step is of length `tau` along it, `dg = tau g_hat / ||g_hat||` (zero when every pair is
    tied), so the ES arm and the proposal head move the genome by the same distance."""
    check_shape(eps, [None, None], name="eps")
    n = eps.shape[0]
    check_shape(r_plus, [n], name="r_plus")
    check_shape(r_minus, [n], name="r_minus")
    g_hat = ((r_plus.float() - r_minus.float())[:, None] * eps.float()).sum(0) / (2.0 * n * sigma_es)
    norm = g_hat.norm()
    if float(norm) < 1e-12:
        return torch.zeros_like(g_hat)
    return tau * g_hat / norm


# =============================================================================
# Salience and rehearsal. SPEC: 06 §6.9
# =============================================================================

def salience_targets(delta_ell: torch.Tensor) -> torch.Tensor:
    """`sigma(delta_ell / sigma_ell)`: a logistic z-score of the measured improvement, in (0, 1)."""
    d = delta_ell.float()
    s = d.std(unbiased=False).clamp_min(1e-8)
    return torch.sigmoid(d / s)


def salience_fit(cons: Consolidator, c: torch.Tensor, targets: torch.Tensor, *, steps: int = 200,
                 lr: float = 1e-2) -> float:
    """(6.11): fit the salience head `w_s` alone on cached window outputs `c (n, d_c)`; the
    mixer is not moved by this term (it would otherwise move the policy off its samples,
    11 §P-C.28). Returns the final mean squared error."""
    check_shape(targets, [c.shape[0]], name="targets")
    opt = torch.optim.Adam([cons.w_s], lr=lr)
    loss = torch.zeros(())
    for _ in range(steps):
        opt.zero_grad()
        loss = (cons.salience(c) - targets).square().mean()
        loss.backward()
        opt.step()
    return float(loss)


class Rehearsal:
    """The rehearsal buffer of one episode: `n_reh` windows, the top by salience or a uniform
    sample, drawn without replacement `∝ s_w` (uniformly under the uniform arm). SPEC: 06 §6.9."""

    def __init__(self, n_reh: int):
        self.n_reh = int(n_reh)
        self.ident: list[tuple] = []
        self.t0: list[float] = []            # each identity's wall clock (11 §P-D.12)
        self.s: torch.Tensor = torch.zeros(0)
        self.mode: str = "salience"

    def build(self, idents: list[tuple], s: Optional[torch.Tensor], *, generator: Optional[torch.Generator] = None,
              uniform: bool = False, t0: Optional[list[float]] = None) -> None:
        n = len(idents)
        t0s = list(t0) if t0 is not None else [0.0] * n
        k = min(self.n_reh, n)
        # a *constant* salience is as degenerate as a zero one: at birth `w_s = 0` gives
        # `s_w = 0.5` everywhere, and `topk` on a constant returns the first `k` indices, so the
        # buffer would be the episode's earliest windows recorded as if it were ranked
        # (11 §P-C.36)
        degenerate = (s is None or not torch.isfinite(s.float()).all()
                      or float(s.float().sum()) <= 0.0
                      or float(s.float().max() - s.float().min()) < 1e-6)
        if uniform or degenerate:
            perm = torch.randperm(n, generator=generator)[:k]
            self.ident = [idents[int(i)] for i in perm]
            self.t0 = [t0s[int(i)] for i in perm]
            self.s = torch.ones(k)
            self.mode = "uniform"
            return
        check_shape(s, [n], name="s")
        top = torch.topk(s.float(), k).indices
        self.ident = [idents[int(i)] for i in top]
        self.t0 = [t0s[int(i)] for i in top]
        self.s = s.float()[top].clone()
        self.mode = "salience"

    def __len__(self) -> int:
        return len(self.ident)

    def draw(self, n: int, generator: Optional[torch.Generator] = None) -> list[tuple]:
        """`n` distinct `(ident, t0)` pairs with probability `∝ s_w` (uniform in the uniform
        mode). The `t0` travels with the identity because a replayed window must be presented at
        the wall clock it actually had -- 02 (2.13) makes the time code a function of it, and a
        replay at `t = 0` is a different input (11 §P-D.12)."""
        if not self.ident:
            raise ValueError("empty buffer")
        n = min(n, len(self.ident))
        idx = torch.multinomial(self.s, n, replacement=False, generator=generator)
        return [(self.ident[int(i)], self.t0[int(i)] if int(i) < len(self.t0) else 0.0) for i in idx]

    def state_dict(self) -> dict:
        return {"n_reh": self.n_reh, "ident": self.ident, "s": self.s, "mode": self.mode,
                "t0": self.t0}

    def load_state_dict(self, d: dict) -> None:
        self.n_reh, self.ident = int(d["n_reh"]), list(d["ident"])
        self.t0 = list(d.get("t0", [0.0] * len(self.ident)))
        self.s = d["s"].cpu()                      # drawn with a CPU generator (08 §8.5)
        self.mode = d.get("mode", "salience")
