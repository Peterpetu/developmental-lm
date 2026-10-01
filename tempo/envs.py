"""The reward-bearing streams: the point-mass environment, the verifier tasks, and
the tau_1 policy-gradient term.

SPEC: 02 §2.6    (the group-normalised advantage with the DAPO clip, (2.12))
SPEC: 02 §2.9.2  (symbolic tasks over the reserved alphabet, exact verifiers)
SPEC: 02 §2.9.3  (the point-mass: dynamics (2.14), the manoeuvre variant)
SPEC: 10 G-ACT   (the dedicated MLP policy control)

The point-mass is a toy and is labelled one (11 §P-0). Everything is a
deterministic function of the supplied generators.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from gnomon.utils.shapes import check_shape
from soma.openended.archive import TaskArchive
from soma.openended.tasks import (
    BOS, EOS, PAD, SEP, Instance, TaskParams, generate, verify,
)

__all__ = [
    "PointMass", "MLPPolicy", "group_advantage", "grpo_loss", "SymbolicTaskStream",
    "pad_to_spans", "CLIP_LO", "CLIP_HI", "G_ROLLOUTS",
]

CLIP_LO, CLIP_HI = 0.2, 0.28
"""SPEC: 02 (2.12) -- DAPO's asymmetric clip."""
G_ROLLOUTS = 8
"""SPEC: 02 §2.6 -- rollouts per prompt or start state."""


# =============================================================================
# The point-mass. SPEC: 02 §2.9.3
# =============================================================================

class PointMass:
    """State `s = (p_x, p_y, v_x, v_y, q_x, q_y, o_x, o_y)`, dynamics (2.14) at 20 Hz.

    ```
        v <- v + (a - c v) dt,  |a|_inf <= 1;   p <- clip(p + v dt, -1, 1)
        r  = -||p - q|| - 0.01 ||a||^2 - 5 [ ||p - o|| < 0.15 ]
    ```
    `manoeuvre=True` moves the goal on a circle of radius 0.5 at 0.1 Hz about its
    initial position (kept inside `[-0.5, 0.5]^2` so the circle stays in the box).
    """

    DT, C, RADIUS, HORIZON = 0.05, 0.5, 0.15, 100
    RATE_HZ = 20.0
    D_STATE, D_ACTION = 8, 2

    def __init__(self, *, manoeuvre: bool = False, horizon: int = HORIZON):
        self.manoeuvre = manoeuvre
        self.horizon = int(horizon)

    def reset(self, n: int, generator: torch.Generator, device="cpu") -> torch.Tensor:
        """`n` start states: `p, q, o ~ U[-1, 1]^2` with `o` at least 0.3 from both."""
        p = torch.rand((n, 2), generator=generator) * 2 - 1
        q = torch.rand((n, 2), generator=generator) * 2 - 1
        if self.manoeuvre:
            q = q * 0.5
        o = torch.rand((n, 2), generator=generator) * 2 - 1
        for _ in range(256):
            d_q = (o - q).norm(dim=-1)
            bad = ((o - p).norm(dim=-1) < 0.3) | (d_q < 0.3)
            if self.manoeuvre:
                bad = bad | ((d_q > 0.2) & (d_q < 0.8))              # keep the whole orbit (radius 0.5) clear
            if not bad.any():
                break
            o = torch.where(bad[:, None], torch.rand((n, 2), generator=generator) * 2 - 1, o)
        s = torch.cat([p, torch.zeros(n, 2), q, o], dim=-1)
        return s.to(device)

    def goal_at(self, s: torch.Tensor, t: int) -> torch.Tensor:
        """The goal at tick `t` (the state's `q` is the centre under manoeuvre)."""
        q = s[:, 4:6]
        if not self.manoeuvre:
            return q
        ang = 2.0 * math.pi * 0.1 * t * self.DT
        return q + 0.5 * torch.tensor([math.cos(ang), math.sin(ang)], dtype=s.dtype, device=s.device)

    def step(self, s: torch.Tensor, a: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(s', r)` for a batch. SPEC: 02 (2.14)."""
        check_shape(s, [None, self.D_STATE], name="state")
        check_shape(a, [s.shape[0], self.D_ACTION], name="action")
        a = a.clamp(-1.0, 1.0)
        p, v = s[:, 0:2], s[:, 2:4]
        v2 = v + (a - self.C * v) * self.DT
        p2 = (p + v2 * self.DT).clamp(-1.0, 1.0)
        s2 = torch.cat([p2, v2, s[:, 4:8]], dim=-1)
        q = self.goal_at(s2, t + 1)
        o = s[:, 6:8]
        hit = ((p2 - o).norm(dim=-1) < self.RADIUS).to(s.dtype)
        r = -(p2 - q).norm(dim=-1) - 0.01 * a.square().sum(-1) - 5.0 * hit
        return s2, r

    def rollout(self, policy: Callable[[torch.Tensor, int], torch.Tensor], s0: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(states (n, H+1, 8), actions (n, H, 2), rewards (n, H))` from `s0` under
        `policy(obs, t)`, the policy seeing `observation(s, t)` (the goal as it is at `t`)."""
        n = s0.shape[0]
        states = [s0]
        actions, rewards = [], []
        s = s0
        for t in range(self.horizon):
            a = policy(self.observation(s, t), t)
            s, r = self.step(s, a, t)
            states.append(s)
            actions.append(a)
            rewards.append(r)
        return torch.stack(states, 1), torch.stack(actions, 1), torch.stack(rewards, 1)

    def observation(self, s: torch.Tensor, t: int) -> torch.Tensor:
        """The perceived proprioception: the state with the goal *as seen* at tick `t`."""
        q = self.goal_at(s, t)
        return torch.cat([s[:, 0:4], q, s[:, 6:8]], dim=-1)


class MLPPolicy(nn.Module):
    """The G-ACT control: `8 -> 64 -> 64 -> 2` Gaussian policy. SPEC: 10 §10.2."""

    def __init__(self, d_in: int = 8, d_out: int = 2, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.Tanh(), nn.Linear(width, width),
                                 nn.Tanh(), nn.Linear(width, d_out))
        self.log_std = nn.Parameter(torch.zeros(d_out))

    def mean(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu = self.mean(s).float()
        ls = self.log_std.clamp(-5.0, 2.0).float()
        z = (a.float() - mu) * torch.exp(-ls)
        return -(0.5 * z.square().sum(-1) + ls.sum() + 0.5 * a.shape[-1] * math.log(2 * math.pi))

    def sample(self, s: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        mu = self.mean(s)
        xi = torch.randn(mu.shape, generator=generator, dtype=torch.float32).to(mu.device, mu.dtype)
        return mu + torch.exp(self.log_std.clamp(-5.0, 2.0)).to(mu.dtype) * xi

    @torch.no_grad()
    def project(self) -> None:
        self.log_std.clamp_(-5.0, 2.0)


# =============================================================================
# The policy-gradient term. SPEC: 02 (2.12)
# =============================================================================

def group_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, bool]:
    """`A_i = (r_i - mean r) / (std r + eps)` over a group; `keep=False` when every
    reward is equal (dynamic sampling drops the group)."""
    check_shape(rewards, [None], name="rewards")
    r = rewards.float()
    std = r.std(unbiased=False)
    if r.numel() < 2 or std < 1e-8:
        return torch.zeros_like(r), False
    return (r - r.mean()) / (std + eps), True


def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantage: torch.Tensor,
              mask: torch.Tensor, *, clip_lo: float = CLIP_LO, clip_hi: float = CLIP_HI
              ) -> torch.Tensor:
    """`-(1/sum|o_i|) sum_i sum_t min(rho A_i, clip(rho, 1-lo, 1+hi) A_i)` over one or more
    groups. `logp_* (G, T)` per emitted unit (token or action), `advantage (G,)`,
    `mask (G, T)` selecting the agent's own emissions."""
    check_shape(logp_new, [None, None], name="logp_new")
    G, T = logp_new.shape
    check_shape(logp_old, [G, T], name="logp_old")
    check_shape(advantage, [G], name="advantage")
    check_shape(mask, [G, T], name="mask")
    rho = torch.exp(logp_new.float() - logp_old.float().detach())
    A = advantage.float()[:, None]
    term = torch.minimum(rho * A, rho.clamp(1.0 - clip_lo, 1.0 + clip_hi) * A)
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return -(term * m).sum() / denom


# =============================================================================
# The verifier tasks as a stream. SPEC: 02 §2.9.2
# =============================================================================

def pad_to_spans(tokens: list[int], mask: list[bool], K: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad with `PAD` (mask False) to a multiple of `K`; returns int64 / bool arrays."""
    n = len(tokens)
    m = (-n) % K
    toks = np.asarray(list(tokens) + [PAD] * m, dtype=np.int64)
    msk = np.asarray(list(mask) + [False] * m, dtype=bool)
    return toks, msk


class SymbolicTaskStream:
    """Instances from the SOMA families through the learnability-gated archive.

    `verify(prompt, candidate, params)` is the exact reward. `solve_rate` is
    supplied by the life (greedy decoding of the agent); until the archive is
    seeded, instances are drawn from the family lower bounds. SPEC: 02 §2.9.2.
    """

    RATE_HZ = 4.0

    def __init__(self, T_ctx: int, seed: int):
        self.archive = TaskArchive(T_ctx=int(T_ctx))
        self.rng = np.random.default_rng(int(seed))
        self._seeded = False

    def seed_archive(self, solve_rate: Callable[[TaskParams, int], float]) -> None:
        self.archive.seed(solve_rate, self.rng)
        self._seeded = True

    def archive_step(self, solve_rate: Callable[[TaskParams, int], float], n: int = 1) -> dict:
        if not self._seeded:
            raise RuntimeError("seed the archive first")
        return self.archive.step(solve_rate, self.rng, n)

    def sample_params(self, rng: Optional[np.random.Generator] = None) -> TaskParams:
        """One point of the curriculum. `rng` is supplied per draw by the stream so that which
        family a window carries is a function of the *order* seed and the draw index, not of a
        long-lived generator: a shared generator made the draw depend on how many draws had
        come before, so two arms of one contrast read different families and a window was not
        reproducible from its cursor (11 §P-C.35)."""
        r = self.rng if rng is None else rng
        if self._seeded and len(self.archive):
            return self.archive.sample_params(r)
        from soma.openended.tasks import family_lower_bounds
        lbs = family_lower_bounds()
        return lbs[int(r.integers(0, len(lbs)))]

    def sample_instance(self, params: Optional[TaskParams] = None) -> Instance:
        return generate(params or self.sample_params(), self.rng)

    @staticmethod
    def reward(inst: Instance, candidate: list[int]) -> int:
        """`verify(prompt, candidate, params) in {0, 1}`; the candidate is the emitted
        answer with `EOS` and everything after it removed."""
        cand = list(candidate)
        if EOS in cand:
            cand = cand[:cand.index(EOS)]
        return int(verify(inst.prompt, cand, inst.params))

    @staticmethod
    def prompt_tokens(inst: Instance) -> list[int]:
        """`[BOS] + prompt + [SEP]`: what the agent perceives before it answers."""
        return [BOS] + list(inst.prompt) + [SEP]
