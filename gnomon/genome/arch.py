"""`Gamma_a`: the head that decodes the computation graph.

SPEC: 01 §1.4
SPEC: 07 §7.1
SPEC: 07 §7.2
SPEC: 07 §7.3
SPEC: 07 §7.4
SPEC: 07 §7.5

C3 of the constitution. Mutating `Gamma` changes the architecture, so
architecture search and genome search are one search over one object -- there is
no separate NAS loop, no outer level, and no repair operator.

The head is **total** (§7.3): every point in `R^{|Gamma_a|}` decodes to an
architecture that builds and runs. `argmax` over a fixed menu cannot produce an
out-of-range value, so the search never has to handle an invalid individual, and
there is no second undocumented piece of design hiding in a constraint handler.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GenomeShape, PhenoConfig
from ..pheno.plan import Arch
from . import codes as C

__all__ = ["ArchHead", "Strategy", "SIGMA_FLOOR", "MENU_L", "MENU_H", "MENU_F", "MENU_W",
           "D_A", "D_GLOB", "mutate", "crossover"]

# SPEC: 07 §7.1 -- the menus, fixed for the life of the project.
MENU_L = (2, 3, 4, 6, 8, 12)                 # depth
MENU_H = (0.25, 0.5, 1.0, 2.0)               # multiples of the config's n_h
MENU_F = (1, 2, 4, 8)                        # multiples of d_model
MENU_W = (0.25, 0.5, 1.0, math.inf)          # multiples of T
D_A = 16
D_GLOB = 8


class ArchHead(nn.Module):
    """`Gamma_a`. SPEC: 07 §7.2 -- these tensors and nothing else."""

    def __init__(self, shape: GenomeShape, *, seed: int = 0):
        super().__init__()
        self.shape = shape
        g = torch.Generator(device="cpu").manual_seed(seed ^ 0xA2C4)

        def normal(*sz, fan: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(*sz, generator=g) / math.sqrt(fan))

        n_in = shape.n_L + D_GLOB
        self.A_1 = normal(D_A, n_in, fan=n_in)
        self.a_c1 = nn.Parameter(torch.zeros(D_A))
        self.A_2 = normal(D_A, D_A, fan=D_A)
        self.a_c2 = nn.Parameter(torch.zeros(D_A))
        self.A_op = normal(2, D_A, fan=D_A)
        self.b_op = nn.Parameter(torch.zeros(2))
        self.A_h = normal(len(MENU_H), D_A, fan=D_A)
        self.b_h = nn.Parameter(torch.zeros(len(MENU_H)))
        self.A_f = normal(len(MENU_F), D_A, fan=D_A)
        self.b_f = nn.Parameter(torch.zeros(len(MENU_F)))
        self.A_w = normal(len(MENU_W), D_A, fan=D_A)
        self.b_w = nn.Parameter(torch.zeros(len(MENU_W)))
        self.A_d = normal(len(MENU_L), D_GLOB, fan=D_GLOB)
        self.b_d = nn.Parameter(torch.zeros(len(MENU_L)))
        self.g_glob = nn.Parameter(torch.randn(D_GLOB, generator=g))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def decode_arch(self, cfg: PhenoConfig) -> Arch:
        """SPEC: 07 §7.2, §7.3. Total: every genome yields a valid architecture.

        Reads `r_l` and `g_glob` only, never the soma (§7.1): a lifetime cannot
        change the architecture it was born with.
        """
        dev, dt = self.A_1.device, self.A_1.dtype
        L = min(MENU_L[int(torch.argmax(self.A_d @ self.g_glob + self.b_d))],
                cfg.L)
        r = C.layer_codes(L, self.shape, dev).to(dt)
        glob = self.g_glob.unsqueeze(0).expand(L, -1)
        a1 = torch.tanh(F.linear(torch.cat([r, glob], -1), self.A_1, self.a_c1))
        a2 = torch.tanh(F.linear(a1, self.A_2, self.a_c2))

        def pick(A, b, menu):
            return [menu[int(i)] for i in torch.argmax(F.linear(a2, A, b), -1)]

        ops = ["attn" if i == 0 else "delta"
               for i in torch.argmax(F.linear(a2, self.A_op, self.b_op), -1)]
        # SPEC: 07 §7.3 -- clamping is saturation to the nearest admissible menu
        # entry, never rejection: a search whose mutants can be invalid spends
        # its budget on repair.
        n_h = [_clamp_heads(int(round(m * cfg.n_h)), cfg, self.shape)
               for m in pick(self.A_h, self.b_h, MENU_H)]
        d_ff = [min(m * cfg.d_model, self.shape.d_max)
                for m in pick(self.A_f, self.b_f, MENU_F)]
        win = [cfg.T if math.isinf(m) else max(1, int(m * cfg.T))
               for m in pick(self.A_w, self.b_w, MENU_W)]
        return Arch(tuple(ops), tuple(n_h), tuple(d_ff), tuple(win))


def _clamp_heads(n: int, cfg: PhenoConfig, shape: GenomeShape) -> int:
    """Admissible head counts: a multiple of `n_kv`, addressable, at least 1."""
    n = max(cfg.n_kv, n - n % cfg.n_kv)
    while n * cfg.d_head > shape.d_max and n > cfg.n_kv:
        n -= cfg.n_kv
    return n


# --------------------------------------------------------------- variation

# SPEC: 07 §7.5. The floor below which `sigma` stops tracking a tensor's scale,
# so that a zero-initialised tensor -- `c1`, `c2`, `w_sig`, `sig_0`, `o_0`, and
# every bias in `Gamma_a` -- can be reached at all.
#
# It was `1.0` in the first implementation, and since **every** genome tensor has
# rms below 1 at initialisation, that made the perturbation absolute for all of
# them: the floor swallowed the very scaling it existed to protect. Found by
# `G-MUTATE` M17 once §07.6's T10 was corrected to divide by `rms` rather than by
# `max(rms, 1)`. At 0.1 a zero tensor moves by `0.015` per mutation, reaching
# O(0.1) over the tens of generations §08.3 budgets.
SIGMA_FLOOR = 0.1


def _tensor_sigma(t: torch.Tensor, sigma_rel: float) -> float:
    """SPEC: 07 §7.5. Scale-relative above `SIGMA_FLOOR`, absolute below it."""
    rms = float(t.pow(2).mean().sqrt()) if t.numel() else 0.0
    return sigma_rel * max(rms, SIGMA_FLOOR)


class Strategy(nn.Module):
    """Self-adaptive mutation strengths: one `log_sigma` per genome tensor.

    SPEC: 10 §10.4. The germline half of §10's principle -- the mathematics of
    change is itself under selection. `sigma_rel = 0.15` was a number the author
    chose; here the genome carries its own strengths and they are selected along
    with everything else (Rechenberg 1973; Schwefel 1977; the diagonal case of
    Wierstra et al., *Natural Evolution Strategies*, JMLR 15:949-980, 2014).

    **Per tensor, not per coordinate.** A per-coordinate strength would add
    `|Gamma|` reals and double the genome for a second-order effect. Per tensor
    it is 27 reals against 5,279, and it captures what §07.5 actually measured:
    tensors drift apart in scale across generations, so one global `sigma_rel`
    is then wrong for all of them.
    """

    def __init__(self, names):
        super().__init__()
        self.names = list(names)
        # Zero: generation 0 mutates exactly as §07.5 specifies, so
        # self-adaptation is a departure evolution has to earn -- the same
        # discipline as centring the plasticity rule on Adam (§10.3).
        self.log_sigma = nn.Parameter(torch.zeros(len(self.names)))

    def n_params(self) -> int:
        return self.log_sigma.numel()

    def index_of(self, name: str) -> int:
        return self.names.index(name)


def _tau(n: int) -> float:
    """Schwefel's rate for a self-adapted strength: `1 / sqrt(2 sqrt(n))`."""
    return 1.0 / math.sqrt(2.0 * math.sqrt(max(n, 1)))


@torch.no_grad()
def mutate(modules: list[nn.Module], *, sigma_rel: float = 0.15,
           p_mut: float = 0.25, generator: Optional[torch.Generator] = None,
           strategy=None) -> None:
    """In-place Gaussian mutation of every genome tensor. SPEC: 07 §7.5, 10 §10.4.

    With a `strategy`, the strengths are **mutated first and then used**. That
    ordering is what makes self-adaptation work: a genome is evaluated with the
    strengths it will pass on, so a strength that produced good offspring is
    selected together with them. Applying the old strength and storing a new one
    severs that link and the strengths drift at random -- §10.6's P6 plants
    exactly that defect and requires the suite to reject it.
    """
    if strategy is not None:
        tau = _tau(strategy.log_sigma.numel())
        strategy.log_sigma.add_(
            torch.randn(strategy.log_sigma.shape, generator=generator) * tau)
        strategy.log_sigma.clamp_(-3.0, 3.0)

    for mod in modules:
        for name, p in mod.named_parameters():
            if p.numel() == 0:
                continue
            scale = sigma_rel
            if strategy is not None and name in strategy.names:
                scale = sigma_rel * float(
                    torch.exp(strategy.log_sigma[strategy.index_of(name)]))
            noise = torch.randn(p.shape, generator=generator,
                                device="cpu").to(p.device, p.dtype)
            mask = (torch.rand(p.shape, generator=generator, device="cpu")
                    < p_mut).to(p.device)
            p.add_(noise * _tensor_sigma(p, scale) * mask)


@torch.no_grad()
def crossover(child: list[nn.Module], a: list[nn.Module], b: list[nn.Module],
              *, generator: Optional[torch.Generator] = None) -> None:
    """Per-**tensor** uniform crossover into `child`. SPEC: 07 §7.5.

    Per tensor rather than per element because the decoder's tensors are
    functionally coupled: `W_row` and `W_col` feed one shared `tanh`, and
    blending them elementwise between two genomes that organised their hidden
    space differently gives a decoder that is neither parent's in the way that
    matters. This is the standard linkage-respecting choice.
    """
    for cm, am, bm in zip(child, a, b):
        for (name, cp), ap, bp in zip(cm.named_parameters(),
                                      am.parameters(), bm.parameters()):
            take_a = bool(torch.rand((), generator=generator) < 0.5)
            cp.copy_(ap if take_a else bp)
