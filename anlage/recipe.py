"""Optimisers, schedules and the step. SPEC: 04 §4.6

```
    lr(p)  = lr_max · p / p_warm                      p <  p_warm = 0.02
           = lr_max                                   p_warm ≤ p < p_dec = 0.8
           = lr_max · (1 - p) / (1 - p_dec)           p ≥ p_dec          (decay to zero)
    B(p)   = B_final · clip( 0.25 + 0.75 p / 0.1, 0.25, 1 )   rounded to a multiple of 8
```

with `p` the fraction of the stage's token budget consumed. Both are functions
of the token fraction rather than the step index because the batch warm-up
makes the number of steps a derived quantity.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

from gnomon.optim.muon import Muon
from .model import Phenotype

__all__ = ["TrainConfig", "lr_at", "batch_at", "build_optimizers", "set_lr", "clip_grads"]


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """One stage's training hyperparameters. SPEC: 04 §4.6."""

    tokens: int                      # the stage's token budget
    batch_final: int                 # sequences per optimiser step at full batch
    T: int
    lr_muon: float = 0.02
    lr_emb: float = 3e-3
    lr_vec: float = 1e-3
    weight_decay_muon: float = 0.1
    p_warm: float = 0.02
    p_dec: float = 0.8
    p_batch_warm: float = 0.1
    batch_min_frac: float = 0.25
    clip: float = 1.0
    #: micro-batch (sequences per forward/backward); 0 = the whole batch. Gradient
    #: accumulation over micro-batches gives the same step at the same batch (P14), so a
    #: stage that does not fit in memory at `batch_final` trains the same recipe.
    micro_batch: int = 0
    lr_provenance: str = "NOT MEASURED at this configuration"

    def __post_init__(self) -> None:
        if self.batch_final < 8 or self.batch_final % 8:
            raise ValueError("batch_final must be a positive multiple of 8")
        if not (0 < self.p_warm < self.p_dec < 1):
            raise ValueError("need 0 < p_warm < p_dec < 1")


def lr_at(p: float, lr_max: float, tc: TrainConfig) -> float:
    """The schedule at token fraction `p in [0, 1]`."""
    if p < tc.p_warm:
        return lr_max * p / tc.p_warm
    if p < tc.p_dec:
        return lr_max
    return lr_max * max(0.0, 1.0 - p) / (1.0 - tc.p_dec)


def batch_at(p: float, tc: TrainConfig) -> int:
    """The batch warm-up at token fraction `p`, in sequences (a multiple of 8)."""
    frac = min(1.0, max(tc.batch_min_frac, tc.batch_min_frac
                        + (1.0 - tc.batch_min_frac) * p / tc.p_batch_warm))
    b = int(round(tc.batch_final * frac / 8.0)) * 8
    return max(8, min(tc.batch_final, b))


def build_optimizers(model: Phenotype, tc: TrainConfig
                     ) -> tuple[Muon, torch.optim.AdamW]:
    """Group M under Muon, group A under AdamW with two learning rates."""
    muon_p, adam_p = model.optimizer_groups()
    from .config import ROLE_EMB, ROLE_HEAD_OUT, ROLE_VE
    emb_ids = set()
    for role in (ROLE_EMB, ROLE_HEAD_OUT, ROLE_VE):
        for l in range(-1, model.cfg.L):
            if model.has(role, l):
                emb_ids.add(id(model.get(role, l)))
    emb = [p for p in adam_p if id(p) in emb_ids]
    vec = [p for p in adam_p if id(p) not in emb_ids]
    muon = Muon(muon_p, lr=tc.lr_muon, momentum=0.95, weight_decay=tc.weight_decay_muon)
    adam = torch.optim.AdamW(
        [{"params": emb, "lr": tc.lr_emb, "base_lr": tc.lr_emb},
         {"params": vec, "lr": tc.lr_vec, "base_lr": tc.lr_vec}],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        fused=torch.cuda.is_available() and muon_p[0].is_cuda if muon_p else False)
    for g in muon.param_groups:
        g["base_lr"] = tc.lr_muon
    return muon, adam


def set_lr(p: float, tc: TrainConfig, *optimizers) -> None:
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = lr_at(p, g["base_lr"], tc)


def clip_grads(model: Phenotype, tc: TrainConfig) -> float:
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip))


def micro_batch_for(n_params: int, batch: int) -> int:
    """SPEC: 04 §4.8 -- the micro-batch this card (16 GB) takes at `T = 512`: measured
    8.35 GB peak at 13M parameters and batch 32; activations scale with width × depth."""
    if n_params <= 20_000_000:
        return batch
    if n_params <= 45_000_000:
        return max(8, batch // 2)
    return max(8, batch // 4)
