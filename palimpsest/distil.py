"""Consolidation: per-position distillation and dreams.

SPEC: docs/palimpsest/01-model.md §1.5 (Definitions 5-7), §1.10 (numerics)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterator, Optional

import torch
import torch.nn.functional as F

from anlage import ops
from anlage.config import ROLE_LOGIT_SCALE
from anlage.model import Phenotype
from anlage.recipe import TrainConfig, build_optimizers, clip_grads, set_lr

__all__ = ["EOS_ID", "distil_loss", "dream", "distil"]

EOS_ID = 2
"""The document terminator of every shard here (`gnomon/data/corpus.py::tokenize_corpus`)."""


def _hidden(model: Phenotype, window: torch.Tensor) -> torch.Tensor:
    return model.forward(window[:, :-1], return_hidden=True)


def _kl_chunks(xs: torch.Tensor, student: Phenotype, xt: torch.Tensor, teacher: Phenotype,
               chunk: int) -> torch.Tensor:
    """`sum_t sum_v p_T (log p_T - log p_S)` over positions in chunks. SPEC: (5.1)."""
    d_s, d_t = xs.shape[-1], xt.shape[-1]
    xs2, xt2 = xs.reshape(-1, d_s), xt.reshape(-1, d_t)
    E_s, ls_s = student.head(), student.get(ROLE_LOGIT_SCALE, -1)
    E_t, ls_t = teacher.head(), teacher.get(ROLE_LOGIT_SCALE, -1)
    n = xs2.shape[0]
    acc = xs2.new_zeros((), dtype=torch.float32)
    for a in range(0, n, chunk):
        with torch.no_grad():
            lt = ops.softcap_logits(F.linear(xt2[a:a + chunk], E_t), ls_t).float()
            log_pt = torch.log_softmax(lt, -1)
            pt = log_pt.exp()
        ls_ = ops.softcap_logits(F.linear(xs2[a:a + chunk], E_s), ls_s).float()
        log_ps = torch.log_softmax(ls_, -1)
        acc = acc + (pt * (log_pt - log_ps)).sum()
    return acc / n


def distil_loss(student: Phenotype, teacher: Phenotype, window: torch.Tensor, *,
                chunk: int = 2048) -> torch.Tensor:
    """`D(S || Theta; w)` of (5.1), a scalar with a gradient into the student only."""
    if window.dim() != 2 or window.shape[1] < 2:
        raise ValueError("window (B, T+1)")
    with torch.no_grad():
        xt = _hidden(teacher, window)
    xs = _hidden(student, window)
    return _kl_chunks(xs, student, xt, teacher, chunk)


@torch.no_grad()
def dream(core: Phenotype, n: int, T_d: int, *, temperature: float = 1.0, seed: int,
          device: str = "cuda", batch: int = 512, bos: int = EOS_ID, cached: bool = True,
          start: str = "mixed") -> torch.Tensor:
    """`Dream(C; n, T_d, tau, seed)`: `(n, T_d + 1)` sequences by ancestral sampling from the
    start token. SPEC: Definition 6. One uniform per (sequence, position) is drawn up front
    from the seed, and the draw is by inverse CDF in float64, so the sample is a function of
    the seed alone and not of the batch grouping (T4). `cached=True` steps the model with
    `palimpsest/sample.py::Sampler` (the exact recurrence with carried state, a KV cache);
    `cached=False` re-runs the full forward on the growing prefix (`O(T_d^2)`, the reference
    of T4b)."""
    if n < 1 or T_d < 1 or temperature <= 0:
        raise ValueError("n >= 1, T_d >= 1, temperature > 0")
    if start not in ("eos", "uniform", "mixed"):
        raise ValueError("start in eos|uniform|mixed")
    from .sample import Sampler
    was = core.training
    core.eval()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    U = torch.rand(n, T_d, generator=g, dtype=torch.float64)
    # the start token: EOS (the state every *short* document begins in), a uniformly random
    # token, or half and half by sequence index (P-P.5: a domain whose documents are long is
    # never entered from EOS by a model that only saw windows of it)
    first = torch.full((n,), bos, dtype=torch.long)
    rnd = torch.randint(0, core.vocab, (n,), generator=g)
    if start == "uniform":
        first = rnd
    elif start == "mixed":
        first[1::2] = rnd[1::2]
    out = []
    for s0 in range(0, n, batch):
        b = min(batch, n - s0)
        seq = first[s0:s0 + b, None].to(device)
        sampler = Sampler(core, b, device) if cached else None
        for t in range(T_d):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                if cached:
                    logits = sampler.step(seq[:, -1]).float()
                else:
                    logits = core(seq)[:, -1, :].float()
            probs = torch.softmax(logits.double() / temperature, -1).cpu()
            cdf = probs.cumsum(-1)
            u = U[s0:s0 + b, t, None]
            draw = (cdf < u).sum(-1, keepdim=True).clamp_(max=probs.shape[-1] - 1).to(device)
            seq = torch.cat([seq, draw], 1)
        out.append(seq)
    if was:
        core.train()
    return torch.cat(out, 0)


def distil(core: Phenotype, teacher: Phenotype, data_batches: Iterator[torch.Tensor], *,
           tc: TrainConfig, dreams: Optional[torch.Tensor] = None,
           prev_core: Optional[Phenotype] = None, lam_dream: float = 1.0,
           dream_batch: int = 32, log: Optional[Callable[[str], None]] = None,
           device: str = "cuda", seed: int = 0) -> dict:
    """The argmin of (7.2): `N_d / (B T)` steps on data batches with the phenotype as teacher,
    each followed by one step on a dream batch with the previous core as teacher when
    `lam_dream > 0` and dreams exist. Returns the curves and the token counts."""
    if lam_dream < 0:
        raise ValueError("lam_dream >= 0")
    core.to(device).train()
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if prev_core is not None:
        prev_core.to(device).eval()
        for p in prev_core.parameters():
            p.requires_grad_(False)
    muon, adam = build_optimizers(core, tc)
    steps = max(1, tc.tokens // (tc.batch_final * tc.T))
    use_dreams = dreams is not None and prev_core is not None and lam_dream > 0 and dreams.shape[0] > 0
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    curve, dcurve = [], []
    t0 = time.time()
    micro = tc.batch_final if tc.micro_batch <= 0 else min(tc.batch_final, tc.micro_batch)
    for step in range(steps):
        p = step / steps
        set_lr(p, tc, muon, adam)
        w = next(data_batches)[: tc.batch_final].to(device)
        loss_acc = 0.0
        for wm in w.split(micro):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss = distil_loss(core, teacher, wm)
            (loss * (wm.shape[0] / w.shape[0])).backward()
            loss_acc += float(loss) * (wm.shape[0] / w.shape[0])
        gn = clip_grads(core, tc)
        if not math.isfinite(gn):
            raise FloatingPointError(f"non-finite gradient in distillation at step {step}")
        muon.step(); adam.step()
        muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
        curve.append(loss_acc)
        if use_dreams:
            idx = torch.randint(0, dreams.shape[0], (dream_batch,), generator=g)
            wd = dreams[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                ld = distil_loss(core, prev_core, wd)
            (lam_dream * ld).backward()
            gn = clip_grads(core, tc)
            if not math.isfinite(gn):
                raise FloatingPointError(f"non-finite gradient in dream distillation at step {step}")
            muon.step(); adam.step()
            muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
            dcurve.append(float(ld))
        if log and (step + 1) % 25 == 0:
            log(f"    distil step {step + 1}/{steps}  D_data {loss_acc:.4f}"
                + (f"  D_dream {dcurve[-1]:.4f}" if dcurve else ""))
    return {"steps": steps, "data_tokens": steps * tc.batch_final * tc.T,
            "dream_tokens": (len(dcurve) * dream_batch * (dreams.shape[1] - 1)) if use_dreams else 0,
            "curve": curve, "dream_curve": dcurve, "seconds": time.time() - t0}
