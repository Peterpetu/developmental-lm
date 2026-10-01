"""One lifetime: train a phenotype on a stage's budget and measure it.

SPEC: 04 §4.6
SPEC: 04 §4.8
SPEC: 06 §6.6
SPEC: 08 §8.2

The loop of `docs/anlage/06 §3.6` without its growth and consolidation hooks,
which `grow.py` and `generation.py` supply through `hooks`. The training step
holds `W` as free parameters and touches no genome tensor (A3;
`G-NOFREE-INVERSE`).

Timing is the median of per-step CUDA events with an IQR, never a wall-clock
difference: this GPU is shared with a desktop (SCREEN's rule).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from typing import Callable, Optional

import torch

from .data import EvalSet, N_STRATA
from .model import Phenotype
from .recipe import TrainConfig, batch_at, build_optimizers, clip_grads, set_lr

__all__ = ["LifetimeResult", "evaluate", "per_token_loss", "train", "compile_model"]


@dataclasses.dataclass
class LifetimeResult:
    final_ce: float
    final_strata: list[float]
    curve: list[tuple[int, float]]             # (tokens, held-out CE)
    train_curve: list[tuple[int, float]]
    tokens: int
    steps: int
    ms_per_step_median: float
    ms_per_step_iqr: float
    tokens_per_s: float
    peak_mem_gb: float
    seconds: float
    backend: str
    events: list[dict]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@torch.no_grad()
def per_token_loss(model: Phenotype, window: torch.Tensor, *, chunk: int = 4096
                   ) -> torch.Tensor:
    """`(B, T)` next-token CE in nats, float32."""
    from . import ops
    from .config import ROLE_LOGIT_SCALE
    xn = model.forward(window[:, :-1], return_hidden=True)
    B, T, d = xn.shape
    x = xn.reshape(-1, d)
    t = window[:, 1:].reshape(-1)
    out = torch.empty(B * T, dtype=torch.float32, device=x.device)
    E, ls = model.head(), model.get(ROLE_LOGIT_SCALE, -1)
    for a in range(0, B * T, chunk):
        lg = ops.softcap_logits(torch.nn.functional.linear(x[a:a + chunk], E), ls).float()
        lse = torch.logsumexp(lg, -1)
        out[a:a + chunk] = lse - lg.gather(-1, t[a:a + chunk, None]).squeeze(-1)
    return out.view(B, T)


@torch.no_grad()
def evaluate(model: Phenotype, ev: EvalSet, *, batch: int = 16
             ) -> tuple[float, list[float]]:
    """Token-weighted mean CE and per-stratum CE on a fixed set. SPEC: 08 §8.2."""
    was = model.training
    model.eval()
    total = torch.zeros(N_STRATA, dtype=torch.float64, device=ev.windows.device)
    count = torch.zeros(N_STRATA, dtype=torch.float64, device=ev.windows.device)
    for i in range(0, ev.n_windows, batch):
        w = ev.windows[i:i + batch]
        s = ev.strata[i:i + batch].reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
            l = per_token_loss(model, w).reshape(-1).double()
        total.index_add_(0, s, l)
        count.index_add_(0, s, torch.ones_like(l))
    if was:
        model.train()
    ce = float(total.sum() / count.sum())
    per = (total / count.clamp_min(1)).tolist()
    return ce, [p if c > 0 else float("nan") for p, c in zip(per, count.tolist())]


def compile_model(model: Phenotype, mode: Optional[str]) -> tuple[Callable, str]:
    """`torch.compile` with a fallback to eager; the backend is reported."""
    if mode is None or not torch.cuda.is_available():
        return model, "eager"
    try:
        fn = torch.compile(model, mode=mode, dynamic=False)
        return fn, f"compile:{mode}"
    except Exception as exc:                                      # noqa: BLE001
        return model, f"eager (compile unavailable: {type(exc).__name__})"


def train(
    model: Phenotype, pool, ev: EvalSet, tc: TrainConfig, *, order_seed: int,
    eval_every_tokens: int = 0, compile_mode: Optional[str] = "default",
    hooks: Optional[dict] = None, log: Optional[Callable[[str], None]] = None,
    device: str = "cuda", train_log_every: int = 50,
) -> LifetimeResult:
    """Train for `tc.tokens` tokens drawn from `pool` in the order `order_seed`.

    `hooks`: `{"before_step": f(step, tokens_done) -> None}` for growth and
    consolidation events (they may replace optimiser state through the returned
    dict `{"optimizers": (muon, adam)}`), `{"after_eval": f(tokens, ce, strata)}`.
    """
    hooks = hooks or {}
    model.to(device).train()
    fwd, backend = compile_model(model, compile_mode)
    muon, adam = build_optimizers(model, tc)
    step_ms: list[float] = []
    curve: list[tuple[int, float]] = []
    train_curve: list[tuple[int, float]] = []
    events: list[dict] = []
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ev_a = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    ev_b = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    tokens_done, step, next_eval = 0, 0, eval_every_tokens
    if eval_every_tokens:
        ce, per = evaluate(model, ev)
        curve.append((0, ce))
    # The pool yields batches of a fixed size; the warm-up takes a prefix of
    # each batch, so every arm still consumes the same windows in the same
    # order (CRN) and only the number consumed per step differs.
    gen = pool.batches(batch=tc.batch_final, steps=1 << 30, seed=order_seed)
    ema = None
    while tokens_done < tc.tokens:
        p = tokens_done / tc.tokens
        B = batch_at(p, tc)
        if "before_step" in hooks:
            res = hooks["before_step"](step, tokens_done)
            if res and "optimizers" in res:
                muon, adam = res["optimizers"]
                fwd, backend = compile_model(model, compile_mode)
        set_lr(p, tc, muon, adam)
        window = next(gen)[:B]
        if ev_a is not None:
            ev_a.record()
        micro = B if tc.micro_batch <= 0 else min(B, tc.micro_batch)
        ce_acc = 0.0
        for w in window.split(micro):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                ce_m, zl_m = fwd.loss(w) if hasattr(fwd, "loss") else model.loss(w)
            # the mean over the full batch is the size-weighted mean over micro-batches
            ((ce_m + zl_m) * (w.shape[0] / B)).backward()
            ce_acc += float(ce_m) * (w.shape[0] / B)
        ce = torch.tensor(ce_acc)
        gnorm = clip_grads(model, tc)
        if not math.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        muon.step()
        adam.step()
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        if ev_b is not None:
            ev_b.record()
            ev_b.synchronize()
            step_ms.append(ev_a.elapsed_time(ev_b))
        tokens_done += B * tc.T
        step += 1
        cef = float(ce)
        ema = cef if ema is None else 0.98 * ema + 0.02 * cef
        if train_log_every and step % train_log_every == 0:
            train_curve.append((tokens_done, ema))
            if log:
                log(f"    step {step:6d}  tokens {tokens_done:>12,}  train {ema:.4f}  "
                    f"lr_M {muon.param_groups[0]['lr']:.2e}  B {B}  "
                    f"{step_ms[-1] if step_ms else float('nan'):.0f} ms")
        if eval_every_tokens and tokens_done >= next_eval:
            ce_h, per = evaluate(model, ev)
            curve.append((tokens_done, ce_h))
            next_eval += eval_every_tokens
            if "after_eval" in hooks:
                hooks["after_eval"](tokens_done, ce_h, per)
            if log:
                log(f"  eval tokens {tokens_done:>12,}  held-out {ce_h:.4f}")
    final, per = evaluate(model, ev)
    if not curve or curve[-1][0] != tokens_done:
        curve.append((tokens_done, final))
    med = statistics.median(step_ms) if step_ms else float("nan")
    if len(step_ms) >= 4:
        s = sorted(step_ms)
        iqr = s[3 * len(s) // 4] - s[len(s) // 4]
    else:
        iqr = float("nan")
    seconds = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return LifetimeResult(
        final_ce=final, final_strata=per, curve=curve, train_curve=train_curve,
        tokens=tokens_done, steps=step, ms_per_step_median=med, ms_per_step_iqr=iqr,
        tokens_per_s=tokens_done / max(seconds, 1e-9), peak_mem_gb=peak, seconds=seconds,
        backend=backend, events=events)
