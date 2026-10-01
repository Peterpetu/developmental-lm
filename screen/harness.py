"""Train one model for N steps and return a curve. Nothing cleverer.

An earlier draft of this work proposed a `torch.func.stack_module_state` +
`vmap` population trainer with a gate demanding an 8x speedup. Measured on this
GPU it raised on three of four genome axes (heterogeneous plans have different
tensor sets and different shapes, and `ops.py:92`'s window branch is not
traceable), and where it did run it reached 0.47 Mtok/s against 0.71 Mtok/s for
`torch.compile` on a single model -- 0.6x, not 8x. It is deleted on measurement
rather than deferred.

**Timing.** This machine's GPU is shared with the desktop (Chrome, Discord,
Spotify and ~30 other processes hold CUDA contexts). Identical runs have
differed by 3.4x in wall clock. So every duration here is the *median* of
per-step CUDA-event timings, never a wall-clock difference, and `ms_per_step`
carries its own interquartile range so a contaminated measurement is visible
instead of silent.
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from typing import Optional

import torch

from .bauplan import Bauplan, BodyModel
from .crn import WindowPool

__all__ = ["RunResult", "train_and_eval", "evaluate"]


@dataclasses.dataclass
class RunResult:
    """Everything a screen needs from one lifetime, and nothing it does not."""

    plan: str
    seed: int
    steps: int
    batch: int
    final: float                       # held-out CE in nats
    curve: list[tuple[int, float]]
    ms_per_step_median: float
    ms_per_step_iqr: float
    seconds: float
    n_unique_params: int
    flops_per_token: int
    train_flops: int
    backend: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@torch.no_grad()
def evaluate(model, eval_win: torch.Tensor, *, batch: int = 32) -> float:
    """Mean next-token CE over a fixed evaluation set, in nats.

    Token-weighted rather than batch-weighted: every window is the same length
    here, so the two agree, but a mean-of-means would silently stop agreeing the
    moment ragged windows were introduced.
    """
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i in range(0, eval_win.shape[0], batch):
        w = eval_win[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=w.is_cuda):
            loss = model.loss(w)
        k = w.shape[0] * (w.shape[1] - 1)
        total += float(loss) * k
        n += k
    if was_training:
        model.train()
    return total / n


def _lr_at(step: int, steps: int, lr: float, *, warmup: int, floor: float) -> float:
    """Linear warmup then cosine to `floor * lr`.

    Not `train.py:wsd_lr`: warmup-stable-decay is right for a run whose length is
    the thing being studied, and here the horizon is fixed per screen while the
    *plans* vary, so the schedule should be the same well-understood shape for
    every arm. Cosine is the shape every baseline in the literature this screen
    is compared against was tuned under.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    import math
    t = (step - warmup) / max(steps - warmup, 1)
    return lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))


def train_and_eval(
    plan: Bauplan,
    *,
    pool: WindowPool,
    eval_win: torch.Tensor,
    steps: int,
    batch: int = 32,
    seed: int = 0,
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    warmup: Optional[int] = None,
    lr_floor: float = 0.1,
    eval_every: int = 0,
    compile_mode: Optional[str] = "default",
    device: str = "cuda",
) -> RunResult:
    """One lifetime. `seed` selects init *and* data order, and nothing else.

    The `seed` contract is what makes a screen paired: arm `i` seed `s` and arm
    `j` seed `s` consume identical windows in identical order (`crn.py`), so the
    difference between them is the genome and the initialisation, not the data.
    """
    if warmup is None:
        warmup = max(1, steps // 20)
    torch.manual_seed(seed)
    model = BodyModel(plan, seed=seed).to(device)
    model.train()

    backend = "eager"
    fwd = model
    if compile_mode is not None and device == "cuda":
        try:
            fwd = torch.compile(model, mode=compile_mode)
            backend = f"compile:{compile_mode}"
        except Exception:                                    # noqa: BLE001
            fwd, backend = model, "eager (compile unavailable)"

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=weight_decay, fused=(device == "cuda"))

    step_ms: list[float] = []
    curve: list[tuple[int, float]] = []
    t0 = time.time()
    if eval_every:
        curve.append((0, evaluate(fwd, eval_win)))

    ev_a = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    ev_b = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    for n, window in enumerate(pool.batches(batch=batch, steps=steps, seed=seed)):
        for grp in opt.param_groups:
            grp["lr"] = _lr_at(n, steps, lr, warmup=warmup, floor=lr_floor)
        if ev_a is not None:
            ev_a.record()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = fwd.loss(window) if hasattr(fwd, "loss") else model.loss(window)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ev_b is not None:
            ev_b.record()
            ev_b.synchronize()
            step_ms.append(ev_a.elapsed_time(ev_b))
        if eval_every and (n + 1) % eval_every == 0:
            curve.append((n + 1, evaluate(fwd, eval_win)))

    final = evaluate(fwd, eval_win)
    if not curve or curve[-1][0] != steps:
        curve.append((steps, final))

    # Median and IQR, never the mean: a desktop application stealing the GPU for
    # one step is an outlier, not a component of the model's cost.
    med = statistics.median(step_ms) if step_ms else float("nan")
    if len(step_ms) >= 4:
        s = sorted(step_ms)
        iqr = s[3 * len(s) // 4] - s[len(s) // 4]
    else:
        iqr = float("nan")

    return RunResult(
        plan=plan.summary(), seed=seed, steps=steps, batch=batch,
        final=final, curve=curve, ms_per_step_median=med, ms_per_step_iqr=iqr,
        seconds=time.time() - t0,
        n_unique_params=plan.n_unique_params(),
        flops_per_token=plan.flops_per_token(T=pool.T),
        train_flops=3 * plan.flops_per_token(T=pool.T) * steps * batch * pool.T,
        backend=backend,
    )
