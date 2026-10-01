"""Manual CUDA-graph capture of one training step.

SPEC: 05 §5.1
SPEC: 07 §7.1

Why not `torch.compile`. The step is **CPU-dispatch bound, not GPU bound** --
profiled at M-pico at 131 ms of CPU against 25 ms of CUDA per step, across
roughly 4,000 kernel launches, because the decode is a long chain of cheap
elementwise operations over `(R, C, d_h)` tensors. `torch.compile` with
`mode="reduce-overhead"` fixes exactly that, taking 79.5 ms/step to 11.5, and
was the first thing tried.

It does not survive architecture search. Every genome decodes its own graph
(§07), so each distinct architecture pays a fresh ~30 s inductor compile; worse,
inductor emits a CPU kernel for some of those graphs and this machine has no
MSVC, so the compile *fails* and the run has to degrade to eager. Measured:
eight architectures took over ten minutes just to bind.

Capturing the step as a CUDA graph directly gives the same benefit -- replay
issues the whole step as one launch -- with **no code generation and no C++
toolchain**. Capture costs about a second, so paying it per architecture is
affordable, which is what makes §08's search possible at all.

Requirements the rest of the codebase already satisfies:

* static input addresses -- the runner owns `_tok`, `_eps` and `_W_rand` buffers
  and copies into them, rather than passing fresh tensors;
* static shapes for one architecture -- guaranteed by `Arch`;
* no data-dependent control flow inside the step -- the Python loops over
  layers and chunks are unrolled at capture;
* no RNG inside the step -- there is no dropout anywhere in the model.

**[NORMATIVE -- one requirement the codebase did *not* satisfy, and the fix.]**
This module was written at v1.0 and never imported. Connecting it at v1.1 found
why in one line of error text:

    unique_by_key: failed to synchronize: cudaErrorStreamCaptureUnsupported

`F.embedding`'s backward accumulates the gradient by **sorting** its indices,
and the sort calls `unique_by_key`, which synchronises. A synchronising kernel
cannot be captured. `ops.embed` therefore switches to the advanced-indexing
form `E[tokens]` under capture, whose backward accumulates with `atomicAdd`.

The two backwards compute the same sum in a different order, so they agree to
within float32 rounding and no further. Measured, `max|difference| / max|grad|`:

| `(V, d)` | `B*T` | tokens per row | relative difference |
|---|---|---|---|
| 512, 32 | 512 | 1.0 | `3.8e-08` |
| 4096, 128 | 4096 | 1.0 | **`0`** |
| 8192, 512 | 12288 | 1.5 | **`0`** |
| 512, 32 | 4096 | 8.0 | `6.8e-08` |
| 4096, 128 | 16384 | 4.0 | `6.7e-08` |
| 64, 32 | 256 | 4.0 | `5.9e-08` |

Every entry is under one float32 ulp (`1.19e-07`), and the exact rows are the
ones where barely any two tokens share a row -- there is then nothing to
accumulate and no order to differ in. So this is a change of *kernel*, not of
arithmetic, and the bound on the change is machine epsilon.

**[NORMATIVE -- what the captured step does and does not reproduce, measured.]**
M-pico, batch 128, `lr = 0.09`, one genome, one seed, against the eager path:

| | |
|---|---|
| step 1 train loss | **identical**, `0.00e+00` |
| steps 2-12 | `0` to `2.4e-06` -- float32 epsilon, from kernel selection |
| step 20 | `6.2e-03`, and growing |
| each backend against itself | **bit-for-bit identical** |
| throughput | 179.9 -> 112.4 ms/step, **1.60x** |

The step is therefore semantically the same and numerically the same to within
one ulp; what separates the trajectories is that an optimiser running through a
`tanh` decoder amplifies an epsilon exponentially. §05.7's L7 -- "two lifetimes
with one `(Gamma, seed)` give bit-identical final `Z` and CE" -- holds **within**
a backend, which is the sense in which any CUDA library satisfies it, and does
not hold across backends. A run states which backend produced it.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from ..utils.shapes import set_finite_checks

__all__ = ["CapturedStep", "capture_step"]


class CapturedStep:
    """One captured forward+backward, replayable at near-zero CPU cost."""

    def __init__(self, graph: torch.cuda.CUDAGraph, outputs: Sequence[torch.Tensor],
                 params: Sequence[torch.nn.Parameter]):
        self.graph = graph
        self.outputs = list(outputs)
        self.params = list(params)

    def __call__(self) -> list[torch.Tensor]:
        """Replay. The gradients land in the same `.grad` tensors every time.

        `.grad` must never be set to `None` between replays: the graph writes
        into those exact allocations, so `zero_grad(set_to_none=True)` would
        silently detach the gradients the optimizer then reads as stale zeros.
        """
        self.graph.replay()
        return self.outputs


def capture_step(fn: Callable[[], tuple], params: Sequence[torch.nn.Parameter],
                 *, warmup: int = 3,
                 pool: Optional[tuple] = None) -> CapturedStep:
    """Capture `loss, *aux = fn()` followed by `loss.backward()`.

    `params` are the tensors whose `.grad` the caller will read. They are
    pre-allocated here so that capture writes into stable addresses.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-graph capture requires a CUDA device")

    # Gradients must exist and be zeroed before capture, and must never be set
    # to None afterwards.
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    # Warmup on a side stream: cuBLAS and the caching allocator both need to
    # have taken their one-off decisions before capture, or those allocations
    # end up baked into the graph.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    was = set_finite_checks(False)
    try:
        with torch.cuda.stream(side):
            for _ in range(warmup):
                for p in params:
                    p.grad.zero_()
                out = fn()
                out[0].backward()
    finally:
        set_finite_checks(was)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    for p in params:
        p.grad.zero_()
    g = torch.cuda.CUDAGraph()
    was = set_finite_checks(False)
    try:
        with torch.cuda.graph(g, pool=pool):
            out = fn()
            out[0].backward()
    finally:
        set_finite_checks(was)
    return CapturedStep(g, out, params)
