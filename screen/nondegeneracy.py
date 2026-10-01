"""No genome axis may be silently inert. Asserted, not assumed.

An earlier draft of this work proposed enumerating 256 body plans over
`op x window` at `L = 4`. `M_PICO` has `T = 128, window = 128`, and
`gnomon/pheno/ops.py:92` masks only when `window < T`. `128 < 128` is false, so
"local" and "global" produced numerically identical tensors: **240 of the 256
plans were duplicates**, and the "exhaustive landscape" was 16 models repeated
sixteen times. Nothing in the design would have reported that. The enumeration
would have returned 16 groups of 16 exactly-equal numbers and been read as a
flat landscape.

The point is not the bug. The point is that no gate existed that could see it,
and one that can is three lines: build both plans with the *same* weights and
the *same* input, and require the logits to differ. This module is that gate,
and every screen calls it before it spends a GPU-hour.
"""

from __future__ import annotations

import torch

from .bauplan import Bauplan, BodyModel

__all__ = ["logit_distance", "assert_axis_is_live", "AxisNotLive"]


class AxisNotLive(AssertionError):
    """A genome axis that was varied but changed nothing."""


@torch.no_grad()
def logit_distance(a: Bauplan, b: Bauplan, *, T: int = 128, batch: int = 4,
                   seed: int = 0, device: str = "cpu") -> float:
    """Relative max-abs logit difference between two plans, same weights, same input.

    Both models are constructed from the same `seed`, so parameters that exist in
    both are drawn from the same stream. That is what makes the comparison a
    statement about the *plan* rather than about two random initialisations --
    and where the plans differ in shape it is trivially non-degenerate anyway.

    Returned relative to the logit scale, so the threshold does not have to move
    with `d_model`, the vocabulary or the dtype.
    """
    if a.vocab != b.vocab:
        return float("inf")
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    tokens = torch.randint(0, a.vocab, (batch, T), generator=g).to(device)
    ma = BodyModel(a, seed=seed).to(device).eval()
    mb = BodyModel(b, seed=seed).to(device).eval()
    la, lb = ma(tokens), mb(tokens)
    scale = max(float(la.abs().max()), float(lb.abs().max()), 1e-12)
    return float((la - lb).abs().max()) / scale


def assert_axis_is_live(a: Bauplan, b: Bauplan, *, axis: str, T: int = 128,
                        seed: int = 0, tol: float = 1e-3,
                        device: str = "cpu") -> float:
    """Raise unless varying `axis` from `a` to `b` actually changes the model.

    `tol` is a *relative* difference. Floating-point reassociation alone moves
    logits by O(1e-7) in fp32, so 1e-3 is three orders of magnitude clear of
    numerical noise while still catching an axis that only perturbs the model
    slightly rather than not at all.
    """
    if a.summary() == b.summary():
        raise AxisNotLive(
            f"axis {axis!r}: the two plans are identical -- {a.summary()}")
    d = logit_distance(a, b, T=T, seed=seed, device=device)
    if not (d > tol):
        raise AxisNotLive(
            f"axis {axis!r} is INERT at T={T}: relative logit distance {d:.2e} "
            f"<= tol {tol:.0e}.\n"
            f"  a = {a.summary()}\n  b = {b.summary()}\n"
            f"The plans differ on paper and not in the model. This is the defect "
            f"that made 240 of 256 enumerated plans duplicates: at T={T} an "
            f"attention window >= T is global, so a 'local' option of {T} is not "
            f"local. Pick a window strictly below T, or raise T."
        )
    return d
