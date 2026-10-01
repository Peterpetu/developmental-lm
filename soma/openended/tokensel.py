"""Token-level selection: the excess-loss criterion and `L_task`.

SPEC: 06 §6.2

```
Delta_i = L_theta(x_i) - L_ref(x_i)                                      (6.3)
I_i     = 1 if rank(Delta_i) is in the top k_tok fraction *within its source*
          or with probability p_explore, else 0                          (6.4)
L_task  = - ( 1 / max(1, sum_i I_i) ) * sum_i I_i * log P(x_i | x_{<i}; theta)  (6.5)
```

**[NORMATIVE — within-source ranking]** Ranking is performed **separately within
the natural and synthetic sources**, and each source contributes its own `k_tok`
fraction. Spec v1 ranked jointly; synthetic tokens carry persistently high
`Delta` and would have crowded out natural tokens, silently re-weighting the
objective and confounding ablation **A5** with **A6**.

**[NORMATIVE — synthetic tokens are ranked by raw loss (audit A-25)]** `L_ref`
exists only over the natural-text corpus, so `Delta_i` is **undefined** on the
synthetic source. Within-source ranking fixes the *comparability* of the
ranking, not the *existence* of the quantity being ranked. Synthetic tokens are
therefore ranked by **raw** `L_theta` within their own source — a weaker
criterion, stated as such. It is acceptable here because the archive already
supplies the learnability signal that `Delta_i` supplies for natural text: a
mastered generator is evicted by `Learn = 4 s (1-s)`, so raw loss over the
surviving generators is bounded away from the already-learned regime by
construction.

**[NORMATIVE]** The normaliser is `sum_i I_i`, the realised count, not
`N_tok * k_tok`; `p_explore` inclusions make the two differ.

**[NORMATIVE]** Ranking is over the **micro-batch**. Ranking over a full
gradient-accumulated batch would require all micro-batches' `Delta` before any
backward pass, which is not implementable in one pass.

**[EMPIRICAL]** Token-loss trajectories partition as: low->low **51%** (already
learned), high->low **26%** (the learnable band), high->high **11%**
(unlearnable), low->high **12%**. The ratio curve is flat from `k_tok = 0.9` to
`0.6` and falls **below** the no-selection baseline at `0.5`; `k_tok = 0.65`,
and values below `0.55` are rejected by config validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from soma.utils.shapes import check_shape

__all__ = ["TokenSelection", "select_tokens", "masked_cross_entropy", "ExploreState"]


@dataclass
class TokenSelection:
    """The realised inclusion mask and its diagnostics."""

    include: torch.Tensor
    """`(B, T)` bool."""
    n_selected: int
    synthetic_share: float
    """§06.2: the realised synthetic share of **selected** tokens is a required
    log field, because it is what de-confounds A5 from A6."""
    n_explore: int
    natural_threshold: float
    synthetic_threshold: float


@dataclass
class ExploreState:
    """`p_explore` and its two-sided degeneracy rule (§06.2).

    **[NORMATIVE — the maturity gap, OURS]** Token selection has no analogue of
    `M_mat`: a token that becomes temporarily hard can be deselected
    indefinitely. With probability `p_explore = 0.05` a token is included
    regardless of rank.

    **Degeneracy rule.** Every `10 * T_log` steps, restricted to vocabulary items
    that **occurred at least 32 times** in that window, let `f_starved` be the
    fraction whose realised inclusion rate is below `0.5 * k_tok`. Then

    ```
    if f_starved > 0.10:   p_explore <- min( 1.5 * p_explore , 0.20 )
    else:                  p_explore <- max( 0.05 , p_explore / 1.5 )
    ```

    Spec v1 left the rule unspecified inside a `[NORMATIVE]` block. v2 supplied
    one that was inert in both directions: over `10 * T_log` steps the fraction
    of the *whole* vocabulary with zero inclusions is dominated by items that
    never **occur** at all, so at `V = 8192` it sits near its ceiling permanently
    instead of crossing `0.5` on a selection pathology; and the rule could only
    ever raise `p_explore`, with no cadence and no decay, so once raised it never
    returned (audit A-40).
    """

    V: int
    p_explore: float = 0.05
    min_occurrences: int = 32
    starved_threshold: float = 0.10
    p_min: float = 0.05
    p_max: float = 0.20
    occurrences: Optional[torch.Tensor] = None
    inclusions: Optional[torch.Tensor] = None
    n_adjustments: int = 0
    last_f_starved: float = 0.0

    def __post_init__(self) -> None:
        if self.occurrences is None:
            self.occurrences = torch.zeros(self.V, dtype=torch.long)
        if self.inclusions is None:
            self.inclusions = torch.zeros(self.V, dtype=torch.long)

    @torch.no_grad()
    def observe(self, tokens: torch.Tensor, include: torch.Tensor) -> None:
        flat = tokens.reshape(-1).cpu()
        inc = include.reshape(-1).cpu()
        assert self.occurrences is not None and self.inclusions is not None
        self.occurrences.index_add_(0, flat, torch.ones_like(flat))
        self.inclusions.index_add_(0, flat, inc.long())

    @torch.no_grad()
    def adjust(self, k_tok: float) -> float:
        """Apply the degeneracy rule and reset the window."""
        assert self.occurrences is not None and self.inclusions is not None
        seen = self.occurrences >= self.min_occurrences
        if bool(seen.any()):
            rate = self.inclusions[seen].float() / self.occurrences[seen].float()
            f_starved = float((rate < 0.5 * k_tok).float().mean())
        else:
            f_starved = 0.0
        self.last_f_starved = f_starved
        if f_starved > self.starved_threshold:
            self.p_explore = min(self.p_max, 1.5 * self.p_explore)
        else:
            self.p_explore = max(self.p_min, self.p_explore / 1.5)
        self.n_adjustments += 1
        self.occurrences.zero_()
        self.inclusions.zero_()
        return self.p_explore


def _top_fraction_mask(
    scores: torch.Tensor, valid: torch.Tensor, k_tok: float
) -> Tuple[torch.Tensor, float]:
    """Top-`k_tok` fraction of `scores` among `valid`, as a boolean mask."""
    n = int(valid.sum())
    if n == 0:
        return torch.zeros_like(valid), float("nan")
    k = max(1, int(round(k_tok * n)))
    vals = scores[valid]
    thresh = float(torch.kthvalue(vals.float(), max(1, n - k + 1)).values)
    return valid & (scores >= thresh), thresh


def select_tokens(
    loss_per_token: torch.Tensor,
    ref_per_token: Optional[torch.Tensor],
    is_synthetic: torch.Tensor,
    scored: torch.Tensor,
    k_tok: float,
    explore: ExploreState,
    generator: Optional[torch.Generator] = None,
) -> TokenSelection:
    """(6.3)-(6.4). All tensors are `(B, T)`.

    `scored` marks positions that carry a loss at all — a synthetic instance's
    prompt span is excluded by §06.3.2's answer-span masking, and padding is
    never scored.

    Both terms of (6.3) are accumulated in **`float32`** (§05.9).
    """
    b = {}
    check_shape(loss_per_token, ["B", "T"], name="L_theta", bindings=b)
    check_shape(is_synthetic, ["B", "T"], name="is_synthetic", bindings=b)
    check_shape(scored, ["B", "T"], name="scored", bindings=b)

    L = loss_per_token.float()
    natural = scored & (~is_synthetic)
    synthetic = scored & is_synthetic

    # Natural: excess loss against the frozen, better reference.
    if ref_per_token is not None and bool(natural.any()):
        check_shape(ref_per_token, ["B", "T"], name="L_ref", bindings=b)
        delta = L - ref_per_token.float()
    else:
        # No cache (M0's own run, or a smoke test): fall back to raw loss, which
        # is the same criterion the synthetic source uses.
        delta = L

    nat_mask, nat_thresh = _top_fraction_mask(delta, natural, k_tok)
    # Synthetic: raw loss within its own source (audit A-25).
    syn_mask, syn_thresh = _top_fraction_mask(L, synthetic, k_tok)

    include = nat_mask | syn_mask

    # p_explore: include regardless of rank, so a temporarily hard token cannot
    # be deselected indefinitely.
    n_explore = 0
    if explore.p_explore > 0:
        u = torch.rand(
            L.shape, generator=generator, device=L.device, dtype=torch.float32
        )
        extra = scored & (~include) & (u < explore.p_explore)
        n_explore = int(extra.sum())
        include = include | extra

    n_sel = int(include.sum())
    syn_sel = int((include & is_synthetic).sum())
    return TokenSelection(
        include,
        n_sel,
        (syn_sel / n_sel) if n_sel else 0.0,
        n_explore,
        nat_thresh,
        syn_thresh,
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    include: torch.Tensor,
    *,
    lambda_z: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(6.5) plus the z-loss of §05.6, over **the same token set**.

    SPEC: 06 §6.2
    SPEC: 05 §5.6

    Returns `(L_task, L_z, loss_per_token)`.

    §05.6: the z-loss is averaged over **the same token set as `L_task`** (the
    selected tokens), so the two terms share a denominator. Averaging them over
    different sets would make their relative weight depend on `k_tok`, and A5
    (`k_tok = 1.0`) would then silently change the regulariser as well as the
    selection.

    The normaliser is `max(1, sum_i I_i)`, the **realised** count, not
    `N_tok * k_tok`.
    """
    b = {}
    check_shape(logits, ["B", "T", "V"], name="logits", bindings=b)
    check_shape(targets, ["B", "T"], name="targets", bindings=b)
    check_shape(include, ["B", "T"], name="include", bindings=b)

    lf = logits.float()
    log_Z = torch.logsumexp(lf, dim=-1)                     # (B, T)
    gathered = lf.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss_per_token = log_Z - gathered                       # (B, T) = -log P

    mask = include.to(lf.dtype)
    denom = mask.sum().clamp_min(1.0)
    L_task = (loss_per_token * mask).sum() / denom
    L_z = lambda_z * ((log_Z.square() * mask).sum() / denom)
    return L_task, L_z, loss_per_token.detach()
