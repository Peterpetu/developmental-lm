"""Evaluation metrics and the lifetime fitness.

SPEC: 08 §8.1
SPEC: 07 §7.1
SPEC: 07 §7.3

**[NORMATIVE]** Every reported curve is **held-out `L_task` with `k_tok = 1` on
natural text only**, with structural events annotated. The training objective
(which sums `L_met` and normalises over selected tokens) is **never** plotted as
a loss curve.

That rule is what makes SOMA's curve interpretable at all. §4 of the plan states
the problem: the training objective is not a comparable quantity, capacity is
non-stationary, the data distribution is non-stationary, up to 50% of the run is
exploratory by design, and structural events cause transients. A curve that
mixes all of that measures nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.config import SomaConfig
from soma.utils.stats import effective_rank, log_slope

__all__ = [
    "EvalPoint",
    "CurveTracker",
    "Fitness",
    "compute_fitness",
    "bits_per_byte",
    "held_out_ce",
    "erank_probe",
]


def bits_per_byte(ce_nats: float, n_tokens: int, n_bytes: int) -> float:
    """`BPB = (CE / ln 2) * (tokens / bytes)` (§08.1).

    The token/byte ratio is what makes BPB comparable across tokenizers, which
    is the whole point of reporting it rather than perplexity: `PPL` is
    tokenizer-dependent and §08.1 marks it **reporting only, never a decision
    basis**.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return (ce_nats / math.log(2.0)) * (n_tokens / n_bytes)


@torch.no_grad()
def held_out_ce(
    model,
    batches: Sequence,
    *,
    genome=None,
    device: torch.device | str = "cpu",
) -> Tuple[float, int]:
    """Mean next-token CE in nats over natural text, `k_tok = 1` (§08.1).

    Returns `(ce, n_tokens)`. No token selection, no metabolic term, no z-loss:
    this is the quantity every reported curve plots, and mixing anything else
    into it would make two runs incomparable.
    """
    model.eval()
    total = 0.0
    count = 0
    for b in batches:
        b = b.to(device)
        out = model(
            b.tokens, genome=genome, doc_id=b.doc_id, valid=b.valid,
            want_metabolic=False,
        )
        lf = out.logits.float()
        logZ = torch.logsumexp(lf, dim=-1)
        gathered = lf.gather(-1, b.targets.unsqueeze(-1)).squeeze(-1)
        loss = logZ - gathered
        mask = (b.scored & (~b.is_synthetic)).float()
        total += float((loss * mask).sum())
        count += int(mask.sum())
    model.train()
    if count == 0:
        # **[NORMATIVE — no scored token is not a loss of zero; v3.1, audit A-90]**
        # `total / max(1, count)` returned **0.0** when nothing was scored, and
        # `CurveTracker.record` accepts any `ce <= L_star` as reaching the
        # criterion — so an evaluation batch with no scorable natural token
        # instantly un-censored the lifetime and set `n_crit` to that step.
        # `CriterionScore` is the primary statistic of the decisive experiment;
        # a zero from an empty batch is the one value that cannot be a
        # measurement.
        return (float("nan"), 0)
    return (total / count, count)


@torch.no_grad()
def erank_probe(model, probe_tokens: torch.Tensor, genome, *, max_rows: int = 2048):
    """(3.9) on the fixed probe batch, per Site.

    SPEC: 03 §3.4.1

    `Act` in `R^{2048 x N_G}` is the **post-activation Site inner state**
    `relu(A x)^2` on a **fixed** probe batch of 2048 tokens held constant for the
    whole run. Spec v1 left `Act` unspecified, collided its name with `phi`, and
    logged every `T_log = 50` steps, which costs `L` SVDs of a `2048 x N_G`
    matrix at that cadence; §03.4 coarsens it to `10 * T_log` and §07.8 costs it.

    Returned per Site, because a collapse confined to one Site is exactly what an
    average would hide — and K8 is a trend test on the reported series.
    """
    model.eval()
    out = model(probe_tokens, genome=genome, want_metabolic=False, keep_z=True)
    ranks: List[float] = []
    for so in out.site_outputs:
        z = so.z.reshape(-1, so.z.shape[-1])
        if z.shape[0] > max_rows:
            z = z[:max_rows]
        act = torch.relu(z).square()
        # **[NORMATIVE — the SVD runs on CPU]** `Act` is `2048 x N_G` with
        # `N_G = 2048`, and cuSOLVER's batched Jacobi driver does not reliably
        # converge on it: it emits a convergence warning and falls back, and the
        # resulting asynchronous error state was observed to corrupt *later*
        # kernels, producing NaN in a subsequent forward pass with every
        # parameter still finite. LAPACK's `gesdd` on CPU is deterministic and
        # robust, and `erank` is a diagnostic logged once per `10 * T_log`
        # steps, so the transfer is affordable where the corruption is not.
        a = act.detach().float().cpu()
        if not torch.isfinite(a).all():
            # A diagnostic must not be the thing that ends the run, and a
            # crash inside `svdvals` says only "something was not finite".
            # Reporting NaN keeps the run alive and keeps the failure
            # attributable: `erank` is logged per Site, so a NaN names the Site.
            ranks.append(float("nan"))
            continue
        s = torch.linalg.svdvals(a)
        ranks.append(effective_rank(s.numpy()))
    model.train()
    return ranks


@dataclass
class EvalPoint:
    step: int
    tokens: int
    ce: float
    n_tokens_scored: int


@dataclass
class CurveTracker:
    """The held-out curve, `n_crit`, and the two slopes.

    SPEC: 07 §7.1

    `n_crit` is the first **evaluation step** at which held-out CE `<= L_star`.
    `T_val = max(50, floor(N_step/100))`, so it has at least 100-point
    resolution — spec v1 defined `n_crit` without specifying the evaluation
    cadence, leaving the entire ES fitness a function of an unstated constant.
    """

    cfg: SomaConfig
    L_star: Optional[float] = None
    points: List[EvalPoint] = field(default_factory=list)
    n_crit: Optional[int] = None
    censored: bool = True
    structural_events: List[Dict[str, object]] = field(default_factory=list)

    def record(self, step: int, tokens: int, ce: float, n_scored: int) -> None:
        """Record one held-out evaluation, and note if it reached `L_star`.

        A non-finite `ce`, or one computed over **no** scored tokens, is
        recorded on the curve — a gap in the evaluation is a fact about the run
        — but cannot satisfy the criterion (audit A-90).
        """
        self.points.append(EvalPoint(step, tokens, ce, n_scored))
        if (
            self.n_crit is None
            and self.L_star is not None
            and n_scored > 0
            and math.isfinite(ce)
            and ce <= self.L_star
        ):
            self.n_crit = step
            self.censored = False

    def annotate(self, step: int, kind: str, **fields: object) -> None:
        """Structural events are annotated on the curve, never smoothed away."""
        self.structural_events.append({"step": step, "kind": kind, **fields})

    @property
    def effective_n_crit(self) -> int:
        """`n_crit`, **right-censored** at `N_step` if `L_star` was not reached.

        §07.1: it is not scored as 0. §07.1 v3 additionally requires the
        consequence to be handled — a generation in which every member is
        censored has zero variance in `CriterionScore` and contributes no ES
        signal, so `compute_fitness` reports `censored` and `ES_STEP` falls back
        to `SlopeScore` (audit A-34).
        """
        return self.n_crit if self.n_crit is not None else self.cfg.N_step

    def early_slope(self) -> float:
        lo, hi = 0.02 * self.cfg.N_step, 0.20 * self.cfg.N_step
        pts = [(p.tokens, p.ce) for p in self.points if lo <= p.step <= hi]
        return log_slope(pts)

    def late_slope(self) -> float:
        lo = 0.50 * self.cfg.N_step
        pts = [(p.tokens, p.ce) for p in self.points if p.step >= lo]
        return log_slope(pts)

    def final_ce(self) -> float:
        return self.points[-1].ce if self.points else float("nan")

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_points": len(self.points),
            "n_crit": self.n_crit,
            "censored": self.censored,
            "L_star": self.L_star,
            "early_slope": self.early_slope(),
            "late_slope": self.late_slope(),
            "final_ce": self.final_ce(),
            "structural_events": len(self.structural_events),
        }


@dataclass
class Fitness:
    """(7.1) and its components."""

    criterion: float
    slope: float
    final: float
    birth: float
    fitness: float
    censored: bool
    substituted: bool
    """True when `CriterionScore` carried no signal and `SlopeScore` was used
    alone (§07.1, audit A-34)."""

    def as_dict(self) -> Dict[str, object]:
        return {
            "CriterionScore": self.criterion,
            "SlopeScore": self.slope,
            "FinalScore": self.final,
            "BirthScore": self.birth,
            "FITNESS": self.fitness,
            "censored": self.censored,
            "substituted": self.substituted,
        }


W_C, W_S, W_F = 0.5, 0.3, 0.2
"""§07.3. `BirthScore` carries **weight zero**."""


def compute_fitness(
    curve: CurveTracker,
    cfg: SomaConfig,
    *,
    L_birth: float,
    slope_ref: float,
    force_slope_only: bool = False,
) -> Fitness:
    """(7.1).

    SPEC: 07 §7.3

    ```
    CriterionScore = 1 - min(n_crit, N_step_ref) / N_step_ref
    SlopeScore     = clip( -early_slope / slope_ref , 0 , 2 ) / 2
    FinalScore     = ( L_rand - L_final ) / L_rand
    BirthScore     = ( L_rand - L_birth ) / L_rand          # diagnostic only
    FITNESS = w_c * CriterionScore + w_s * SlopeScore + w_f * FinalScore
    ```

    **[NORMATIVE — `BirthScore` is a bug detector]** It is *predicted* to be
    `~0`, because every tensor that touches data is soma and is re-randomised
    each lifetime, so inheriting gene index sets cannot improve step-0 loss. A
    significantly non-zero `BirthScore` means information is crossing the
    Weismann barrier and **fails gate `G-BARRIER`**. Spec v1 made it the primary
    statistic of the decisive experiment and of K1, where it had no mechanism by
    which to improve.

    `force_slope_only` implements §07.1's censoring fallback: when every member
    of a generation is censored, `CriterionScore` is identically 0 and carries no
    ES signal, so fitness is computed from `SlopeScore` alone and the
    substitution is logged.
    """
    L_rand = cfg.L_rand
    n_crit = curve.effective_n_crit
    criterion = 1.0 - min(n_crit, cfg.N_step_ref) / cfg.N_step_ref
    early = curve.early_slope()
    slope = min(max(-early / slope_ref if slope_ref else 0.0, 0.0), 2.0) / 2.0
    final = (L_rand - curve.final_ce()) / L_rand
    birth = (L_rand - L_birth) / L_rand

    if force_slope_only:
        fitness = slope
    else:
        fitness = W_C * criterion + W_S * slope + W_F * final

    return Fitness(
        criterion, slope, final, birth, fitness, curve.censored, force_slope_only
    )
