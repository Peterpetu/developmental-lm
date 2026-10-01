"""The behavioural regression suite and greedy decoding.

SPEC: 08 §8.2

**[NORMATIVE]** Runs from M1 onward at every checkpoint. Deterministic; gates
every capability claim. All tasks are serialised over the reserved symbol block
(§06.3.2), and all evaluation lengths lie **outside** the archive's admissible
range (§06.3.5).

| Suite | Train lengths | Eval lengths | `acc_random` | Pass bar (scaled) |
|---|---|---|---|---|
| `PARITY` | 3-32 | 40-256 | 0.5 | `>= 0.95` at 256 |
| `MODCOUNT` (`m_mod` 3, 5) | 3-32 | 40-256 | `1/m_mod` | `>= 0.80` |
| `PERM` (`p_grp = 5`, swaps) | 32 | 256 | `1/120` | `>= 0.80` sequence acc |
| `DYCK` (`b_types = 2`, `depth = 8`) | 3-32 | 40-256 | 0.5 | `>= 0.80` |
| `MQAR` | `P_mqar = 16` | `P_mqar = 32` | `1/alpha_sz` | `>= 0.90` |
| `COPY` with gap | 32 | 96 | `1/alpha_sz` | `>= 0.95` |

**[NORMATIVE]** Raw accuracy is never reported for these tasks: for parity,
chance is 0.5.

**[NORMATIVE — A-32]** At M-nano the suite runs a **reduced band**, eval lengths
`40..96`, with pass bars unchanged: `T_ctx = 128` cannot hold a length-256
prompt plus its answer, so the `40..256` rows are not computable there. K7 is
adjudicated at M1 on the full `{40, 64, 96, 128, 192, 256}` curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from soma.openended.tasks import (
    BOS,
    EOS,
    PAD,
    SEP,
    Instance,
    TaskParams,
    generate,
    verify,
)
from soma.utils.stats import scaled_accuracy

__all__ = [
    "SUITE",
    "SuiteSpec",
    "greedy_decode",
    "solve_rate",
    "run_suite",
    "SuiteResult",
]


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    params: TaskParams
    acc_random: float
    pass_bar: float
    eval_lengths: Tuple[int, ...] = ()
    """Empty means the family is scored at its own configured parameters rather
    than swept over `n_len`."""


ACC_RANDOM_TOL = 0.03
"""How far a declared `acc_random` may sit from its measured value (§08.2).

**[NORMATIVE — `acc_random` is the best *constant answer*; v3.1, audit A-87]**
`solve_rate` scores **exact match** on the whole answer, so the chance baseline
is the probability that the single most likely answer is right — not the chance
of any one token. Two declared values were wrong under that reading, both badly
and both in the direction of a harder bar:

| family | v3.0 declared | measured modal-answer frequency |
|---|---|---|
| `DYCK`  | `0.5` (as if the answer were a balanced/unbalanced bit) | **0.023** |
| `COPY`  | `1/alpha_sz = 0.0625` (the per-*token* chance) | **0.0003** |

`DYCK`'s answer is the residual **open-bracket stack**, 339 distinct values at
`b_types = 2, depth = 8`; `COPY`'s is a random string, so exact-match chance is
`alpha_sz^-n_copy ~ 2e-10`. Under `scaled_acc = (acc - r)/(1 - r)` a wrong `r`
moves the bar: `DYCK`'s `0.80` bar demanded raw `0.90` instead of raw `0.805`, so
a model at raw `0.85` reported `0.70` and failed K7 where the honest figure is
`0.847` and it passes.

The values are declared rather than measured at import — 7 families x 4000
instances is not an import-time cost — and `tests/regression` re-measures them
against this tolerance, so a declared value that drifts from the generator fails.
"""

ACC_RANDOM_PROTOCOL = "modal answer frequency over 4000 instances, seed 20260828"
"""The measurement any declared `acc_random` must reproduce."""


def _suite(eval_lengths: Sequence[int]) -> Tuple[SuiteSpec, ...]:
    el = tuple(eval_lengths)
    return (
        SuiteSpec("PARITY", TaskParams("PARITY", n_len=el[0]), 0.5, 0.95, el),
        SuiteSpec("MODCOUNT3", TaskParams("MODCOUNT", n_len=el[0], m_mod=3), 1 / 3, 0.80, el),
        SuiteSpec("MODCOUNT5", TaskParams("MODCOUNT", n_len=el[0], m_mod=5), 1 / 5, 0.80, el),
        SuiteSpec("PERM", TaskParams("PERM", p_grp=5, n_len=el[-1]), 1 / 120, 0.80),
        # 0.023, not 0.5: the answer is the residual open-bracket stack, not a
        # balanced/unbalanced bit (audit A-87).
        SuiteSpec("DYCK", TaskParams("DYCK", b_types=2, depth=8, n_len=el[0]), 0.023, 0.80, el),
        # MQAR generalises along P_mqar, not length: the archive is held out at
        # P_mqar <= 16 and the suite scores 32 (§06.3.5, audit A-30).
        SuiteSpec("MQAR", TaskParams("MQAR", P_mqar=32, alpha_sz=32, distract=8), 1 / 32, 0.90),
        # Exact-match chance is `alpha_sz ** -n_copy ~ 2e-10`, not the
        # per-token `1/alpha_sz` (audit A-87). Declared 0: at this magnitude
        # `scaled_acc` and raw accuracy agree to nine decimal places, and a
        # non-zero literal would imply a precision the estimate does not have.
        SuiteSpec("COPY", TaskParams("COPY", alpha_sz=16, n_len=min(96, el[-1]), gap=8), 0.0, 0.95),
    )


SUITE = _suite((40, 64, 96, 128, 192, 256))
"""The M1 suite. `run_suite` builds the M-nano variant from `cfg.eval_lengths`."""


@torch.no_grad()
def greedy_decode(
    model,
    prompt: Sequence[int],
    max_new: int,
    *,
    genome=None,
    T_ctx: int,
    device: torch.device | str = "cpu",
    eos_id: int = EOS,
) -> List[int]:
    """Greedy continuation after `SEP` (§06.3.3's `s_task` protocol).

    Temperature 0, `max_new` = the answer length, stop at the answer-span
    terminator. **No sampling**: `s_task` is the archive's central quantity and a
    sampled estimate would make `Learn = 4s(1-s)` a statement about the decoder's
    temperature rather than about the model.
    """
    model.eval()
    ctx = [BOS] + list(prompt) + [SEP]
    out: List[int] = []
    for _ in range(max_new):
        window = ctx[-T_ctx:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x, genome=genome, want_metabolic=False).logits
        nxt = int(logits[0, -1].argmax())
        if nxt == eos_id:
            break
        out.append(nxt)
        ctx.append(nxt)
    model.train()
    return out


def solve_rate(
    model,
    params: TaskParams,
    n_eval: int,
    rng: np.random.Generator,
    *,
    genome=None,
    T_ctx: int,
    device: torch.device | str = "cpu",
    max_new_slack: int = 2,
) -> float:
    """`s_task`: the fraction of `n_eval` fresh instances that verify exactly.

    SPEC: 06 §6.3.3
    """
    ok = 0
    for _ in range(n_eval):
        inst = generate(params, rng)
        cand = greedy_decode(
            model, inst.prompt, len(inst.answer) + max_new_slack,
            genome=genome, T_ctx=T_ctx, device=device,
        )
        ok += verify(inst.prompt, cand, params)
    return ok / max(1, n_eval)


@dataclass
class SuiteResult:
    """One suite row's outcome."""

    name: str
    accuracy: float
    scaled: float
    acc_random: float
    pass_bar: float
    passed: bool
    n_len: Optional[int] = None
    n_eval: int = 0
    skipped: bool = False
    """True when the instance does not serialise inside `T_ctx`.

    **[NORMATIVE — a skipped row is reported, not omitted; v3.1, audit A-91]**
    `_fits`' own docstring says a row that does not fit is "skipped and reported
    as skipped", and `run_suite` **omitted it entirely**. At M-nano that silently
    drops `COPY@96`, and a downstream `all(r.passed)` gate cannot tell a suite
    that passed six rows from one that passed five and lost the sixth."""

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_len": self.n_len,
            "accuracy": self.accuracy,
            "scaled_acc": self.scaled,
            "acc_random": self.acc_random,
            "pass_bar": self.pass_bar,
            "passed": self.passed,
            "skipped": self.skipped,
            "n_eval": self.n_eval,
        }


def run_suite(
    model,
    cfg,
    rng: np.random.Generator,
    *,
    genome=None,
    n_eval: int = 32,
    device: torch.device | str = "cpu",
) -> List[SuiteResult]:
    """Run §08.2's suite at `cfg.eval_lengths`.

    Every row reports **scaled** accuracy; §08.2 forbids reporting raw accuracy,
    because for parity chance is 0.5 and a raw 0.5 is no capability at all.
    """
    from dataclasses import replace as _replace

    results: List[SuiteResult] = []
    for spec in _suite(cfg.eval_lengths):
        if spec.eval_lengths:
            for L in spec.eval_lengths:
                p = _replace(spec.params, n_len=L)
                if not _fits(p, cfg, rng):
                    results.append(SuiteResult(
                        spec.name, float("nan"), float("nan"), spec.acc_random,
                        spec.pass_bar, False, L, 0, skipped=True,
                    ))
                    continue
                acc = solve_rate(
                    model, p, n_eval, rng, genome=genome,
                    T_ctx=cfg.T_ctx, device=device,
                )
                sc = scaled_accuracy(acc, spec.acc_random)
                results.append(
                    SuiteResult(spec.name, acc, sc, spec.acc_random,
                                spec.pass_bar, sc >= spec.pass_bar, L, n_eval)
                )
        else:
            if not _fits(spec.params, cfg, rng):
                results.append(SuiteResult(
                    spec.name, float("nan"), float("nan"), spec.acc_random,
                    spec.pass_bar, False, spec.params.n_len, 0, skipped=True,
                ))
                continue
            acc = solve_rate(
                model, spec.params, n_eval, rng, genome=genome,
                T_ctx=cfg.T_ctx, device=device,
            )
            sc = scaled_accuracy(acc, spec.acc_random)
            results.append(
                SuiteResult(spec.name, acc, sc, spec.acc_random, spec.pass_bar,
                            sc >= spec.pass_bar, spec.params.n_len, n_eval)
            )
    return results


def _fits(params: TaskParams, cfg, rng: np.random.Generator) -> bool:
    """Whether an instance serialises inside `T_ctx`.

    A row that does not fit is **skipped and reported as skipped**, never
    truncated: §08.2's v1 defect was setting `COPY` eval length 512 and `PERM`
    500 against `T_ctx = 512`, which cannot fit prompt **and** answer.
    """
    from soma.openended.tasks import serialised_length

    try:
        return serialised_length(params, rng) <= cfg.T_ctx
    except Exception:
        return False
