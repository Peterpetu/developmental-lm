#!/usr/bin/env python
"""M0 / M0-nano: the baseline gate, and the reference artifacts it produces.

SPEC: 07 §7.4.2
SPEC: 07 §7.1
SPEC: 08 §8.4

    python scripts/run_m0.py --config m1   --data data/ts_8192 --steps 12208
    python scripts/run_m0.py --config m-nano --data data/ts_4096

**[NORMATIVE]** No SOMA-specific component is enabled until M0 passes.

* Model: the §05 stack with the Site replaced by a dense `ReLU^2` FFN of inner
  width `N_G`, no germline, no units, no selection, no stream.
* Pass: stable (no NaN/Inf, no divergence), the numeric target met, and
  **wall-clock, tokens/s, peak memory and MFU recorded**.

M0 produces three artifacts the SOMA runs depend on, and §07.1 (audit A-33)
requires them **per configuration** — M-nano runs at `V = 4096, T_ctx = 128`,
where M1's are incomparable:

* `theta_m0`  — the frozen reference whose per-token losses §06.2 caches;
* `L_star`    — held-out CE at a token budget equal to the SOMA lifetime's;
* `slope_ref` — that run's early slope, the denominator of `SlopeScore`.

`T_life` is `[OPEN]` in §01.9 and is **measured here** (audit A-37); `G_gen` and
the whole M3 budget are recomputed from the measurement before the decisive
experiment is scheduled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.config import get_config, m0_config  # noqa: E402
from soma.data import TokenizedCorpus, TrainingStream  # noqa: E402
from soma.germline import Genome  # noqa: E402
from soma.train import LifetimeConfig, lifetime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="m1", choices=["m1", "m-nano"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(REPO / "runs"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--target-ce", type=float, default=None,
                    help="§07.4.2's numeric target in nats; defaults to 1.45 at "
                         "M1 and to none at M-nano (see below)")
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = get_config(args.config)
    cfg = m0_config(base)

    # **[NORMATIVE — the target is per configuration; audit A-70]** §07.4.2's
    # `CE <= 1.45` nats is stated for M1's setting: TinyStories V2, `V = 8192`,
    # `T_ctx = 512`, `D_tok = 8e8`, against published reference checkpoints on
    # that corpus. §07.1 (audit A-33) additionally requires an M0 **per
    # configuration**, and M-nano runs at `V = 4096`, `T_ctx = 128`. A
    # cross-entropy over a 4096-token vocabulary at a quarter of the context is
    # not comparable to one over 8192 at full context, and no published
    # reference exists for it.
    #
    # The M-nano M0 is therefore an **artifact-producing** run, not a gated one:
    # it exists to measure `L_star`, `slope_ref`, `theta_m0` and `T_life`, and
    # its pass condition is stability. Importing M1's number would be a bar
    # invented by transcription.
    target_ce = args.target_ce
    if target_ce is None and args.config == "m1":
        target_ce = 1.45
    if args.target_ce is not None and args.config != "m1":
        print(f"note: applying an explicit --target-ce {args.target_ce} at "
              f"{args.config}; §07.4.2's 1.45 is defined only at M1's vocabulary "
              f"and context.")
    rows = args.rows or max(1, cfg.B_tok // cfg.T_ctx)
    run_id = f"m0-{args.config}-s{args.seed}"

    train = TokenizedCorpus(args.data, "train")
    val = TokenizedCorpus(args.data, "validation")
    if train.index.vocab_size != cfg.V:
        print(
            f"vocabulary mismatch: corpus {train.index.vocab_size} vs config "
            f"{cfg.V}. §07.1 requires M0 at the SAME corpus, tokenizer and "
            f"context as the lifetimes it will score (audit A-33).",
            file=sys.stderr,
        )
        return 2

    evs = TrainingStream(cfg, val, None, rows_per_batch=8,
                         rng=np.random.default_rng(args.seed + 1))
    eval_batches = [evs.batch(0, 0.0) for _ in range(args.eval_batches)]

    genome = Genome.initialise(cfg, generator=torch.Generator().manual_seed(args.seed))
    lc = LifetimeConfig(
        cfg=cfg, run_id=run_id, runs_root=args.out, device=args.device,
        corpus=train, eval_batches=eval_batches, probe_tokens=None,
        L_star=None, slope_ref=1.0, rows_per_batch=rows,
        max_steps=args.steps, dense_sites=True, verbose=True,
        save_final_state=True,
    )

    print(f"M0 [{args.config}]  N_step={args.steps or cfg.N_step}  rows/batch={rows}")
    print(f"  corpus {train.index.n_tokens:,} tokens, vocab {cfg.V}, T_ctx {cfg.T_ctx}")
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    t0 = time.time()
    res = lifetime(genome, lc, seed=args.seed)
    wall = time.time() - t0

    peak_gb = (
        torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.0
    )
    tokens = res.metrics["tokens_seen"]
    n_par = sum(p.numel() for p in _model_params(cfg))
    flops = 6 * n_par * tokens
    mfu = flops / max(wall, 1e-9) / _peak_flops(args.device)

    final_ce = res.curve.final_ce()
    stable = bool(np.isfinite(final_ce)) and all(
        np.isfinite(p.ce) for p in res.curve.points
    )
    if target_ce is None:
        passed = stable
    else:
        passed = bool(stable and final_ce <= target_ce)

    report = {
        "config": args.config,
        "run_id": run_id,
        "seed": args.seed,
        "steps": res.steps_run,
        "tokens": tokens,
        "wall_seconds": wall,
        "tokens_per_second": tokens / max(wall, 1e-9),
        "peak_memory_gib": peak_gb,
        "n_parameters": n_par,
        "flops_6ND": flops,
        "mfu": mfu,
        "final_ce_nats": final_ce,
        "final_ppl": float(np.exp(final_ce)) if np.isfinite(final_ce) else None,
        "target_ce_nats": target_ce,
        "stable": stable,
        "target_applies": target_ce is not None,
        "passed": passed,
        # The three artifacts the SOMA runs consume (§07.1).
        "L_star": final_ce,
        # **[NORMATIVE — v3.1, audit A-81]** `L_star` at the **full** matched
        # budget is not necessarily reachable by a SOMA lifetime, and a criterion
        # no lifetime reaches makes `CriterionScore` identically 0 — which makes
        # the primary statistic of the decisive experiment a constant. §07.1
        # therefore selects `L_star` from this curve at the largest fraction a
        # generation-0 lifetime reaches with headroom. The whole curve is
        # published here so that choice is made from data and recorded, not
        # argued.
        "L_star_by_fraction": {
            f"{f:.2f}": _ce_at_fraction(res.curve, f)
            for f in (0.10, 0.25, 0.50, 0.75, 1.00)
        },
        "slope_ref": res.curve.early_slope(),
        "theta_m0_checkpoint": str(
            (Path(args.out) / run_id / "checkpoints" / "final.pt").resolve()
        ),
        "tokenizer_hash": train.index.tokenizer_hash,
        "corpus_hash": train.index.corpus_hash,
        # T_life is [OPEN] in §01.9 and measured here (audit A-37).
        "T_life_minutes_measured": wall / 60.0,
        "curve": [(p.step, p.tokens, p.ce) for p in res.curve.points],
    }
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== M0 report ===")
    for k in ("steps", "tokens", "wall_seconds", "tokens_per_second",
              "peak_memory_gib", "mfu", "final_ce_nats", "final_ppl",
              "L_star", "slope_ref", "T_life_minutes_measured", "passed"):
        print(f"  {k:24s} {report[k]}")
    print(f"  wrote {out_dir / 'm0_report.json'}")

    train.close()
    val.close()
    return 0 if passed else 1


def _ce_at_fraction(curve, frac: float) -> float:
    """Held-out CE at `frac` of the run's realised token budget.

    Nearest recorded point rather than an interpolation: `L_star` is compared to
    a *measured* SOMA curve, and a criterion set at a loss the baseline never
    actually recorded is a criterion nothing was ever observed to reach.
    """
    pts = [(p.tokens, p.ce) for p in curve.points]
    if not pts:
        return float("nan")
    target = frac * pts[-1][0]
    return float(min(pts, key=lambda t: abs(t[0] - target))[1])


def _model_params(cfg):
    from soma.model.model import SomaModel

    return list(SomaModel(cfg, dense_sites=True).parameters())


def _peak_flops(device: str) -> float:
    """Dense bf16 peak, for the MFU denominator (§07.8).

    §07.8 forbids reporting an efficiency claim from analytic FLOPs alone, so
    this is used **only** to express the measured wall-clock as a fraction of
    peak — the wall-clock itself is the reported quantity.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 1.0
    name = torch.cuda.get_device_name(0)
    # RTX 4080 SUPER: 52.2 TFLOP/s fp32, ~104 TFLOP/s bf16 dense (no sparsity).
    table = {"4080": 104e12, "4090": 165e12, "A100": 312e12, "H100": 989e12}
    for key, val in table.items():
        if key in name:
            return val
    return 104e12


if __name__ == "__main__":
    raise SystemExit(main())
