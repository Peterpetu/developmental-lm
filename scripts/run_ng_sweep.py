#!/usr/bin/env python
"""The `N_G` scaling sweep, for kill criterion K9.

SPEC: 08 §8.6.1
SPEC: 10 §10.1

    python scripts/run_ng_sweep.py --data data/ts_4096 --steps 400

**The question.** `N_G` is the germline's gene count, and §01.2 fixes it at
`2048` at **every** configuration — that is what makes the germline scale-free,
which is the whole basis of the `d|Gamma|/d|U| = 0` claim. K9 asks whether that
is affordable: if the `CE`-minimising `N_G` grows in proportion to `d_model`,
then the germline is not scale-free in any useful sense and §09's scaling path
is wrong.

**The grid.** `N_G in {256, 512, 1024, 2048, 4096}` x `d_model in {96, 192, 384}`
at M-nano, `|U|_0` fixed, held-out CE at matched tokens. §08.6.1 requires the
grid to **bracket the default on both sides**: an optimum pinned at the grid's
upper edge says nothing about proportionality, which is why v2's
`{128, 256, 512, 1024}` — topping out below the value actually used — could not
have answered the question it was posed for.

**K9 fires** if the `CE`-minimising `N_G` grows proportionally to `d_model`. The
test is stated numerically here rather than by eye: fit `log2 N_G*(d) = a + b
log2 d` across the three widths and compare `b` against `1` (proportional) and
`0` (scale-free). With three widths this is a two-point-slope estimate with one
degree of freedom, so the verdict is reported with that stated and K9 is called
only when the fitted `b` is closer to `1` than to `0`.

Spec v1 declared this sweep "cheap to check" and "checked first" while **no
experiment anywhere tested it**. It is neither cheap nor free: 15 lifetimes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.config import M_NANO  # noqa: E402
from soma.data import TokenizedCorpus, TrainingStream  # noqa: E402
from soma.eval.regression import solve_rate  # noqa: E402
from soma.germline import Genome  # noqa: E402
from soma.openended.archive import N_EVAL  # noqa: E402
from soma.train import LifetimeConfig, lifetime  # noqa: E402

N_G_GRID = (256, 512, 1024, 2048, 4096)
D_MODEL_GRID = (96, 192, 384)


def _config(n_g: int, d_model: int, steps: int, B_tok: int):
    """M-nano at one grid point, with `|U|_0` held fixed (§08.6.1)."""
    # `n_m * d_v == d_model * e_v` is a hard constraint (§05.0); `d_v` is held at
    # 64 and `n_m` follows the width, which is the same relation M-nano and M1
    # already satisfy.
    n_m = max(1, d_model // 64)
    cfg = replace(
        M_NANO,
        name=f"ng{n_g}-d{d_model}",
        d_model=d_model,
        n_m=n_m,
        d_v=d_model // n_m,
        germline=replace(M_NANO.germline, N_G=n_g),
        B_tok=B_tok,
        D_tok=B_tok * steps,
    )
    cfg.validate()
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(REPO / "runs" / "ng_sweep"))
    ap.add_argument("--steps", type=int, default=1400,
                    help="N_step per grid point; the token budget is matched "
                         "across the grid by construction")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--b-tok", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-g", type=int, nargs="+", default=list(N_G_GRID))
    ap.add_argument("--d-model", type=int, nargs="+", default=list(D_MODEL_GRID))
    args = ap.parse_args()

    default_n_g = M_NANO.germline.N_G
    if not (min(args.n_g) < default_n_g < max(args.n_g)):
        print(
            f"refusing to run: §08.6.1 requires the grid to bracket the default "
            f"N_G = {default_n_g} on BOTH sides. Grid {sorted(args.n_g)} does not, "
            f"so an optimum at its edge could not distinguish 'the optimum is "
            f"there' from 'the grid stopped there' — which is exactly the defect "
            f"v2's grid had.",
            file=sys.stderr,
        )
        return 2

    train = TokenizedCorpus(args.data, "train")
    val = TokenizedCorpus(args.data, "validation")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_of: dict = {}
    t0 = time.time()
    for d_model in args.d_model:
        for n_g in args.n_g:
            cfg = _config(n_g, d_model, args.steps, args.b_tok)
            if train.index.vocab_size != cfg.V:
                print(f"vocabulary mismatch: {train.index.vocab_size} vs {cfg.V}",
                      file=sys.stderr)
                return 2
            rows = args.rows or max(1, cfg.B_tok // cfg.T_ctx)
            evs = TrainingStream(cfg, val, None, rows_per_batch=8,
                                 rng=np.random.default_rng(args.seed + 1))
            eval_batches = [evs.batch(0, 0.0) for _ in range(8)]
            probe = torch.from_numpy(
                train.gather(train.windows(cfg.T_ctx)[:8], cfg.T_ctx)[0]
            )
            genome = Genome.initialise(
                cfg, generator=torch.Generator().manual_seed(args.seed),
                lineage_id=cfg.name,
            )

            def solve_rate_fn(model, gm, params, n_eval=N_EVAL):
                return solve_rate(
                    model, params, n_eval, np.random.default_rng(args.seed),
                    genome=gm, T_ctx=cfg.T_ctx, device=args.device,
                )

            lc = LifetimeConfig(
                cfg=cfg, run_id=cfg.name, runs_root=args.out, device=args.device,
                corpus=train, eval_batches=eval_batches, probe_tokens=probe,
                L_star=4.0, slope_ref=1.0, rows_per_batch=rows,
                verbose=False, log_to_disk=True,
            )
            t = time.time()
            res = lifetime(genome, lc, seed=args.seed, solve_rate_fn=solve_rate_fn)
            ce = res.curve.points[-1].ce
            rows_of.setdefault(d_model, {})[n_g] = ce
            print(f"  d_model={d_model:4d}  N_G={n_g:5d}  CE={ce:.4f}  "
                  f"({time.time() - t:.0f}s, {res.metrics['tokens_seen']:,} tokens)",
                  flush=True)

    # --- K9's verdict, stated numerically -----------------------------------
    argmins = {d: min(v, key=v.get) for d, v in rows_of.items()}
    widths = sorted(argmins)
    edge = [d for d in widths
            if argmins[d] in (min(args.n_g), max(args.n_g))]
    b = float("nan")
    if len(widths) >= 2:
        x = np.log2(np.array(widths, dtype=float))
        y = np.log2(np.array([argmins[d] for d in widths], dtype=float))
        b = float(np.polyfit(x, y, 1)[0])

    fires = bool(np.isfinite(b) and abs(b - 1.0) < abs(b - 0.0))
    report = {
        "grid_n_g": args.n_g,
        "grid_d_model": args.d_model,
        "steps": args.steps,
        "B_tok": args.b_tok,
        "seed": args.seed,
        "held_out_ce": {str(d): {str(n): c for n, c in v.items()}
                        for d, v in rows_of.items()},
        "argmin_n_g": {str(d): int(n) for d, n in argmins.items()},
        "log2_slope_b": b,
        "K9_fires": fires,
        "argmin_at_grid_edge": [int(d) for d in edge],
        "wall_seconds": time.time() - t0,
        "note": (
            "b = 1 is proportional growth (K9 fires); b = 0 is scale-free. "
            "Three widths give one degree of freedom, so the estimate is weak "
            "by construction and is reported as such. Any width whose argmin "
            "sits at a grid edge is listed: for those, the grid — not the "
            "model — may be what determined the optimum."
        ),
    }
    path = out_dir / "ng_sweep_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  argmin N_G by width: {report['argmin_n_g']}")
    print(f"  log2 slope b = {b:.3f}   K9 fires = {fires}")
    if edge:
        print(f"  WARNING: argmin at a grid edge for d_model in {edge}; the "
              f"grid may be what determined the optimum, not the model.")
    print(f"  wrote {path}")

    train.close()
    val.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
