"""Enumerate the operator landscape exactly, and count its local optima.

The decisive experiment. S1 measured that the body-plan genome has enormous heritability
(broad-sense h2 = 0.991) and that every axis above the noise floor is monotone, so a
coordinate hill-climb reaches the optimum in at most four flips. That is a statement about
*axes*, measured one factor at a time. It cannot see interactions, and interactions are
where local optima live.

This enumerates the complete `2^L` operator cube -- every assignment of `attn`/`delta` to
every layer, including all six *placements* of two delta layers, whose differences are pure
placement -- and measures the topology directly.

WHAT IT TESTS. The corrected prediction, after Husain & Murugan (*Mol. Biol. Evol.* 37(10),
2020) showed that a scalar shared budget produces *global* epistasis and **reduces**
ruggedness rather than creating it:

    the landscape is rugged iff there is an INTERIOR OPTIMUM in the attn/delta mix,
    i.e. iff some hybrid beats both pure forms.

A monotone preference for one operator yields a single peak no matter how expensive the
other becomes. At `T=128` the S1 arms are monotone (`dddd` 2.7293 < `adad` 2.7698 <
`aaaa` 2.9142), so the prediction is one peak. At `T=1024`, attention buys exact recall a
fixed-size recurrent state cannot while costing quadratically more, which is the one knob
that plausibly creates an interior optimum -- and the hybrid-architecture literature reports
exactly that at scale.

Tokens per step are held fixed across `T`, so the operator choice is the thing that changes
and not the token budget.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gnomon.data.stream import Shard
from gnomon.utils.report import write_json
from screen.bauplan import GLOBAL, Bauplan
from screen.crn import WindowPool, eval_windows
from screen.harness import train_and_eval
from screen.landscape import Landscape
from screen.nondegeneracy import assert_axis_is_live


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=4, help="layers; the cube is 2^L")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--tokens-per-step", type=int, default=4096,
                    help="held fixed across T so batch adjusts, not the budget")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5,
                    help=">=5: gnomon/utils/stats.py:395 refuses BCa below five")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--n-h", type=int, default=4)
    ap.add_argument("--corpus", default="data/ts_4096")
    ap.add_argument("--pool-windows", type=int, default=64_000)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--pool-seed", type=int, default=20260901)
    ap.add_argument("--sim-runs", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    batch = max(1, a.tokens_per_step // a.T)
    out = a.out or f"runs/screen/topology-L{a.L}-T{a.T}.json"
    torch.backends.cuda.matmul.allow_tf32 = True

    plans: dict[tuple[int, ...], Bauplan] = {}
    for bits in itertools.product((0, 1), repeat=a.L):
        ops = tuple("delta" if b else "attn" for b in bits)
        plans[bits] = Bauplan(
            ops=ops, window=(GLOBAL,) * a.L, d_ff=(a.d_ff,) * a.L,
            bank=tuple(range(a.L)), n_h=(a.n_h,) * a.L, d_model=a.d_model)

    # ---- G-LIVE: every single-flip edge must change the model -----------------
    print("=" * 76)
    print(f"G-LIVE: all {a.L} single-flip edges from the all-attention corner")
    zero = (0,) * a.L
    for i in range(a.L):
        nb = zero[:i] + (1,) + zero[i + 1:]
        d = assert_axis_is_live(plans[zero], plans[nb], axis=f"operator@layer{i}",
                                T=a.T, device="cpu")
        print(f"  live  layer {i} attn->delta   rel.logit dist {d:.3e}")

    tr = Shard.load(a.corpus, "train")
    fi = Shard.load(a.corpus, "fitness")
    pool = WindowPool(tr, n_windows=a.pool_windows, T=a.T, pool_seed=a.pool_seed)
    ev = eval_windows(fi, n_windows=a.eval_windows, T=a.T)

    manifest = {
        "experiment": "operator-landscape", "config": vars(a), "batch": batch,
        "pool_digest": pool.digest(),
        "plans": {"".join("d" if b else "a" for b in g): {
            "summary": p.summary(),
            "unique_params": p.n_unique_params(),
            "flops_per_token": p.flops_per_token(T=a.T),
            "kv_cache_bytes": p.kv_cache_bytes(T=a.T)} for g, p in plans.items()},
    }
    print()
    print(f"corpus {a.corpus}  L={a.L}  T={a.T}  batch={batch} "
          f"(tokens/step {batch * a.T})  steps={a.steps}  seeds={a.seeds}")
    print(f"CRN pool digest {pool.digest()}")
    print(f"{len(plans)} plans x {a.seeds} seeds = {len(plans) * a.seeds} runs")
    print()

    rows, table = [], {}
    t0 = time.time()
    total = len(plans) * a.seeds
    for i, (bits, seed) in enumerate(
            itertools.product(sorted(plans), range(a.seeds)), start=1):
        r = train_and_eval(plans[bits], pool=pool, eval_win=ev, steps=a.steps,
                           batch=batch, seed=seed, lr=a.lr,
                           eval_every=max(1, a.steps // 2))
        table.setdefault(bits, []).append(r.final)
        rows.append(r.as_dict() | {"genome": list(bits),
                                   "label": "".join("d" if b else "a" for b in bits)})
        print(f"[{i:3d}/{total}] {''.join('d' if b else 'a' for b in bits)} "
              f"seed {seed}  CE {r.final:.4f}  {r.ms_per_step_median:6.1f} ms/step "
              f"(IQR {r.ms_per_step_iqr:5.1f})  {r.seconds:5.0f}s", flush=True)
        # Reproducibility metadata goes in the FIRST write, not the last: the S1 run
        # crashed after all 40 lifetimes and left an artifact that could not be
        # reproduced from itself.
        write_json(out, manifest | {"partial": True, "rows": rows})

    print(f"\nall runs done in {(time.time() - t0) / 60:.1f} min\n")

    # ---- the topology ---------------------------------------------------------
    land = Landscape(table)
    rep = land.report()

    print("=" * 76)
    print(f"OPERATOR LANDSCAPE, L={a.L}, T={a.T}  ({len(table)} genomes, "
          f"{a.seeds} seeds each)")
    print(f"  sigma_e {rep['sigma_e']:.4f} nats ({rep['sigma_e_df']} df)   "
          f"margin {rep['margin']:.4f}")
    print()
    print(f"  {'genome':>8s} {'mean CE':>9s} {'sd':>8s}")
    for g in sorted(table, key=lambda k: land.mean[k]):
        import statistics as st
        lbl = "".join("d" if b else "a" for b in g)
        print(f"  {lbl:>8s} {land.mean[g]:9.4f} {st.stdev(table[g]):8.4f}")

    print()
    print(f"  local optima          {rep['n_local_optima']}")
    for p in rep["peaks"]:
        lbl = "".join("d" if b else "a" for b in p["genome"])
        print(f"      {lbl}  CE {p['fitness']:.4f}  basin {p['basin_size']}/"
              f"{len(table)}  margin over neighbours {p['margin_over_neighbours']:+.4f}")
    print(f"  funnel ratio          {rep['funnel_ratio']:.2f}"
          f"   (>=0.80 = one dominant basin: rugged but navigable)")
    print(f"  additive R^2          {rep['additive_r2']:.4f}"
          f"   (1.0 = perfectly additive; a hill-climb is optimal)")
    se = rep["sign_epistasis"]
    print(f"  squares examined      {se['squares_examined']}"
          f"  (indistinguishable at the margin: {se['indistinguishable']})")
    print(f"  simple sign epistasis {se['simple_sign_epistasis']}")
    print(f"  RECIPROCAL sign epi.  {se['reciprocal_sign_epistasis']}"
          f"   -> peaks <= {se['peak_upper_bound_from_rse']} (Saona 2022)")
    w = rep["adaptive_walks"]
    print(f"  adaptive walk         mean {w['mean_length']:.2f} steps, max "
          f"{w['max_length']}, {w['fraction_reaching_global']:.0%} reach the global")
    print(f"  FDC                   {rep['fitness_distance_correlation']:+.3f}")
    print(f"  autocorrelation       "
          + " ".join(f"{x:+.2f}" for x in rep["autocorrelation"]))
    print()
    print(f"  VERDICT: {rep['verdict']}")

    # ---- searchers, at zero GPU cost -----------------------------------------
    print()
    print("=" * 76)
    print(f"SIMULATED SEARCH on the measured landscape ({a.sim_runs:,} runs each)")
    print("  NOTE: at 2^%d = %d vertices this space is nearly exhaustible, so random"
          % (a.L, len(table)))
    print("  search is a strong baseline by construction. Discriminating searchers")
    print("  needs a larger cube; this is reported for honesty, not as a comparison.")
    sims = {}
    for budget in (len(table) // 4, len(table) // 2):
        print(f"\n  budget {budget} evaluations")
        for nm in ("random", "rmhc", "restart_rmhc", "ga"):
            s = land.simulate(nm, budget=budget, runs=a.sim_runs, seed=1)
            sims[f"{nm}@{budget}"] = {
                "mean_best": s.mean_best, "sd_best": s.sd_best,
                "p_found_global": s.p_found_global,
                "mean_evals_to_global": s.mean_evals_to_global}
            print(f"    {nm:<13s} best {s.mean_best:.4f} +/- {s.sd_best:.4f}   "
                  f"P(global) {s.p_found_global:.3f}")

    write_json(out, manifest | {
        "rows": rows,
        "table": {"".join("d" if b else "a" for b in g): v for g, v in table.items()},
        "landscape": rep, "simulations": sims})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
