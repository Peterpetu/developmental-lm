"""S1: does a discrete body plan move the loss by more than seed noise does?

The first screen, and the template for the other three. It finishes work an
adversarial review of this project's own plan began: that review measured the
axes below at 1-5 seeds and found two of them beneath the noise floor, the
sharing graph algebraically excluded by matched-resource ablation, and the
receptive-field axis **numerically inert** because `M_PICO` had `window == T`.

What this script adds:

* every axis is checked live by `screen.nondegeneracy` before a GPU-hour is
  spent, so an inert axis is an exception and not a flat landscape;
* the receptive-field axis uses windows strictly below `T`, which is what makes
  it an axis at all;
* four seeds everywhere under common random numbers, so contrasts are paired;
* BCa intervals over seeds, the deterministic-price check, and G-SEARCHWORTHY.

`--steps` is the fitness horizon and it is not a detail: the same review measured
the operator advantage decaying x0.67-0.83 per doubling of training, so a short
proxy systematically overstates the value of the genome. Run it twice.
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
from screen.bauplan import Bauplan, GLOBAL
from screen.crn import WindowPool, eval_windows
from screen.harness import train_and_eval
from screen.nondegeneracy import assert_axis_is_live
from screen.variance import decompose, deterministic_term_check, searchworthy


def arms(T: int) -> dict[str, Bauplan]:
    """Ten plans spanning every axis of the body-plan genome.

    A design, not a sample: each arm differs from `base` along exactly one axis
    (or is the matched partner of another arm), so every contrast below is
    interpretable on its own.
    """
    base = Bauplan.uniform(4, "attn", d_ff=512, window=GLOBAL)
    # Param-matched sharing: 2 banks, widened so unique params return to base.
    # layer_params(d=128,h=4,f,attn) = 49412 + 384 f, so 2 banks match 4 at
    # f = (4*(49412+384*512) - 2*49412) / (2*384) = 1152.7 -> 1152.
    return {
        "base":            base,
        "dddd":            base.replace(ops=("delta",) * 4),
        "adad":            base.replace(ops=("attn", "delta", "attn", "delta")),
        "w64":             base.replace(window=(64,) * 4),
        "w32":             base.replace(window=(32,) * 4),
        "share_adj":       base.replace(bank=(0, 0, 1, 1)),
        "share_alt":       base.replace(bank=(0, 1, 0, 1)),
        "share_adj_wide":  base.replace(bank=(0, 0, 1, 1), d_ff=(1152,) * 4),
        "wideA":           base.replace(d_ff=(1024, 256, 1024, 256)),
        "wideB":           base.replace(d_ff=(256, 1024, 256, 1024)),
    }


#: `(label, baseline, other)`. Each isolates one axis at one matching convention.
CONTRASTS = [
    ("operator: attn -> delta (all four)",          "base", "dddd"),
    ("operator: attn -> delta (alternating)",       "base", "adad"),
    ("receptive field: global -> 64",               "base", "w64"),
    ("receptive field: global -> 32",               "base", "w32"),
    ("sharing vs none, FLOP-matched",               "base", "share_adj"),
    ("sharing PATTERN: adjacent -> alternating",    "share_adj", "share_alt"),
    ("sharing vs none, param-matched (FLOPs free)", "base", "share_adj_wide"),
    ("width PATTERN: A -> B (equal params, FLOPs)", "wideA", "wideB"),
    ("width: uniform -> patterned A",               "base", "wideA"),
]

#: Whether a hill-climb solves the axis. G-SEARCHWORTHY needs this and it cannot
#: be measured from a variance table -- it is a claim about the *shape* of the
#: landscape, so it is stated here explicitly and defended in the report.
HILL_CLIMBABLE = {
    "operator: attn -> delta (all four)": True,       # monotone d > ad > a
    "operator: attn -> delta (alternating)": True,
    "receptive field: global -> 64": True,            # monotone in span
    "receptive field: global -> 32": True,
    "sharing vs none, FLOP-matched": True,            # monotone in bank count
    "sharing PATTERN: adjacent -> alternating": False,  # combinatorial: which layers pair
    "sharing vs none, param-matched (FLOPs free)": True,
    "width PATTERN: A -> B (equal params, FLOPs)": False,  # combinatorial allocation
    "width: uniform -> patterned A": True,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--corpus", default="data/ts_4096")
    ap.add_argument("--pool-windows", type=int, default=64_000)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--pool-seed", type=int, default=20260901)
    ap.add_argument("--lam", type=float, default=0.0,
                    help="price coefficient, nats per unique parameter, for the "
                         "deterministic-term check; 0 uses the measured "
                         "exchange rate from this run")
    ap.add_argument("--out", default="runs/screen/s1.json")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    plans = arms(a.T)

    # ---- every axis must be live before anything is trained ----------------
    print("=" * 76)
    print("G-LIVE: no genome axis may be silently inert")
    live = {}
    for label, x, y in CONTRASTS:
        d = assert_axis_is_live(plans[x], plans[y], axis=label, T=a.T, device="cpu")
        live[label] = d
        print(f"  live  {label:<46s} rel.logit dist {d:.3e}")
    print()

    tr = Shard.load(a.corpus, "train")
    fi = Shard.load(a.corpus, "fitness")
    pool = WindowPool(tr, n_windows=a.pool_windows, T=a.T, pool_seed=a.pool_seed)
    ev = eval_windows(fi, n_windows=a.eval_windows, T=a.T)
    print(f"corpus {a.corpus}  T={a.T}  batch={a.batch}  steps={a.steps}  "
          f"seeds={a.seeds}")
    print(f"CRN pool: {a.pool_windows:,} windows, digest {pool.digest()}")
    print(f"eval: {a.eval_windows} fixed windows from the head of the fitness split")
    print()
    print(f"{'arm':<16s} {'uniq params':>12s} {'flops/tok':>11s} {'kv bytes':>10s}  plan")
    for name, p in plans.items():
        print(f"{name:<16s} {p.n_unique_params():>12,} "
              f"{p.flops_per_token(T=a.T):>11,} {p.kv_cache_bytes(T=a.T):>10,}  "
              f"{p.summary()}")
    print()

    # ---- the runs ----------------------------------------------------------
    table: dict[str, list[float]] = {k: [] for k in plans}
    seeds: dict[str, list[int]] = {k: [] for k in plans}
    mid: dict[str, list[float]] = {k: [] for k in plans}
    rows = []
    t_start = time.time()
    total = len(plans) * a.seeds
    for i, (name, seed) in enumerate(
            itertools.product(plans, range(a.seeds)), start=1):
        r = train_and_eval(plans[name], pool=pool, eval_win=ev, steps=a.steps,
                           batch=a.batch, seed=seed, lr=a.lr,
                           eval_every=max(1, a.steps // 2))
        table[name].append(r.final)
        seeds[name].append(seed)
        half = [v for s, v in r.curve if s == a.steps // 2]
        mid[name].append(half[0] if half else float("nan"))
        rows.append(r.as_dict() | {"arm": name})
        print(f"[{i:3d}/{total}] {name:<16s} seed {seed}  CE {r.final:.4f}  "
              f"{r.ms_per_step_median:6.1f} ms/step (IQR {r.ms_per_step_iqr:5.1f})  "
              f"{r.seconds:5.0f}s", flush=True)
        write_json(a.out, {"partial": True, "rows": rows})

    print(f"\nall runs done in {(time.time() - t_start) / 60:.1f} min\n")

    # ---- the decomposition -------------------------------------------------
    dec = decompose(table, seeds, contrasts=CONTRASTS)
    print("=" * 76)
    print("S1 VARIANCE DECOMPOSITION")
    print(dec.render())

    # Exchange rate: nats per unique parameter, from the width arms, which vary
    # parameters at fixed depth and operator. This is what a price coefficient
    # would have to be to be honest, and it is measured rather than chosen.
    import statistics
    dp = plans["wideA"].n_unique_params() - plans["base"].n_unique_params()
    dce = statistics.fmean(table["wideA"]) - statistics.fmean(table["base"])
    rate = dce / dp if dp else float("nan")
    lam = a.lam or abs(rate)
    price = [p.n_unique_params() for p in plans.values()]
    det = deterministic_term_check(price, lam, dec)
    print()
    print("=" * 76)
    print("DETERMINISTIC-TERM CHECK  (would a priced fitness optimise the price?)")
    print(f"  measured exchange rate: {rate:+.3e} nats per unique parameter")
    print(f"    i.e. 1 nat ~ {abs(1/rate):,.0f} parameters" if rate else "")
    print(f"  lambda used: {lam:.3e}")
    print(f"  Var(lambda*price) = {det['var_lambda_price']:.3e}")
    print(f"  Var_genetic(CE)   = {det['var_genetic_ce']:.3e}")
    print(f"  Var(seed noise)   = {det['var_seed_noise']:.3e}")
    print(f"  PRICE DOMINATES: {det['price_dominates']}  "
          f"(ratio {det['ratio_price_to_genetic']:.2f})")

    sw = searchworthy(dec, hill_climbable=HILL_CLIMBABLE)
    print()
    print("=" * 76)
    print("G-SEARCHWORTHY  (>=3 sigma_e AND not solvable by hill-climbing)")
    for r in sw["axes"]:
        mark = "QUALIFIES" if r["qualifies"] else "         "
        print(f"  {mark}  {r['contrast']:<46s} {r['sigma']:6.2f} sigma  "
              f"hill-climbable={r['hill_climbable']}")
    print(f"\n  S1 PASSES G-SEARCHWORTHY: {sw['passes']}")

    write_json(a.out, {
        "screen": "S1-bodyplan", "config": vars(a),
        "pool_digest": pool.digest(),
        "axis_live": live,
        "plans": {k: {"summary": p.summary(),
                      "unique_params": p.n_unique_params(),
                      "flops_per_token": p.flops_per_token(T=a.T),
                      "kv_cache_bytes": p.kv_cache_bytes(T=a.T)}
                  for k, p in plans.items()},
        "final": table, "seeds": seeds, "mid": mid,
        "decomposition": dec.as_dict(),
        "exchange_rate_nats_per_param": rate,
        "deterministic_term": det,
        "searchworthy": sw,
        "rows": rows,
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
