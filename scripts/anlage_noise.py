#!/usr/bin/env python
"""R1 NOISE: seed noise under common random numbers, pooled and per stratum.

SPEC: 08 §8.1
SPEC: 08 §8.2
SPEC: 00 §0.7

    python scripts/anlage_noise.py --lineage tinystories --stage 0 --tokens 40_000_000 --seeds 4

The instrument that runs before the machinery. Every arm of every later run at
this configuration is judged against the `sigma_e` this script writes, and
`scripts/anlage_analyse.py` refuses to evaluate a criterion at a configuration
whose noise file is absent (G-NOISE).
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import torch  # noqa: E402

from anlage.data import N_STRATA, stratum_names  # noqa: E402
from anlage.lifetime import train  # noqa: E402
from anlage.model import Phenotype, save_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from screen.variance import pooled_within_sd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=40_000_000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--compile", default="default")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" /
                       f"noise-{lin.name}-s{args.stage}{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin)
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T,
                     lr_muon=args.lr_muon, lr_emb=args.lr_emb,
                     lr_provenance=f"R1 NOISE grid value; see runs/anlage/lr-{lin.name}-s{args.stage}")
    log(f"R1 NOISE  {lin.name} {cfg.summary()}  tokens {args.tokens:,}  seeds {args.seeds}  "
        f"pool {pool.digest()}  eval test {test.digest} fit {fit.digest}")

    finals, per_strata, results = [], [], []
    for seed in range(args.seeds):
        model = Phenotype(cfg, lin.vocab, seed=seed)
        # The seed selects the initialisation *and* the order; the pool is shared.
        res = train(model, pool, test, tc, order_seed=seed, compile_mode=(
            None if args.compile == "none" else args.compile), log=log,
            eval_every_tokens=max(args.tokens // 4, 1))
        ce_fit, per_fit = __import__("anlage.lifetime", fromlist=["evaluate"]).evaluate(model, fit)
        finals.append(res.final_ce)
        per_strata.append((res.final_strata, per_fit))
        results.append({"seed": seed, **res.as_dict(), "fitness_ce": ce_fit,
                        "fitness_strata": per_fit})
        log(f"  seed {seed}: test {res.final_ce:.4f}  fit {ce_fit:.4f}  "
            f"{res.tokens_per_s:,.0f} tok/s  {res.ms_per_step_median:.0f} ms/step  "
            f"peak {res.peak_mem_gb:.2f} GB  {res.backend}")
        save_phenotype(model, out / f"seed{seed}.pt")
        del model
        torch.cuda.empty_cache()

    sd = statistics.stdev(finals) if len(finals) > 1 else float("nan")
    # per-stratum noise on the fitness split (the eps_c of §07.4) and on test
    strata_sd_fit = []
    strata_sd_test = []
    for c in range(N_STRATA):
        vals_t = [p[0][c] for p in per_strata]
        vals_f = [p[1][c] for p in per_strata]
        strata_sd_test.append(statistics.stdev(vals_t) if len(vals_t) > 1 and all(
            math.isfinite(v) for v in vals_t) else float("nan"))
        strata_sd_fit.append(statistics.stdev(vals_f) if len(vals_f) > 1 and all(
            math.isfinite(v) for v in vals_f) else float("nan"))
    report = manifest({
        "run": "R1-NOISE", "lineage": lin.name, "stage": args.stage,
        "config": cfg.summary(), "tokens": args.tokens, "seeds": args.seeds,
        "train_config": tc.__dict__, "pool_digest": pool.digest(),
        "pool_seed": args.pool_seed, "corpus": corpus.manifest(),
        "eval_test_digest": test.digest, "eval_fit_digest": fit.digest,
        "final_test_ce": finals, "sigma_e": sd, "sigma_e_df": len(finals) - 1,
        "mean_test_ce": statistics.fmean(finals),
        "sigma_e_strata_test": strata_sd_test, "sigma_e_strata_fit": strata_sd_fit,
        "strata": stratum_names(),
        "stratum_counts_test": test.stratum_counts().tolist(),
        "runs": results,
    })
    write_json(out / "noise.json", report)
    log(f"sigma_e = {sd:.4f} nats ({len(finals) - 1} df)  mean {statistics.fmean(finals):.4f}")
    log(f"wrote {out / 'noise.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
