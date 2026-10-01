#!/usr/bin/env python
"""R6 MIX: the regressed programme `mu*` against the corpus mixture, at the full stage-0
budget, paired by seed.

SPEC: 12 §12.5
SPEC: 12 §12.8 (K-MIX)
SPEC: 12 §12.9

    python scripts/anlage_mix.py --live runs/anlage/live-tinystories-s0/live.json --seeds 0 1 2 3

Both arms train from the free initialisation of seed `s` on the classified pool (the same
200,000 windows as R1), 25M tokens, the R1 recipe; only the sampling mixture differs.
K-MIX fires when `mu*` is not better than `mu^corpus` by `>= 3 sigma_e` (paired, `t` at
3-4 seeds).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402
from anlage_analyse import paired  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import ClassifiedPool, classify_documents, classify_eval_windows  # noqa: E402
from anlage.lifetime import evaluate, per_token_loss, train  # noqa: E402
from anlage.model import Phenotype, save_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402


def per_class_ce(model, ev, wc, n_classes):
    tot = torch.zeros(n_classes, dtype=torch.float64)
    cnt = torch.zeros(n_classes, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, ev.windows.shape[0], 16):
            w = ev.windows[i:i + 16]
            ptl = per_token_loss(model, w).float().cpu()
            for j in range(w.shape[0]):
                c = int(wc[i + j])
                tot[c] += float(ptl[j].sum())
                cnt[c] += ptl.shape[1]
    return (tot / cnt.clamp_min(1)).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--live", default="runs/anlage/live-tinystories-s0/live.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["corpus", "star"])
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--noise", default="runs/anlage/noise-tinystories-s0/noise.json")
    ap.add_argument("--save", action="store_true", help="keep the adults (for R8's warm start)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(0)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"mix-{lin.name}-s0{('-' + args.tag) if args.tag else ''}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    live = json.loads(pathlib.Path(args.live).read_text())
    corpus = load_corpus(lin)
    cls_train, man = classify_documents(corpus.train, corpus.rank)
    cls_test, _ = classify_documents(corpus.test, corpus.rank, manifest=man)
    n_C = man["n_classes"]
    pool = ClassifiedPool(corpus.train, cls_train, n_C, n_windows=args.pool_windows, T=lin.T,
                          pool_seed=args.pool_seed, device=device)
    if pool.class_digest() != live["class_digest"]:
        raise RuntimeError("the class partition differs from R3's")
    mixtures = {"corpus": torch.tensor(live["mu_0"], dtype=torch.float64),
                "star": torch.tensor(live["mu_star"], dtype=torch.float64),
                "uniform": torch.full((n_C,), 1.0 / n_C, dtype=torch.float64)}
    test, fit = eval_sets(corpus, lin, device=device)
    cls_fit, _ = classify_documents(corpus.fitness, corpus.rank, manifest=man)
    wc_test = classify_eval_windows(test, [corpus.fitness, corpus.test], [cls_fit, cls_test])
    sigma_e = json.loads(pathlib.Path(args.noise).read_text())["sigma_e"]
    log(f"R6 MIX  {lin.name} {cfg.summary()}  tokens {args.tokens:,}  seeds {args.seeds}  arms {args.arms}  "
        f"mu_corpus {[round(x, 4) for x in mixtures['corpus'].tolist()]}  mu* {[round(x, 4) for x in mixtures['star'].tolist()]}  "
        f"sigma_e {sigma_e:.4f}")
    results = {a: [] for a in args.arms}
    for seed in args.seeds:
        for arm in args.arms:
            model = Phenotype(cfg, lin.vocab, seed=seed)
            tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon,
                             lr_emb=args.lr_emb, lr_provenance="R1 (R2 not run)")
            res = train(model, pool.at(mixtures[arm]), test, tc, order_seed=seed, compile_mode=None, log=None)
            ce_fit, strata = evaluate(model, fit)
            classes = per_class_ce(model, test, wc_test, n_C)
            results[arm].append({"seed": seed, "test_ce": res.final_ce, "fitness_ce": ce_fit, "strata": strata,
                                 "classes_test": classes, "tokens_per_s": res.tokens_per_s})
            log(f"  seed {seed} {arm:>7}: test {res.final_ce:.4f}  fit {ce_fit:.4f}  classes {[round(x, 3) for x in classes]}")
            if args.save:
                save_phenotype(model, out / f"{arm}-seed{seed}.pt")
            del model
            torch.cuda.empty_cache()
    contrasts = {}
    if "corpus" in results:
        base = {r["seed"]: r["test_ce"] for r in results["corpus"]}
        for arm in args.arms:
            if arm == "corpus":
                continue
            a = [base[r["seed"]] for r in results[arm]]
            b = [r["test_ce"] for r in results[arm]]
            est = paired(a, b)
            ci_hi = est.get("ci_high")
            k_pass = bool(est["mean"] is not None and est["mean"] <= -3 * sigma_e and ci_hi is not None and ci_hi < 0)
            contrasts[arm] = {"deltas": [y - x for x, y in zip(a, b)], **est,
                              "in_sigma_e": (est["mean"] / sigma_e) if est["mean"] is not None else None,
                              "k_mix_pass": k_pass}
            lo, hi = est.get("ci_low"), est.get("ci_high")
            log(f"  {arm} - corpus: {est['mean']:+.4f} "
                + (f"[{lo:+.4f}, {hi:+.4f}] " if lo is not None else "")
                + f"= {est['mean'] / sigma_e:+.1f} sigma_e  ({est['method']})  K-MIX {'PASS' if k_pass else 'FIRES'}")
    rec = {"run": "R6-MIX", "lineage": lin.name, "config": cfg.summary(), "tokens": args.tokens, "seeds": args.seeds,
           "mixtures": {k: v.tolist() for k, v in mixtures.items()}, "live": args.live, "classes": man,
           "pool_digest": pool.digest(), "class_digest": pool.class_digest(), "sigma_e": sigma_e,
           "results": results, "contrasts": contrasts}
    write_json(out / "mix.json", manifest(rec))
    log(f"wrote {out / 'mix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
