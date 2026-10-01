#!/usr/bin/env python
"""R3 LIVE: is the mixture axis alive? Siblings in the Dirichlet neighbourhood of the
corpus mixture, from one warm start, against the protocol's own seed noise.

SPEC: 12 §12.4
SPEC: 12 §12.8
SPEC: 12 §12.9

    python scripts/anlage_live.py --checkpoint runs/anlage/noise-tinystories-s0/seed0.pt \\
        --tokens-sib 6250000 --K 8 --alphas 200 50 10 --centre-seeds 4

Arms (all from the same warm start, the same pool, CRN by order seed):

  * centre: the corpus mixture `mu_0`, order seeds `0..centre_seeds-1`  -> `sigma_e(q)` of
    *this* protocol (a 6.25M-token continuation), the yardstick of G-LIVE-mu;
  * for each `alpha`: `K - 1` mixtures `mu_k ~ Dirichlet(alpha mu_0)` (clipped), order seed 0;
  * the `eta` axis: `mu_0` at learning-rate multipliers `{0.5, 2}`, order seed 0 (G-LIVE-eta).

For every arm: test CE, fitness CE per output stratum `q` (the lexicase cases) and per data
class. G-LIVE-mu passes at `alpha` when some stratum's spread across the `alpha`-siblings
exceeds `3 sigma_e(q)`; the surrogate (12.1) is fit on all `(mu_k, F_k)` and its trust-region
optimum `mu*` (12.2) is written for R6 MIX.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import (N_STRATA, ClassifiedPool, classify_documents, classify_eval_windows)  # noqa: E402
from anlage.lifetime import evaluate, per_token_loss, train  # noqa: E402
from anlage.model import load_phenotype, save_phenotype  # noqa: E402
from anlage.program import (ALPHA_0, TAU, dirichlet_neighbourhood, fit_surrogate,  # noqa: E402
                            solve_trust_region)
from anlage.recipe import TrainConfig  # noqa: E402
from screen.variance import pooled_within_sd  # noqa: E402


def per_class_ce(model, ev, wc, n_classes, device):
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
    ap.add_argument("--checkpoint", default="runs/anlage/noise-tinystories-s0/seed0.pt")
    ap.add_argument("--tokens-sib", type=int, default=6_250_000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+", default=[200.0, 50.0, 10.0])
    ap.add_argument("--centre-seeds", type=int, default=4)
    ap.add_argument("--lr-mults", type=float, nargs="*", default=[0.5, 2.0])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda"
    lin = get_lineage(args.lineage)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"live-{lin.name}-s0{('-' + args.tag) if args.tag else ''}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    cls_train, man = classify_documents(corpus.train, corpus.rank)
    cls_fit, _ = classify_documents(corpus.fitness, corpus.rank, manifest=man)
    n_C = man["n_classes"]
    pool = ClassifiedPool(corpus.train, cls_train, n_C, n_windows=args.pool_windows, T=lin.T,
                          pool_seed=args.pool_seed, device=device)
    mu_0 = pool.corpus_mixture
    test, fit = eval_sets(corpus, lin, device=device)
    cls_test, _ = classify_documents(corpus.test, corpus.rank, manifest=man)
    wc_fit = classify_eval_windows(fit, [corpus.fitness, corpus.test], [cls_fit, cls_test])
    log(f"R3 LIVE  {lin.name}  warm start {args.checkpoint}  tokens/sibling {args.tokens_sib:,}  K {args.K}  "
        f"alphas {args.alphas}  centre seeds {args.centre_seeds}  mu_0 {[round(x, 4) for x in mu_0.tolist()]}")

    def run(mu, order_seed, lr_mult=1.0, label=""):
        model = load_phenotype(args.checkpoint, device=device)
        tc = TrainConfig(tokens=args.tokens_sib, batch_final=args.batch, T=lin.T,
                         lr_muon=args.lr_muon * lr_mult, lr_emb=args.lr_emb * lr_mult,
                         lr_provenance="R1 (continuation; R2 not run)")
        res = train(model, pool.at(mu), test, tc, order_seed=order_seed, compile_mode=None, log=None)
        ce_fit, strata = evaluate(model, fit)
        classes = per_class_ce(model, fit, wc_fit, n_C, device)
        fname = "arm_" + label.replace(" ", "_").replace("|", "") + ".pt"
        save_phenotype(model, out / fname)
        rec = {"label": label, "mu": [float(x) for x in mu], "order_seed": order_seed, "lr_mult": lr_mult,
               "test_ce": res.final_ce, "fitness_ce": ce_fit, "strata": strata, "classes": classes,
               "tokens_per_s": res.tokens_per_s, "checkpoint": str(out / fname)}
        log(f"  {label:>14}  test {res.final_ce:.4f}  fit {ce_fit:.4f}  classes {[round(x, 3) for x in classes]}")
        return rec

    arms = []
    # centre replicates: the protocol's noise
    for s in range(args.centre_seeds):
        arms.append(run(mu_0, s, label=f"centre s{s}"))
    centre = [a for a in arms if a["label"].startswith("centre")]
    S = np.array([a["strata"] for a in centre], dtype=float)          # (n, Q)
    Cc = np.array([a["classes"] for a in centre], dtype=float)
    sigma_q = np.nanstd(S, axis=0, ddof=1)
    sigma_c = Cc.std(axis=0, ddof=1)
    sigma_test = float(np.std([a["test_ce"] for a in centre], ddof=1))
    log(f"  protocol noise: test {sigma_test:.4f}  strata {[round(float(x), 4) for x in sigma_q]}  "
        f"classes {[round(float(x), 4) for x in sigma_c]}")
    # the mixture neighbourhoods
    live = {}
    for ai, alpha in enumerate(args.alphas):
        nb = dirichlet_neighbourhood(mu_0, alpha, args.K, seed=1000 + ai)
        recs = []
        for k in range(1, args.K):
            recs.append(run(nb[k], 0, label=f"a{alpha:g} k{k}"))
        arms += recs
        S_a = np.array([r["strata"] for r in recs], dtype=float)
        C_a = np.array([r["classes"] for r in recs], dtype=float)
        base_S = S[0]                                     # centre at order seed 0
        base_C = Cc[0]
        dev_q = np.nanmax(np.abs(S_a - base_S), axis=0)   # largest sibling deviation per stratum
        dev_c = np.abs(C_a - base_C).max(axis=0)
        ratio_q = dev_q / np.where(sigma_q > 0, sigma_q, np.nan)
        ratio_c = dev_c / np.where(sigma_c > 0, sigma_c, np.nan)
        spread_test = float(np.max(np.abs(np.array([r["test_ce"] for r in recs]) - centre[0]["test_ce"])))
        alive = bool(np.nanmax(ratio_q) >= 3.0)
        live[f"{alpha:g}"] = {"max_dev_strata": dev_q.tolist(), "ratio_strata": ratio_q.tolist(),
                              "max_dev_classes": dev_c.tolist(), "ratio_classes": ratio_c.tolist(),
                              "spread_test": spread_test, "spread_test_sigma": spread_test / sigma_test if sigma_test > 0 else None,
                              "alive_3sigma": alive}
        log(f"  alpha {alpha:g}: max |dCE| per stratum / sigma_e(q) = {[round(float(x), 1) for x in ratio_q]}  "
            f"per class {[round(float(x), 1) for x in ratio_c]}  test spread {spread_test:.4f} "
            f"({spread_test / sigma_test if sigma_test > 0 else float('nan'):.1f} sigma)  alive: {alive}")
    # the eta axis
    eta_arms = []
    for m in args.lr_mults:
        eta_arms.append(run(mu_0, 0, lr_mult=m, label=f"lr x{m:g}"))
    arms += eta_arms
    eta_live = {f"{a['lr_mult']:g}": {"d_test": a["test_ce"] - centre[0]["test_ce"],
                                       "d_test_sigma": (a["test_ce"] - centre[0]["test_ce"]) / sigma_test if sigma_test > 0 else None}
                for a in eta_arms}
    # the surrogate and the trust-region step (for R6 MIX)
    mus = torch.tensor([a["mu"] for a in arms if a["lr_mult"] == 1.0], dtype=torch.float64)
    F = torch.tensor([[x if x == x else 0.0 for x in a["strata"]] for a in arms if a["lr_mult"] == 1.0], dtype=torch.float64)
    w = torch.tensor(json.loads((REPO / "runs/anlage/noise-tinystories-s0/noise.json").read_text())["stratum_counts_test"], dtype=torch.float64) \
        if (REPO / "runs/anlage/noise-tinystories-s0/noise.json").exists() else torch.ones(N_STRATA, dtype=torch.float64)
    w = w / w.sum()
    sur = fit_surrogate(mus, F)
    mu_star, info = solve_trust_region(sur, w, mu_0, tau=TAU)
    log(f"  surrogate: {'linear' if sur.linear_only else 'quadratic'} on {sur.n_samples} samples, LOO RMSE {sur.loo_rmse:.4f} "
        f"(sigma_e test {sigma_test:.4f});  mu* {[round(x, 4) for x in mu_star.tolist()]}  predicted {info['value']:.4f}  "
        f"distance {info['distance']:.3f}  boundary {info['on_boundary']}")
    rec = {"run": "R3-LIVE", "lineage": lin.name, "checkpoint": args.checkpoint, "tokens_sib": args.tokens_sib,
           "K": args.K, "alphas": args.alphas, "centre_seeds": args.centre_seeds, "classes": man,
           "mu_0": mu_0.tolist(), "pool_digest": pool.digest(), "class_digest": pool.class_digest(),
           "sigma_test": sigma_test, "sigma_strata": sigma_q.tolist(), "sigma_classes": sigma_c.tolist(),
           "arms": arms, "live_mu": live, "live_eta": eta_live,
           "surrogate": {"linear_only": sur.linear_only, "n_samples": sur.n_samples, "loo_rmse": sur.loo_rmse},
           "mu_star": mu_star.tolist(), "mu_star_info": info, "stratum_weights": w.tolist()}
    write_json(out / "live.json", manifest(rec))
    log(f"wrote {out / 'live.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
