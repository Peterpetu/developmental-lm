#!/usr/bin/env python
"""R7 POP: lexicase over strata and the task-vector merge on R3's mixture siblings,
against the best sibling and against SINGLE at equal compute.

SPEC: 12 §12.3 (lines 7-8)
SPEC: 12 §12.8 (K-POP)
SPEC: 07 §7.4

    python scripts/anlage_pop2.py --live runs/anlage/live-tinystories-s0/live.json

For every `alpha` of R3 the population is the centre (order seed 0) plus the `K - 1`
Dirichlet siblings, all continued 6.25M tokens from one warm start. Parents are chosen by
ε-lexicase with `eps_q = sigma_e(q)` of R3's protocol, merged by (7.1)-(7.2) on windows
disjoint from the scoring set (S8), and the child is scored on the test split against the
best sibling (by fitness CE) and against SINGLE: one continuation of `K × 6.25M` tokens at
the corpus mixture from the same warm start, the same compute as the whole population.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, fit_windows, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import ClassifiedPool, EvalSet, classify_documents  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import load_phenotype, save_phenotype  # noqa: E402
from anlage.population import merge, select_parents  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from gnomon.utils.rng import derive_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--live", default="runs/anlage/live-tinystories-s0/live.json")
    ap.add_argument("--alphas", type=float, nargs="*", default=None, help="default: all of R3's")
    ap.add_argument("--n-par", type=int, default=4)
    ap.add_argument("--merge-steps", type=int, default=200)
    ap.add_argument("--single-tokens", type=int, default=None, help="default: K x tokens_sib")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(0)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"pop2-{lin.name}-s0")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    live = json.loads(pathlib.Path(args.live).read_text())
    alphas = args.alphas or live["alphas"]
    K = int(live["K"])
    tokens_sib = int(live["tokens_sib"])
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    fitw = fit_windows(corpus, lin, n=64, device=device)
    sigma_test = float(live["sigma_test"])
    eps_q = [float(x) if x is not None else float("nan") for x in live["sigma_strata"]]
    arms = {a["label"]: a for a in live["arms"]}
    warm = load_phenotype(live["checkpoint"], device=device)
    W0 = {k: v.detach().clone() for k, v in warm.weights().items()}
    log(f"R7 POP  {lin.name}  from {live['checkpoint']}  K {K}  tokens/sib {tokens_sib:,}  alphas {alphas}  "
        f"sigma_test(protocol) {sigma_test:.4f}")
    per_alpha = {}
    for alpha in alphas:
        labels = ["centre s0"] + [f"a{alpha:g} k{k}" for k in range(1, K)]
        sibs, scores, tests, fits = [], [], [], []
        for lab in labels:
            a = arms[lab]
            m = load_phenotype(a["checkpoint"], device=device)
            sibs.append(m)
            scores.append(a["strata"])
            tests.append(a["test_ce"])
            fits.append(a["fitness_ce"])
        rng = np.random.default_rng(derive_seed(int(alpha), "lexicase"))
        parents = select_parents(scores, eps_q, args.n_par, rng)
        distinct = sorted(set(parents))
        child, mrep = merge([sibs[i] for i in distinct], W0, fitw, fit, sigma_test,
                            steps=args.merge_steps, log=log)
        ce_child, _ = evaluate(child, test)
        best = int(np.argmin(fits))
        # a merge of the whole population, for the record
        all_child, all_rep = merge(sibs, W0, fitw, fit, sigma_test, steps=args.merge_steps, log=None)
        ce_all, _ = evaluate(all_child, test)
        per_alpha[f"{alpha:g}"] = {
            "labels": labels, "parents": distinct, "lexicase_draws": parents, "merge": mrep.as_dict(),
            "child_test_ce": ce_child, "best_sibling": labels[best], "best_sibling_test_ce": tests[best],
            "best_sibling_fitness_ce": fits[best], "centre_test_ce": tests[0],
            "merge_all_test_ce": ce_all, "merge_all": all_rep.as_dict(),
            "delta_vs_best": ce_child - tests[best], "delta_vs_best_sigma": (ce_child - tests[best]) / sigma_test}
        log(f"  alpha {alpha:g}: parents {[labels[i] for i in distinct]}  chosen {mrep.chosen}  "
            f"child test {ce_child:.4f}  best sibling {labels[best]} {tests[best]:.4f}  centre {tests[0]:.4f}  "
            f"merge-all {ce_all:.4f} ({all_rep.chosen})")
        save_phenotype(child, out / f"child-a{alpha:g}.pt")
        del sibs, child, all_child
        torch.cuda.empty_cache()
    # SINGLE at equal compute
    single_tokens = args.single_tokens or K * tokens_sib
    cls_train, man = classify_documents(corpus.train, corpus.rank)
    pool = ClassifiedPool(corpus.train, cls_train, man["n_classes"], n_windows=args.pool_windows, T=lin.T,
                          pool_seed=args.pool_seed, device=device)
    mu_0 = torch.tensor(live["mu_0"], dtype=torch.float64)
    single = load_phenotype(live["checkpoint"], device=device)
    tc = TrainConfig(tokens=single_tokens, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon,
                     lr_emb=args.lr_emb, lr_provenance="R1 (continuation; R2 not run)")
    res = train(single, pool.at(mu_0), test, tc, order_seed=0, compile_mode=None, log=None)
    ce_single_fit, _ = evaluate(single, fit)
    log(f"  SINGLE ({single_tokens:,} tokens at mu_0): test {res.final_ce:.4f}  fit {ce_single_fit:.4f}")
    save_phenotype(single, out / "single.pt")
    for a, r in per_alpha.items():
        r["single_test_ce"] = res.final_ce
        r["delta_vs_single"] = r["child_test_ce"] - res.final_ce
        r["delta_vs_single_sigma"] = r["delta_vs_single"] / sigma_test
        r["k_pop_pass"] = bool(r["delta_vs_best"] <= -3 * sigma_test and r["delta_vs_single"] <= -3 * sigma_test)
        log(f"  alpha {a}: child - best {r['delta_vs_best']:+.4f} ({r['delta_vs_best_sigma']:+.1f} sigma)  "
            f"child - SINGLE {r['delta_vs_single']:+.4f} ({r['delta_vs_single_sigma']:+.1f} sigma)  "
            f"K-POP {'PASS' if r['k_pop_pass'] else 'FIRES'}")
    rec = {"run": "R7-POP", "lineage": lin.name, "live": args.live, "K": K, "tokens_sib": tokens_sib,
           "single_tokens": single_tokens, "single_test_ce": res.final_ce, "single_fitness_ce": ce_single_fit,
           "sigma_test": sigma_test, "eps_q": eps_q, "per_alpha": per_alpha}
    write_json(out / "pop.json", manifest(rec))
    log(f"wrote {out / 'pop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
