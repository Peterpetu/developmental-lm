#!/usr/bin/env python
"""R6 POP: a sibling population with lexicase over strata and merging, against SINGLE.

SPEC: 07 §7.2
SPEC: 07 §7.4
SPEC: 07 §7.5
SPEC: 08 §8.1

    python scripts/anlage_pop.py --K 4 --tokens-sib 6250000 --seeds 0 1

Population arm: `K` siblings from one birth (`eps = 1` warm start from the
given checkpoint, or the free initialisation when no checkpoint is given),
each `tokens_sib` tokens with its own data order; ε-lexicase over the fitness
strata with `eps_c = sigma_e(c)` from R1; merge (7.1) with fitted coefficients;
acceptance (7.2). SINGLE: one lifetime of `K * tokens_sib` tokens from the same
birth. Both are scored on the test split; the merge is additionally compared to
the best sibling (K-POP's second clause).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import EvalSet, N_STRATA  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, load_phenotype, save_phenotype  # noqa: E402
from anlage.population import merge, select_parents  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from gnomon.utils.rng import derive_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--checkpoint", default=None, help="birth weights (eps = 1); free init if absent")
    ap.add_argument("--noise", default="runs/anlage/noise-tinystories-s0/noise.json")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--tokens-sib", type=int, default=6_250_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--merge-steps", type=int, default=200)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" /
                       f"pop-{lin.name}-s{args.stage}-K{args.K}{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    # coefficient-fit windows: disjoint from the scoring windows (S8)
    fitw = EvalSet.build(corpus.fitness, corpus.rank, n_windows=64, T=lin.T, split="fitness",
                         device=device, offset_windows=fit.n_windows).windows
    noise = json.loads(pathlib.Path(args.noise).read_text())
    sigma_e = float(noise["sigma_e"])
    eps_c = [float(x) if x is not None else float("nan") for x in noise["sigma_e_strata_fit"]]
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    log(f"R6 POP  {lin.name} {cfg.summary()}  K {args.K}  tokens/sib {args.tokens_sib:,}  "
        f"sigma_e {sigma_e:.4f}  eps_c {['%.3f' % e for e in eps_c]}")

    def birth(seed):
        if args.checkpoint:
            return load_phenotype(args.checkpoint, device=device)
        return Phenotype(cfg, lin.vocab, seed=seed).to(device)

    all_res = {}
    for seed in args.seeds:
        rec = {"seed": seed}
        tc_sib = TrainConfig(tokens=args.tokens_sib, batch_final=args.batch, T=lin.T,
                             lr_muon=args.lr_muon, lr_emb=args.lr_emb, lr_provenance="R6")
        W0_model = birth(seed)
        W0 = {k: v.detach().clone() for k, v in W0_model.weights().items()}
        # --- siblings ---------------------------------------------------
        sibs, scores, test_ces = [], [], []
        for k in range(args.K):
            m = birth(seed)
            order = derive_seed(seed, "member") ^ k
            r = train(m, pool, test, tc_sib, order_seed=order, compile_mode=None, log=None)
            ce_f, per_f = evaluate(m, fit)
            sibs.append(m)
            scores.append(per_f)
            test_ces.append(r.final_ce)
            log(f"  seed {seed} sibling {k}: test {r.final_ce:.4f}  fit {ce_f:.4f}")
        rng = np.random.default_rng(derive_seed(seed, "lexicase"))
        parents_idx = select_parents(scores, eps_c, max(2, args.K // 2), rng)
        distinct = sorted(set(parents_idx))
        log(f"  lexicase parents: {parents_idx} -> distinct {distinct}")
        child, mrep = merge([sibs[i] for i in distinct], W0, fitw, fit, sigma_e,
                            steps=args.merge_steps, log=log)
        ce_child, per_child = evaluate(child, test)
        best_sib_test = min(test_ces)
        log(f"  merge: fit {mrep.ce_fit_merge:.4f} vs parents {['%.4f' % c for c in mrep.ce_fit_parents]}"
            f"  accepted {mrep.accepted} ({mrep.chosen});  test {ce_child:.4f} vs best sibling {best_sib_test:.4f}")
        rec["siblings_test"] = test_ces
        rec["siblings_fit_strata"] = scores
        rec["parents"] = parents_idx
        rec["merge"] = mrep.as_dict()
        rec["merge_test_ce"] = ce_child
        rec["merge_test_strata"] = per_child
        save_phenotype(child, out / f"merged_seed{seed}.pt")
        # --- SINGLE at matched compute -------------------------------------
        tc_single = TrainConfig(tokens=args.tokens_sib * args.K, batch_final=args.batch, T=lin.T,
                                lr_muon=args.lr_muon, lr_emb=args.lr_emb, lr_provenance="R6")
        single = birth(seed)
        r1 = train(single, pool, test, tc_single, order_seed=derive_seed(seed, "member"),
                   compile_mode=None, log=None)
        rec["single_test_ce"] = r1.final_ce
        rec["single_test_strata"] = r1.final_strata
        save_phenotype(single, out / f"single_seed{seed}.pt")
        log(f"  SINGLE ({args.tokens_sib * args.K:,} tokens): test {r1.final_ce:.4f}")
        all_res[seed] = rec
        write_json(out / f"seed{seed}.json", rec)
        del sibs, child, single
        torch.cuda.empty_cache()

    merged = [all_res[s]["merge_test_ce"] for s in args.seeds]
    best = [min(all_res[s]["siblings_test"]) for s in args.seeds]
    single = [all_res[s]["single_test_ce"] for s in args.seeds]
    write_json(out / "pop.json", manifest({
        "run": "R6-POP", "lineage": lin.name, "stage": args.stage, "K": args.K,
        "tokens_sib": args.tokens_sib, "seeds": args.seeds, "sigma_e": sigma_e,
        "merge_test": merged, "best_sibling_test": best, "single_test": single,
        "paired_merge_minus_best": [m - b for m, b in zip(merged, best)],
        "paired_merge_minus_single": [m - s for m, s in zip(merged, single)],
        "results": all_res, "pool_digest": pool.digest(), "corpus": corpus.manifest()}))
    log(f"merge {merged}  best sibling {best}  single {single}")
    log(f"wrote {out / 'pop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
