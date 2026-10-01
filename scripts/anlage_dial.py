#!/usr/bin/env python
"""R6-dial: the expression dial at one stage -- birth from the genome against a warm start.

SPEC: 05 §5.4
SPEC: 08 §8.1

    python scripts/anlage_dial.py --genome runs/anlage/birth-tinystories-s0-dg64H4z16/genome.pt \\
        --checkpoint runs/anlage/noise-tinystories-s0/seed0.pt --eps 0 0.5 1 --seeds 0 1

`W_0 = eps W_prev + (1 - eps) D(Gamma, Z)` (5.4), then `tokens` more tokens of
training at the same stage under CRN. `eps = 1` is the warm-start control
(B-WARM at this stage); `eps = 0` is genome birth. The curve over `eps` is the
first measurement of K-LOOP's question: does compressing through the genome buy
anything over inheriting the weights directly, at equal compute after birth?
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import torch  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import GenomeShape  # noqa: E402
from anlage.genome import DecodeContext, Genome  # noqa: E402
from anlage.latents import LatentBank  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, load_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from anlage.regress import expression  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tokens", type=int, default=12_500_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"dial-{lin.name}-s{args.stage}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    # a *different* pool seed from R1: the continuation sees windows the
    # checkpoint was not trained on, as generation g+1 would
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    blob = torch.load(args.genome, map_location="cpu")
    shape = GenomeShape(**blob["shape"])
    prev = load_phenotype(args.checkpoint, device=device)
    W_prev = {k: v.detach() for k, v in prev.weights().items()}
    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    bank = LatentBank(shape, cfg, lin.vocab).to(device)
    bank.load_state_dict(blob["bank"])
    G = Genome(shape).to(device)
    G.load_state_dict(blob["genome"])
    ctx = DecodeContext(cfg, tables, bank, lin.vocab, tied=prev.tied)
    with torch.no_grad():
        D = G.decode_all(ctx)
    ce_prev, _ = evaluate(prev, test)
    log(f"R6-dial  {lin.name} {cfg.summary()}  checkpoint CE {ce_prev:.4f}  tokens {args.tokens:,}  "
        f"eps {args.eps}  seeds {args.seeds}  pool {pool.digest()}")
    results = {}
    for eps in args.eps:
        for seed in args.seeds:
            W0 = expression(W_prev, D, eps)
            m = Phenotype(cfg, lin.vocab, tied=prev.tied, weights=W0).to(device)
            ce0, _ = evaluate(m, test)
            tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T,
                             lr_muon=args.lr_muon, lr_emb=args.lr_emb, lr_provenance="R6-dial")
            r = train(m, pool, test, tc, order_seed=seed, compile_mode=None, log=None,
                      eval_every_tokens=max(args.tokens // 5, 1))
            key = f"eps{eps}/seed{seed}"
            results[key] = {"eps": eps, "seed": seed, "start_ce": ce0, "final_ce": r.final_ce,
                            "curve": r.curve, "final_strata": r.final_strata}
            log(f"  {key}: start {ce0:.4f} -> final {r.final_ce:.4f}")
            del m
            torch.cuda.empty_cache()
    summary = {str(e): [results[f"eps{e}/seed{s}"]["final_ce"] for s in args.seeds] for e in args.eps}
    write_json(out / "dial.json", manifest({
        "run": "R6-dial", "lineage": lin.name, "stage": args.stage, "genome": args.genome,
        "checkpoint": args.checkpoint, "checkpoint_ce": ce_prev, "tokens": args.tokens,
        "eps": args.eps, "seeds": args.seeds, "summary": summary, "results": results,
        "pool_digest": pool.digest()}))
    for e, v in summary.items():
        log(f"  eps {e}: {['%.4f' % x for x in v]}  mean {sum(v) / len(v):.4f}")
    log(f"wrote {out / 'dial.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
