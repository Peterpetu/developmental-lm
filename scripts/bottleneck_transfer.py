#!/usr/bin/env python
"""Transfer through the bottleneck: the paper's Fig. 3G / 5G protocol, from a finished source run.

SPEC: docs/bottleneck/00-replication.md §9

    python scripts/bottleneck_transfer.py --source runs/bottleneck/mild-tinystories-s0-seed0 --target tasks
    python scripts/bottleneck_transfer.py --source runs/bottleneck/mild-tinystories-s0-seed0 --target tinystories

On the target the arms are `random`, then `copy` and `decode` on each layer set (`all`, `low`,
`high`); `random` is repeated over `--noise-seeds` seeds first so the target's own `sigma_e` is
measured before any contrast is read (G-NOISE). Every arm at a seed shares the pool and the
window order (CRN). The record is the target learning curve of every arm, its final target CE,
and its final CE on the SOURCE test set (forgetting).
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import torch  # noqa: E402

from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from bottleneck.meta import TaskDistribution  # noqa: E402
from bottleneck.transfer import (LAYER_SETS, initialise, load_source, task_eval_set,  # noqa: E402
                                 task_pool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="a finished run directory of bottleneck_loop.py")
    ap.add_argument("--target", required=True, choices=("tasks", "heldout", "tinystories"),
                    help="tasks: §9.1's default-parameter families; heldout: §10.1's unseen family and length; "
                         "tinystories: continuation on fresh corpus windows (the hot-start test)")
    ap.add_argument("--tokens", type=int, default=5_000_000, help="target training tokens per arm (§9.3)")
    ap.add_argument("--which", default="best", choices=("best", "final"),
                    help="the source state: the generation of best at-birth CE (§3.2) or the run's end")
    ap.add_argument("--noise-seeds", type=int, default=3, help="random-init seeds for the target's sigma_e")
    ap.add_argument("--seed", type=int, default=0, help="the paired seed of every non-noise arm")
    ap.add_argument("--layer-sets", default="all,low,high")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=40_000)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lin = get_lineage("tinystories")
    src_dir = pathlib.Path(args.source)
    out = pathlib.Path(args.out or REPO / "runs" / "bottleneck" / f"xfer-{args.target}-from-{src_dir.name}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    rank = torch.as_tensor(corpus.rank)
    source = load_source(src_dir, rank, device=args.device, which=args.which)
    cfg = source.cfg
    src_test, _ = eval_sets(corpus, lin, device=args.device)               # the source's own test set
    if args.target in ("tasks", "heldout"):
        # `heldout` draws every instance from §10.1's held-out region -- the unseen family and the
        # unseen length -- so a meta-trained source is tested on what it never saw; `tasks` is
        # §9.1's default-parameter families. The pool and the evaluation set use different seeds.
        sampler = TaskDistribution().holdout_sample if args.target == "heldout" else None
        pool, shard = task_pool(args.pool_windows, T=lin.T, seed=1_000 + args.seed, pool_seed=20260907,
                                device=args.device, sampler=sampler)
        tgt_test = task_eval_set(args.n_eval, T=lin.T, seed=2_000 + args.seed, device=args.device, sampler=sampler)
        target_manifest = {"kind": args.target, "shard_digest": shard.digest, "pool_digest": pool.digest(),
                           "eval_digest": tgt_test.digest,
                           "distribution": TaskDistribution().describe() if sampler else "default families"}
    else:
        # continuation on fresh corpus windows: a different pool seed from the source loop's
        pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=20260908, device=args.device)
        tgt_test = src_test
        target_manifest = {"kind": "tinystories", "pool_digest": pool.digest(), "eval_digest": tgt_test.digest}
    tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon,
                     lr_emb=args.lr_emb, micro_batch=args.micro_batch,
                     lr_provenance="R1 NOISE grid value (docs/anlage/RESULTS.md R1)")
    log(f"TRANSFER  source {source.arm} ({src_dir.name}, ratio {source.ratio}, state {source.which}"
        f"{'' if source.generation is None else f' at generation {source.generation}'})  target {args.target}  "
        f"{args.tokens:,} tokens/arm  seed {args.seed}  noise seeds {args.noise_seeds}")

    def run_arm(name: str, mode: str, which: str, seed: int) -> dict:
        model = Phenotype(cfg, lin.vocab, seed=seed).to(args.device)
        written = initialise(model, source if mode != "random" else None, mode, which)
        ce0_t = evaluate(model, tgt_test)[0]
        ce0_s = evaluate(model, src_test)[0]
        t0 = time.time()
        res = train(model, pool, tgt_test, tc, order_seed=seed, compile_mode=None, log=None,
                    eval_every_tokens=max(1, args.tokens // 10), device=args.device, train_log_every=0)
        ce_src_final = evaluate(model, src_test)[0]
        rec = {"name": name, "mode": mode, "layers": which, "seed": seed, "n_written": len(written),
               "target_ce_at_birth": ce0_t, "source_ce_at_birth": ce0_s,
               "curve": res.curve, "target_ce_final": res.final_ce, "source_ce_final": ce_src_final,
               "seconds": time.time() - t0}
        log(f"  {name:<14} birth tgt {ce0_t:.4f} src {ce0_s:.4f}  ->  final tgt {res.final_ce:.4f} "
            f"src {ce_src_final:.4f}   curve {[round(c, 3) for _, c in res.curve]}   {rec['seconds']:.0f}s")
        del model
        torch.cuda.empty_cache()
        return rec

    arms = []
    # the target's own noise first (G-NOISE): random initialisations over seeds
    for s in range(args.noise_seeds):
        arms.append(run_arm(f"random-s{s}", "random", "all", s))
    finals = [a["target_ce_final"] for a in arms]
    sigma_e = statistics.stdev(finals) if len(finals) > 1 else float("nan")
    log(f"  target sigma_e {sigma_e:.4f} over {len(finals)} random seeds (3 sigma_e = {3 * sigma_e:.4f})")
    if source.D is None:
        log("  source has no genome (free arm): copy arms only")
    for which in args.layer_sets.split(","):
        arms.append(run_arm(f"copy-{which}", "copy", which, args.seed))
        if source.D is not None:
            arms.append(run_arm(f"decode-{which}", "decode", which, args.seed))

    # the paired contrasts §9.3 reads, at the seed every non-noise arm shares
    base = next(a for a in arms if a["name"] == f"random-s{args.seed}")
    def at(a, frac):
        c = a["curve"]
        tgt = frac * args.tokens
        return min(c, key=lambda p: abs(p[0] - tgt))[1]
    contrasts = {}
    for a in arms:
        if a["mode"] == "random":
            continue
        contrasts[a["name"]] = {f"vs_random_at_{int(f * 100)}": at(a, f) - at(base, f) for f in (0.25, 0.5, 1.0)}
        contrasts[a["name"]]["vs_random_at_birth"] = a["target_ce_at_birth"] - base["target_ce_at_birth"]
    for which in args.layer_sets.split(","):
        c = next((a for a in arms if a["name"] == f"copy-{which}"), None)
        d = next((a for a in arms if a["name"] == f"decode-{which}"), None)
        if c and d:
            contrasts[f"decode_minus_copy-{which}"] = {f"at_{int(f * 100)}": at(d, f) - at(c, f) for f in (0.25, 0.5, 1.0)}

    report = manifest({
        "run": "BOTTLENECK-TRANSFER", "source": {"dir": str(src_dir), "arm": source.arm, "ratio": source.ratio,
                                                "seed": source.seed, "which": source.which,
                                                "generation": source.generation},
        "target": target_manifest | {"tokens": args.tokens}, "args": vars(args), "train_config": tc.__dict__,
        "source_test_digest": src_test.digest, "sigma_e_target": sigma_e, "bar_3sigma": 3 * sigma_e,
        "arms": arms, "contrasts": contrasts, "complete": True,
    })
    write_json(out / "transfer.json", report)
    log(f"contrasts (target CE, positive = worse than the reference):")
    for k, v in contrasts.items():
        log(f"  {k:<26} " + "  ".join(f"{kk} {vv:+.4f}" for kk, vv in v.items()))
    log(f"wrote {out / 'transfer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
