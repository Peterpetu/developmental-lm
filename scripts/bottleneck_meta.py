#!/usr/bin/env python
"""The loop over a task distribution: the routine aimed at learning how to learn.

SPEC: docs/bottleneck/00-replication.md §10

    python scripts/bottleneck_meta.py --arm meta-GN30
    python scripts/bottleneck_meta.py --arm meta-mild
    python scripts/bottleneck_meta.py --arm meta-free

Each generation trains the phenotype on one freshly drawn task; the genome arms refit and
regenerate at every boundary (§3), `meta-free` only switches task. Every arm at a seed sees the
same task sequence (CRN over tasks). The record is `loop.json` in the shape `bottleneck_loop.py`
writes, so `scripts/bottleneck_transfer.py` reads a meta run as a source unchanged, plus the
task sequence and the distribution.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402
from bottleneck_loop import solve_H_for_ratio  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.config import ROLE_BY_NAME  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, save_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from bottleneck.genome import BottleneckGenome  # noqa: E402
from bottleneck.loop import shuffle_rows  # noqa: E402
from bottleneck.meta import AdvanceOnly, MetaIntermittent, MetaPool, TaskDistribution  # noqa: E402
from bottleneck.transfer import task_eval_set  # noqa: E402

ARMS = {"meta-GN30": (30, 10), "meta-mild": (None, None), "meta-free": (None, None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--T-gen", type=int, default=1_000_000, help="tokens per task (§10: matched to the single-task loop)")
    ap.add_argument("--windows-per-task", type=int, default=2048)
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--n-refit", type=int, default=50)
    ap.add_argument("--fraction", type=float, default=1 / 6)
    ap.add_argument("--cap", type=int, default=1 << 16)
    ap.add_argument("--g-lr", type=float, default=2e-3)
    # §11.1 (selection, §11.2, is not applicable here: decodes are not comparable across tasks)
    ap.add_argument("--fisher", action="store_true", help="§11.1: Fisher-weighted regression on the current task's probe")
    ap.add_argument("--fisher-alpha", type=float, default=0.1)
    ap.add_argument("--n-fisher", type=int, default=32)
    ap.add_argument("--fisher-micro", type=int, default=4)
    ap.add_argument("--keep", nargs="*", default=[], help="§11.4: roles left uncompressed and counted, e.g. EMB")
    ap.add_argument("--activation", default="relu", choices=("relu", "leaky"), help="§11.5")
    ap.add_argument("--emb-rank", type=int, default=0, help="§11.6: the table as its rank-r SVD; 0 = off")
    ap.add_argument("--target-ratio", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--p-dec", type=float, default=0.999, help="no decay: the paper's constant rate (§3.1, corrected)")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--shuffle-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lin = get_lineage("tinystories")
    cfg = lin.stage(0)
    for name in args.keep:
        if name not in ROLE_BY_NAME:
            raise SystemExit(f"--keep: unknown role {name!r}; roles are {sorted(ROLE_BY_NAME)}")
    keep = [ROLE_BY_NAME[name] for name in args.keep]
    if args.emb_rank < 0 or (args.emb_rank and "EMB" in args.keep):
        raise SystemExit("--emb-rank is >= 0 and excludes --keep EMB")
    rank_roles = {ROLE_BY_NAME["EMB"]: args.emb_rank} if args.emb_rank else {}
    letters = (("F" if args.fisher else "") + ("E" if args.keep == ["EMB"] else ("K" if args.keep else ""))
               + ("L" if args.activation == "leaky" else "") + (f"R{args.emb_rank}" if args.emb_rank else ""))
    variant = f"-{letters}" if letters else ""
    out = pathlib.Path(args.out or REPO / "runs" / "bottleneck" / f"{args.arm}{variant}-tinystories-s0-seed{args.seed}{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)                       # the token codes still read the corpus's ranks (§2)
    rank = torch.as_tensor(corpus.rank)
    dist = TaskDistribution()
    tokens_total = (args.generations + 1) * args.T_gen
    pool = MetaPool(dist, n_tasks=args.generations + 1, n_windows=args.windows_per_task, T=lin.T,
                    seed=args.seed, device=args.device)
    ev_train = task_eval_set(args.n_eval, T=lin.T, seed=5_000 + args.seed, device=args.device, sampler=dist.sample)
    ev_held = task_eval_set(args.n_eval // 2, T=lin.T, seed=6_000 + args.seed, device=args.device, sampler=dist.holdout_sample)
    tc = TrainConfig(tokens=tokens_total, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon,
                     lr_emb=args.lr_emb, micro_batch=args.micro_batch, p_dec=args.p_dec,
                     lr_provenance="R1 NOISE grid value (docs/anlage/RESULTS.md R1); p_dec per 00-replication §3.1")
    model = Phenotype(cfg, lin.vocab, seed=args.seed).to(args.device)
    ev_fn = lambda m: evaluate(m, ev_train)[0]  # noqa: E731

    genome, hook, ratio, ce_random = None, None, None, None
    if args.arm != "meta-free":
        H, H2 = ARMS[args.arm]
        if H is None:
            H, H2 = solve_H_for_ratio(model, rank, args.target_ratio, seed=args.seed, device=args.device, keep=keep)
        genome = BottleneckGenome(model, rank, H=H, H2=H2, seed=args.seed, device=args.device, keep=keep,
                                  activation=args.activation, rank_roles=rank_roles)
        ratio = genome.compression_ratio(model)
        hook = MetaIntermittent(genome, model, pool, T_gen=args.T_gen, lam=args.lam, n_refit=args.n_refit,
                                fraction=args.fraction, cap=args.cap, lr=args.g_lr, evaluate=ev_fn,
                                shuffle_every=args.shuffle_every, seed=args.seed, log=log,
                                max_generations=args.generations, out_dir=out,
                                # §11.1 on the task being learned: the probe follows the pool (M-I4)
                                fisher_windows=((lambda: pool.probe(args.n_fisher)) if args.fisher else None),
                                fisher_alpha=args.fisher_alpha, fisher_micro=args.fisher_micro)
        ce_random = hook._evaluate_weights(genome.decode_all({k: p.detach() for k, p in model.weights().items()}))
        log(f"META {args.arm}{variant}  H {H} H2 {H2}  genome {genome.n_params():,}  ratio {ratio:.1f}x  "
            f"{args.generations} tasks x {args.T_gen:,} tokens  pool {pool.digest()}  dist {dist.describe()}")
        log(f"  random genome at birth (train dist): {ce_random:.4f}")
    else:
        hook = AdvanceOnly(pool, T_gen=args.T_gen, max_generations=args.generations, evaluate=ev_fn, model=model, log=log)
        log(f"META meta-free  {args.generations} tasks x {args.T_gen:,} tokens  pool {pool.digest()}  (sequential multi-task, no genome)")

    t0 = time.time()
    res = train(model, pool, ev_train, tc, order_seed=args.seed, compile_mode=None,
                hooks={"before_step": hook.before_step}, log=log, eval_every_tokens=max(1, 10 * args.T_gen),
                device=args.device, train_log_every=200)
    seconds = time.time() - t0

    final = {"test_ce": res.final_ce, "heldout_ce": evaluate(model, ev_held)[0], "tokens": res.as_dict().get("tokens", tokens_total)}
    if genome is not None:
        W = {k: p.detach() for k, p in model.weights().items()}
        D = genome.decode_all(W)
        final["birth_ce_of_final_decode"] = hook._evaluate_weights(D)
        final["birth_heldout_ce_of_final_decode"] = _eval_with(model, D, ev_held)
        final["shuffle_ce_of_final_decode"] = hook._evaluate_weights(
            shuffle_rows(D, torch.Generator(device="cpu").manual_seed(args.seed + 99)))
        final["residual_final"] = {genome._name(k): v for k, v in genome.residuals(W, D).items()}
        torch.save({"state": genome.state_dict(), "summary": genome.summary(), "keys": genome.keys,
                    "shapes": genome.shapes}, out / "genome.pt")
    save_phenotype(model, out / "phenotype.pt")

    report = manifest({
        "run": "BOTTLENECK-META", "arm": args.arm, "variant": variant, "lineage": lin.name, "stage": 0, "config": cfg.summary(),
        "args": vars(args), "tokens_total": tokens_total, "train_config": tc.__dict__,
        "pool_digest": pool.digest(), "distribution": dist.describe(),
        "task_sequence": [dataclasses.asdict(p) for p in pool.sequence],
        "eval_train_digest": ev_train.digest, "eval_heldout_digest": ev_held.digest,
        "genome": (genome.summary() | {"compression_ratio": ratio}) if genome else None,
        "ce_random_genome": ce_random, "hook": hook.as_dict(), "train": res.as_dict(),
        "final": final, "seconds": seconds, "complete": True,
    })
    write_json(out / "loop.json", report)
    log(f"final train-dist {res.final_ce:.4f}  held-out {final['heldout_ce']:.4f}"
        + (f"  birth(final decode) {final['birth_ce_of_final_decode']:.4f}  held-out birth "
           f"{final['birth_heldout_ce_of_final_decode']:.4f}  shuffle {final['shuffle_ce_of_final_decode']:.4f}" if genome else "")
        + f"  {seconds / 3600:.2f} h  wrote {out / 'loop.json'}")
    return 0


@torch.no_grad()
def _eval_with(model: Phenotype, W_new: dict, ev) -> float:
    live = model.weights()
    saved = {k: p.data.clone() for k, p in live.items()}
    try:
        for k, p in live.items():
            p.data.copy_(W_new[k].to(p.dtype))
        return float(evaluate(model, ev)[0])
    finally:
        for k, p in live.items():
            p.data.copy_(saved[k])


if __name__ == "__main__":
    raise SystemExit(main())
