#!/usr/bin/env python
"""The genomic bottleneck of Shuvaev et al. (PNAS 2024), run as printed on the 13M language model.

SPEC: docs/bottleneck/00-replication.md

    python scripts/bottleneck_loop.py --arm GN30 --generations 200 --T-gen 1000000 --seed 0
    python scripts/bottleneck_loop.py --arm free --generations 200 --T-gen 1000000 --seed 0

Arms: `GN30` and `GN50` are the paper's g-network structures; `mild` picks `H` for a ratio near
`--target-ratio` (default 5, the paper's HalfCheetah); `free` is the tabula-rasa control, the
same training with no genome. Every arm at the same seed sees the same windows in the same
order (CRN). One record per generation; the at-birth CE of the decode is the paper's metric,
the final CE at matched training tokens is this repository's.

Variants (§11, pre-registered after 11 §P-D.48): `--fisher` refits in the Fisher metric of the
fitness probe (11.1); `--select` keeps a refit only if its decode is no worse on the fitness set
(11.2). Run names carry `-F`, `-S` or `-FS`; without the flags the loop is Algorithm 1 exactly.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import torch  # noqa: E402

from anlage.config import ROLE_BY_NAME  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, save_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from bottleneck.genome import BottleneckGenome  # noqa: E402
from bottleneck.loop import Intermittent, shuffle_rows  # noqa: E402

ARMS = {
    # H, H2 -- Tables 1-2 of the paper; `mild` is solved for a target ratio at build time (§4)
    "GN30": (30, 10),
    "GN50": (50, 10),
    "mild": (None, None),
    "free": (None, None),
}


def solve_H_for_ratio(model: Phenotype, rank: torch.Tensor, target: float, *, seed: int,
                      device: str, keep=()) -> tuple[int, int]:
    """The smallest `H` (with `H2 = max(10, H // 8)`) whose genome reaches `ratio <= target`.
    SPEC: §4 -- the ratio is computed, not assumed (kept roles count, §11.4)."""
    lo, hi = 8, 8
    while BottleneckGenome(model, rank, H=hi, H2=max(10, hi // 8), seed=seed, device="cpu", keep=keep).compression_ratio(model) > target:
        lo, hi = hi, hi * 2
        if hi > 1 << 16:
            raise RuntimeError("no H reaches the target ratio")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = BottleneckGenome(model, rank, H=mid, H2=max(10, mid // 8), seed=seed, device="cpu", keep=keep).compression_ratio(model)
        if r > target:
            lo = mid
        else:
            hi = mid
    return hi, max(10, hi // 8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--generations", type=int, default=200, help="the paper uses 500 (§3)")
    ap.add_argument("--T-gen", type=int, default=1_000_000, help="training tokens per generation (§7)")
    ap.add_argument("--lam", type=float, default=20.0, help="Eq. 3's lambda (§3)")
    ap.add_argument("--n-refit", type=int, default=50, help="Adam steps per refit (§3 b)")
    ap.add_argument("--fraction", type=float, default=1 / 6, help="entries sampled per refit step; the paper fits on 1/6 (p. 10)")
    ap.add_argument("--cap", type=int, default=1 << 16, help="entries per matrix per refit step, at most")
    ap.add_argument("--g-lr", type=float, default=2e-3)
    # §11 variants (off = the paper's Algorithm 1 exactly)
    ap.add_argument("--fisher", action="store_true", help="§11.1: Fisher-weighted regression on the fitness probe")
    ap.add_argument("--fisher-alpha", type=float, default=0.1, help="floor on the sampling weight, (11.1); 1 = uniform")
    ap.add_argument("--n-fisher", type=int, default=32, help="fitness windows in the Fisher probe (one training batch)")
    ap.add_argument("--fisher-micro", type=int, default=4, help="windows per squared gradient")
    ap.add_argument("--select", action="store_true", help="§11.2: accept a refit only if its decode is no worse on the fitness set")
    ap.add_argument("--select-from", type=int, default=30, help="first generation with selection; eps(30) = 0.22")
    ap.add_argument("--keep", nargs="*", default=[], help="§11.4: roles left uncompressed and counted in the ratio, e.g. EMB")
    ap.add_argument("--activation", default="relu", choices=("relu", "leaky"),
                    help="§11.5: the g-network's hidden activation; `leaky` (slope 0.01) cannot die")
    ap.add_argument("--emb-rank", type=int, default=0,
                    help="§11.6: carry the vocabulary table as its rank-r truncated SVD, r(V + d) parameters counted; 0 = off")
    ap.add_argument("--target-ratio", type=float, default=5.0, help="the `mild` arm's ratio (§4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=16,
                    help="gradient accumulation is exact (anlage P14); 16 keeps the 13M run's peak near "
                         "half of R1's 8.35 GB on a card shared with a game (11 §P-D.22)")
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--p-dec", type=float, default=0.999,
                    help="where the learning rate starts decaying, as a fraction of the run. 0.999 = no decay: "
                         "the paper's constant per-generation rate (§3.1, corrected). The R1 recipe's 0.8 "
                         "collapsed the first GN30 run's last thirty-five generations (11 §P-D.46)")
    ap.add_argument("--pool-windows", type=int, default=400_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--shuffle-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", default="none")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    for name in args.keep:
        if name not in ROLE_BY_NAME:
            raise SystemExit(f"--keep: unknown role {name!r}; roles are {sorted(ROLE_BY_NAME)}")
    keep = [ROLE_BY_NAME[name] for name in args.keep]
    if args.emb_rank < 0 or (args.emb_rank and "EMB" in args.keep):
        raise SystemExit("--emb-rank is >= 0 and excludes --keep EMB")
    rank_roles = {ROLE_BY_NAME["EMB"]: args.emb_rank} if args.emb_rank else {}
    letters = (("F" if args.fisher else "") + ("S" if args.select else "")
               + ("E" if args.keep == ["EMB"] else ("K" if args.keep else ""))
               + ("L" if args.activation == "leaky" else "")
               + (f"R{args.emb_rank}" if args.emb_rank else ""))
    variant = f"-{letters}" if letters else ""
    out = pathlib.Path(args.out or REPO / "runs" / "bottleneck" /
                       f"{args.arm}{variant}-{lin.name}-s{args.stage}-seed{args.seed}{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=args.device)
    # `generations + 1` bursts: the hook fires at the START of the step that reaches a boundary,
    # and `train` stops the moment `tokens_done >= tokens`, so a budget of exactly
    # `generations · T_gen` ends on the last boundary before its regeneration can fire (found by
    # the smoke run: 3 generations requested, 2 fired). One extra burst lets every one of the
    # `generations` regenerations happen, and the phenotype then trains once more from the last
    # decode -- which is also Algorithm 1's shape, where each generation ends with the combine
    # step. The tabula-rasa arm gets the same `generations + 1` bursts, so tokens stay matched.
    tokens_total = (args.generations + 1) * args.T_gen
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed, device=args.device)
    tc = TrainConfig(tokens=tokens_total, batch_final=args.batch, T=lin.T,
                     lr_muon=args.lr_muon, lr_emb=args.lr_emb, micro_batch=args.micro_batch, p_dec=args.p_dec,
                     lr_provenance="R1 NOISE grid value (docs/anlage/RESULTS.md R1); p_dec per 00-replication §3.1")
    model = Phenotype(cfg, lin.vocab, seed=args.seed).to(args.device)
    rank = torch.as_tensor(corpus.rank)

    genome, hook = None, None
    if args.arm != "free":
        H, H2 = ARMS[args.arm]
        if H is None:
            H, H2 = solve_H_for_ratio(model, rank, args.target_ratio, seed=args.seed, device=args.device, keep=keep)
        genome = BottleneckGenome(model, rank, H=H, H2=H2, seed=args.seed, device=args.device, keep=keep,
                                  activation=args.activation, rank_roles=rank_roles)
        ratio = genome.compression_ratio(model)
        hook = Intermittent(genome, model, T_gen=args.T_gen, lam=args.lam, n_refit=args.n_refit,
                            fraction=args.fraction, cap=args.cap, lr=args.g_lr,
                            evaluate=lambda m: evaluate(m, test)[0], shuffle_every=args.shuffle_every,
                            seed=args.seed, log=log, max_generations=args.generations, out_dir=out,
                            # §11.1: the probe is a slice of the FITNESS set; the test set decides nothing
                            fisher_windows=(fit.windows[:args.n_fisher] if args.fisher else None),
                            fisher_alpha=args.fisher_alpha, fisher_micro=args.fisher_micro,
                            # §11.2: selection scores the decode on the fitness set
                            evaluate_fit=((lambda m: evaluate(m, fit)[0]) if args.select else None),
                            select_from=args.select_from)
        log(f"BOTTLENECK {args.arm}{variant}  {lin.name} {cfg.summary()}  H {H} H2 {H2}  genome {genome.n_params():,} "
            f"params  ratio {ratio:.1f}x  generations {args.generations} x {args.T_gen:,} tokens  "
            f"lambda {args.lam}  refit {args.n_refit} steps on {args.fraction:.3f} of entries (cap {args.cap:,})"
            + (f"  fisher alpha {args.fisher_alpha} on {args.n_fisher} fitness windows" if args.fisher else "")
            + (f"  select from {args.select_from}" if args.select else "")
            + (f"  keep {args.keep} uncompressed ({genome.summary()['n_params_uncompressed']:,} entries counted)" if keep else "")
            + (f"  activation {args.activation}" if args.activation != "relu" else "")
            + (f"  EMB as rank {args.emb_rank} ({genome.n_params_rank():,} parameters counted)" if args.emb_rank else ""))
        # generation 0: the random genome's decode, the paper's "random genome" reference (§5)
        W0 = {k: p.detach() for k, p in model.weights().items()}
        D0 = genome.decode_all(W0)
        ce_random_genome = hook._evaluate_weights(D0)
        log(f"  random genome at birth: {ce_random_genome:.4f}   (L_null = {torch.log(torch.tensor(float(lin.vocab))).item():.4f})")
    else:
        ratio, ce_random_genome = None, None
        log(f"BOTTLENECK free  {lin.name} {cfg.summary()}  tokens {tokens_total:,}  (the tabula-rasa control)")

    t0 = time.time()
    res = train(model, pool, test, tc, order_seed=args.seed, compile_mode=(None if args.compile == "none" else args.compile),
                hooks=({"before_step": hook.before_step} if hook else None), log=log,
                eval_every_tokens=max(1, 10 * args.T_gen), device=args.device, train_log_every=200)
    seconds = time.time() - t0

    ce_fit, _ = evaluate(model, fit)
    final = {"test_ce": res.final_ce, "fitness_ce": ce_fit, "tokens": res.as_dict().get("tokens", tokens_total)}
    if genome is not None:
        W = {k: p.detach() for k, p in model.weights().items()}
        D = genome.decode_all(W)
        final["birth_ce_of_final_decode"] = hook._evaluate_weights(D)
        final["shuffle_ce_of_final_decode"] = hook._evaluate_weights(
            shuffle_rows(D, torch.Generator(device="cpu").manual_seed(args.seed + 99)))
        final["residual_final"] = {genome._name(k): v for k, v in genome.residuals(W, D).items()}
        torch.save({"state": genome.state_dict(), "summary": genome.summary(), "keys": genome.keys,
                    "shapes": genome.shapes}, out / "genome.pt")
    save_phenotype(model, out / "phenotype.pt")

    report = manifest({
        "run": "BOTTLENECK", "arm": args.arm, "variant": variant, "lineage": lin.name, "stage": args.stage,
        "config": cfg.summary(), "args": vars(args), "tokens_total": tokens_total,
        "train_config": tc.__dict__, "pool_digest": pool.digest(), "corpus": corpus.manifest(),
        "eval_test_digest": test.digest, "eval_fit_digest": fit.digest,
        "genome": (genome.summary() | {"compression_ratio": ratio}) if genome else None,
        "ce_random_genome": ce_random_genome,
        "hook": hook.as_dict() if hook else None,
        "train": res.as_dict(), "final": final, "seconds": seconds, "complete": True,
    })
    write_json(out / "loop.json", report)
    log(f"final test {res.final_ce:.4f}  fit {ce_fit:.4f}"
        + (f"  birth(final decode) {final['birth_ce_of_final_decode']:.4f}  "
           f"shuffle {final['shuffle_ce_of_final_decode']:.4f}" if genome else "")
        + f"  {seconds / 3600:.2f} h  wrote {out / 'loop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
