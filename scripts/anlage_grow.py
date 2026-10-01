#!/usr/bin/env python
"""R5 GROW: genome-sampled growth against its controls at matched compute.

SPEC: 06 §6.6
SPEC: 08 §8.1

    python scripts/anlage_grow.py --checkpoint-dir runs/anlage/noise-tinystories-s0 \\
        --seeds 0 1 2 3 --tokens-next 25000000 --arms genome copy random flat

Every arm starts from the same trained stage-`s` checkpoint (one per seed,
from R1) and continues at stage `s+1` for `tokens_next` tokens under the same
window pool and order seed (CRN). `flat` is B-FLAT: stage `s+1` from scratch
for the FLOP-matched token count `tokens_next + tokens_prev * N_s / N_{s+1}`.
The genome arm regresses the genome on the checkpoint first (§06.4), then
grows; `mean` and `zero` are the latent ablations of §06.6.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import torch  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import GenomeShape  # noqa: E402
from anlage.genome import DecodeContext, Genome  # noqa: E402
from anlage.grow import unit_prior, Ramp, grow, layer_placement  # noqa: E402
from anlage.latents import LatentBank  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, load_phenotype, save_phenotype  # noqa: E402
from anlage.recipe import TrainConfig, micro_batch_for  # noqa: E402
from anlage.regress import RegressConfig, regress  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["prior", "copy", "random", "flat"])
    ap.add_argument("--tokens-prev", type=int, default=25_000_000,
                    help="tokens the checkpoint was trained on (for FLOP matching)")
    ap.add_argument("--tokens-next", type=int, default=25_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--ramp-frac", type=float, default=0.02)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg0, cfg1 = lin.stage(args.stage), lin.stage(args.stage + 1)
    shape = GenomeShape()
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" /
                       f"grow-{lin.name}-s{args.stage}to{args.stage + 1}{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    N0 = cfg0.n_params_nonemb(lin.vocab)
    N1 = cfg1.n_params_nonemb(lin.vocab)
    tokens_flat = args.tokens_next + int(round(args.tokens_prev * N0 / N1))
    log(f"R5 GROW  {lin.name} {cfg0.summary()} -> {cfg1.summary()}  N0 {N0:,} N1 {N1:,}  "
        f"tokens_next {args.tokens_next:,}  flat tokens {tokens_flat:,}  arms {args.arms}  "
        f"seeds {args.seeds}  pool {pool.digest()}")

    results: dict[str, dict] = {}
    for seed in args.seeds:
        ck = pathlib.Path(args.checkpoint_dir) / f"seed{seed}.pt"
        for arm in args.arms:
            key = f"{arm}/seed{seed}"
            path = out / f"{arm}_seed{seed}.json"
            if path.exists():
                results[key] = json.loads(path.read_text())
                log(f"  {key}: cached {results[key]['final_ce']:.4f}")
                continue
            log(f"== {key} ==")
            events = []
            if arm == "flat":
                model = Phenotype(cfg1, lin.vocab, seed=seed).to(device)
                tokens = tokens_flat
                ramp = None
                pre_ce = None
            else:
                base = load_phenotype(ck, device=device)
                pre_ce, _ = evaluate(base, test)
                decoded = None
                extra = {}
                if arm in ("genome", "mean", "zero"):
                    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg0.L,
                                        device=device)
                    bank = LatentBank(shape, cfg0, lin.vocab, seed=1).to(device)
                    ctx = DecodeContext(cfg0, tables, bank, lin.vocab, tied=base.tied)
                    G = Genome(shape, seed=1).to(device)
                    rep = regress(G, ctx, {k: v.detach() for k, v in base.weights().items()},
                                  RegressConfig(n_fit=args.n_fit), log=log)
                    extra["regress"] = rep.as_dict()
                    lm, ops1 = layer_placement(cfg0, cfg1)
                    cfg1g = dataclasses.replace(cfg1, ops_override=ops1)
                    mode = {"genome": "sample", "mean": "mean", "zero": "zero"}[arm]
                    bank1 = bank.grown(cfg1g, lm, torch.Generator().manual_seed(seed), mode=mode)
                    tables1 = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg1g.L,
                                         device=device)
                    ctx1 = DecodeContext(cfg1g, tables1, bank1, lin.vocab, tied=base.tied)
                    with torch.no_grad():
                        decoded = G.decode_all(ctx1)
                    model, plan = grow(base, cfg1, mode="genome", decoded=decoded)
                    del G, bank, bank1, tables, tables1, ctx, ctx1, decoded
                else:
                    model, plan = grow(base, cfg1, mode=arm,
                                       generator=torch.Generator().manual_seed(seed),
                                       Lambda=unit_prior(base) if arm == "prior" else None)
                    if arm == "prior":
                        extra["Lambda"] = unit_prior(base)
                del base
                torch.cuda.empty_cache()
                post_ce_gate0 = None
                tokens = args.tokens_next
                n_steps_est = tokens // (args.batch * lin.T)
                ramp = Ramp(model, plan, n_ramp=max(1, int(args.ramp_frac * n_steps_est)))
                post_ce, _ = evaluate(model, test)
                events.append({"event": "grow", "pre_ce": pre_ce, "post_ce_rho0": post_ce,
                               "rho_0": ramp.rho_0, "layer_map": plan.layer_map,
                               "ops": list(plan.cfg_new.ops())})
                log(f"  grown: pre {pre_ce:.4f} -> at rho_0 {post_ce:.4f}  "
                    f"ops {''.join(o[0] for o in plan.cfg_new.ops())}")
            tc = TrainConfig(tokens=tokens, batch_final=args.batch, T=lin.T,
                             lr_muon=args.lr_muon, lr_emb=args.lr_emb,
                             lr_provenance="R5 GROW: R1/R2 value", micro_batch=micro_batch_for(model.n_params(), args.batch))
            hooks = {}
            if ramp is not None:
                def before_step(step, tokens_done, _r=ramp, _m=model, _ev=events):
                    if not _r.done:
                        _r.step()
                        if _r.done:
                            # G4: the held-out CE the moment the ramp completes
                            _ev[-1]["post_ramp_ce"] = evaluate(_m, test)[0]
                            _ev[-1]["ramp_steps"] = step + 1
                hooks["before_step"] = before_step
            res = train(model, pool, test, tc, order_seed=seed, compile_mode=None,
                        eval_every_tokens=max(tokens // 8, 1), hooks=hooks, log=log)
            ce_fit, per_fit = evaluate(model, fit)
            rec = {"arm": arm, "seed": seed, "tokens": tokens, "pre_ce": pre_ce,
                   "final_ce": res.final_ce, "fitness_ce": ce_fit, "curve": res.curve,
                   "final_strata": res.final_strata, "events": events,
                   "tokens_per_s": res.tokens_per_s, "peak_mem_gb": res.peak_mem_gb,
                   "n_params": model.n_params(), "seconds": res.seconds}
            if arm != "flat":
                rec["ops"] = list(model.cfg.ops())
            results[key] = rec
            write_json(path, rec)
            save_phenotype(model, out / f"{arm}_seed{seed}.pt")
            log(f"  {key}: final {res.final_ce:.4f}  fit {ce_fit:.4f}  "
                f"{res.tokens_per_s:,.0f} tok/s  {res.seconds:.0f}s")
            del model
            torch.cuda.empty_cache()

    # paired summary
    summary = {}
    for arm in args.arms:
        vals = [results[f"{arm}/seed{s}"]["final_ce"] for s in args.seeds
                if f"{arm}/seed{s}" in results]
        summary[arm] = {"final_ce": vals, "mean": sum(vals) / len(vals) if vals else None}
    write_json(out / "grow.json", manifest({
        "run": "R5-GROW", "lineage": lin.name, "stage": args.stage,
        "cfg0": cfg0.summary(), "cfg1": cfg1.summary(), "tokens_prev": args.tokens_prev,
        "tokens_next": args.tokens_next, "tokens_flat": tokens_flat, "N0": N0, "N1": N1,
        "seeds": args.seeds, "arms": args.arms, "pool_digest": pool.digest(),
        "summary": summary, "results": results, "corpus": corpus.manifest()}))
    for arm, s in summary.items():
        log(f"  {arm:8s} mean {s['mean']:.4f}  {s['final_ce']}")
    log(f"wrote {out / 'grow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
