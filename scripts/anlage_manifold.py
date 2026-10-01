#!/usr/bin/env python
"""Manifold probe: train the decoded phenotype `D(Gamma, Z)` directly on the language loss.

SPEC: 08 §8.5
SPEC: 05 §5.6

K-BIRTH fired at the registered dials (at-birth 8.53 nats after one regression of a
free-weight stage-0 adult, R4). Two explanations are separable:

  (i)  capacity: the decodable set `{D(Gamma, Z)}` at these dials holds no good
       stage-0 model;
  (ii) assimilation: it does, but a one-shot regression of an *unconstrained* adult
       does not land there (the PNAS loop needed hundreds of intermittent rounds).

This probe measures (i). Every weight is decoded on every step and the gradient of
the CE is taken through the decoder into `(Gamma, Z)` under the regression's priors
(5.3); the CE reached in a stage-0 token budget is the manifold's capacity at that
budget, a lower bound on what any assimilation scheme can reach. It is a probe, not a
lifetime: §04.6 forbids a decode in the training step of the lifetime (G-NOFREE-INVERSE),
and the probe's throughput is reported to show why.

    python scripts/anlage_manifold.py --tokens 25000000 [--d-g 64 --H 4 --d-z 16 --d-h 256]
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.func import functional_call  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import GenomeShape  # noqa: E402
from anlage.genome import DecodeContext, Genome  # noqa: E402
from anlage.latents import LatentBank  # noqa: E402
from anlage.lifetime import evaluate  # noqa: E402
from anlage.model import Phenotype, key_name  # noqa: E402
from anlage.recipe import TrainConfig, batch_at, lr_at  # noqa: E402
from anlage.regress import decoded_erank_report  # noqa: E402
from gnomon.utils.rng import derive_seed  # noqa: E402


class _Loss(nn.Module):
    def __init__(self, model: Phenotype):
        super().__init__()
        self.model = model

    def forward(self, window):
        ce, zl = self.model.loss(window)
        return ce, zl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-g", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d-z", type=int, default=16)
    ap.add_argument("--d-h", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda-z", type=float, default=1e-3)
    ap.add_argument("--lambda-g", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5_000_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    shape = GenomeShape(d_g=args.d_g, H=args.H, d_z=args.d_z, d_h=args.d_h)
    tag = args.tag or f"dg{args.d_g}H{args.H}z{args.d_z}h{args.d_h}"
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"manifold-{lin.name}-s{args.stage}-{tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, _ = eval_sets(corpus, lin, device=device)
    pool = corpus.pool(n_windows=200_000, T=lin.T, pool_seed=args.pool_seed)

    G = Genome(shape, seed=1).to(device)
    bank = LatentBank(shape, cfg, lin.vocab, seed=1).to(device)
    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    ctx = DecodeContext(cfg, tables, bank, lin.vocab, tied=True)
    G.requires_grad_(True)
    bank.requires_grad_(True)
    params_g = list(G.parameters())
    params_z = list(bank.parameters())
    opt = torch.optim.Adam(params_g + params_z, lr=args.lr, betas=(0.9, 0.99))
    tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T, lr_provenance="probe")

    with torch.no_grad():
        D0 = G.decode_all(ctx)
    wrapper = _Loss(Phenotype(cfg, lin.vocab, tied=True, weights=D0).to(device))
    wrapper.requires_grad_(False)
    ce0, _ = evaluate(wrapper.model, test)
    n_g, n_z = G.n_params(), bank.n_params()
    log(f"MANIFOLD PROBE  {lin.name} {cfg.summary()}  shape {shape}  |Gamma| {n_g:,}  |Z| {n_z:,}  "
        f"phenotype {wrapper.model.n_params():,}  ratio {wrapper.model.n_params() / (n_g + n_z):.1f}x  "
        f"untrained decode CE {ce0:.4f}")

    def decoded_model():
        with torch.no_grad():
            D = G.decode_all(ctx)
        return Phenotype(cfg, lin.vocab, tied=True, weights=D).to(device)

    gen = pool.batches(batch=args.batch, steps=1 << 30, seed=args.seed)
    tokens_done, step, next_eval = 0, 0, args.eval_every
    curve, train_curve = [(0, ce0)], []
    ema = None
    step_ms = []
    t0 = time.time()
    while tokens_done < args.tokens:
        p = tokens_done / args.tokens
        for g in opt.param_groups:
            g["lr"] = lr_at(p, args.lr, tc)
        B = batch_at(p, tc)
        window = next(gen)[:B]
        ta = torch.cuda.Event(enable_timing=True)
        tb = torch.cuda.Event(enable_timing=True)
        ta.record()
        D = G.decode_all(ctx)
        fparams = {"model.W." + key_name(*k): v for k, v in D.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, zl = functional_call(wrapper, fparams, (window,))
        prior = args.lambda_z * sum(z.pow(2).sum() for z in params_z) \
            + args.lambda_g * sum(g_.pow(2).sum() for g_ in params_g)
        (ce + zl + prior).backward()
        gn = torch.nn.utils.clip_grad_norm_(params_g + params_z, args.clip)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        tb.record()
        tb.synchronize()
        step_ms.append(ta.elapsed_time(tb))
        tokens_done += B * lin.T
        step += 1
        cef = float(ce)
        ema = cef if ema is None else 0.98 * ema + 0.02 * cef
        if step % 50 == 0:
            z_rms = float(torch.sqrt(sum(z.pow(2).sum() for z in params_z) / max(1, sum(z.numel() for z in params_z))))
            train_curve.append((tokens_done, ema))
            log(f"    step {step:6d}  tokens {tokens_done:>12,}  train {ema:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                f"B {B}  |z|_rms {z_rms:.3f}  {step_ms[-1]:.0f} ms")
        if tokens_done >= next_eval:
            ce_h, _ = evaluate(decoded_model(), test)
            curve.append((tokens_done, ce_h))
            next_eval += args.eval_every
            log(f"  eval tokens {tokens_done:>12,}  held-out {ce_h:.4f}")
    model = decoded_model()
    final, strata = evaluate(model, test)
    curve.append((tokens_done, final))
    seconds = time.time() - t0
    ms = sorted(step_ms)[len(step_ms) // 2]
    log(f"final test CE {final:.4f}  {tokens_done / seconds:,.0f} tok/s  {ms:.0f} ms/step (median)  {seconds / 60:.1f} min")
    Wd = {k: v.detach() for k, v in model.weights().items()}
    probe_keys = [k for k in Wd if k[1] in (0, cfg.L // 2, cfg.L - 1, -1) and Wd[k].dim() == 2][:8]
    eranks = decoded_erank_report(Wd, Wd, probe_keys)
    for name, r in eranks.items():
        log(f"  erank {name}: {r['decoded']:.1f} of {r['min_dim']}")
    G.requires_grad_(False)
    bank.requires_grad_(False)
    torch.save({"genome": G.state_dict(), "bank": bank.state_dict(), "shape": shape.__dict__,
                "cfg": dataclasses.asdict(cfg)}, out / "genome.pt")
    rec = {"run": "MANIFOLD-PROBE", "lineage": lin.name, "stage": args.stage, "config": cfg.summary(),
           "shape": shape.__dict__, "n_genome": n_g, "n_latents": n_z, "n_phenotype": model.n_params(),
           "tokens": tokens_done, "steps": step, "lr": args.lr, "lambda_z": args.lambda_z,
           "lambda_g": args.lambda_g, "untrained_decode_ce": ce0, "final_test_ce": final,
           "final_strata": strata, "curve": curve, "train_curve": train_curve,
           "ms_per_step_median": ms, "tokens_per_s": tokens_done / seconds, "seconds": seconds,
           "eranks": eranks}
    write_json(out / "manifold.json", manifest(rec))
    log(f"wrote {out / 'manifold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
