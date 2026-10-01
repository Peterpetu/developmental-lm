#!/usr/bin/env python
"""R4 BIRTH: regress a trained phenotype into the genome and measure the decode.

SPEC: 05 §5.2
SPEC: 05 §5.5
SPEC: 05 §5.6
SPEC: 08 §8.1

    python scripts/anlage_birth.py --checkpoint runs/anlage/noise-tinystories-s0/seed0.pt

Reports per-role residuals, decoded vs trained effective rank, both compression
ratios, the at-birth CE with the shuffle and random-genome controls, the
`d_z = 0` arm, and (K-SCALE) the at-birth CE of the *grown* decode at the next
stage against a random-latent decode.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import torch  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import (GenomeShape, ROLE_EMB, ROLE_Q, ROLE_UP,  # noqa: E402
                           ROLE_DOWN, ROLE_O)
from anlage.genome import DecodeContext, Genome  # noqa: E402
from anlage.grow import grow, layer_placement  # noqa: E402
from anlage.latents import LatentBank  # noqa: E402
from anlage.lifetime import evaluate  # noqa: E402
from anlage.model import Phenotype, load_phenotype  # noqa: E402
from anlage.regress import (RegressConfig, decoded_erank_report, regress,  # noqa: E402
                            shuffled)


def _phen(cfg, vocab, W, device):
    return Phenotype(cfg, vocab, weights={k: v.detach() for k, v in W.items()}).to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d-g", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d-z", type=int, default=16)
    ap.add_argument("--d-h", type=int, default=256)
    ap.add_argument("--no-scale", action="store_true", help="skip the K-SCALE grown decode")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    shape = GenomeShape(d_g=args.d_g, H=args.H, d_z=args.d_z, d_h=args.d_h)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" /
                       f"birth-{lin.name}-s{args.stage}-dg{args.d_g}H{args.H}z{args.d_z}{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    model = load_phenotype(args.checkpoint, device=device)
    assert model.cfg == cfg, (model.cfg, cfg)
    W = {k: v.detach() for k, v in model.weights().items()}
    ce_adult, _ = evaluate(model, test)
    L_null = math.log(lin.vocab)
    log(f"R4 BIRTH  {lin.name} {cfg.summary()}  shape {shape}  adult CE {ce_adult:.4f}  "
        f"L_null {L_null:.4f}")

    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    bank = LatentBank(shape, cfg, lin.vocab, seed=1).to(device)
    ctx = DecodeContext(cfg, tables, bank, lin.vocab, tied=model.tied)
    G = Genome(shape, seed=1).to(device)

    # random-genome control at birth
    with torch.no_grad():
        D_rand = G.decode_all(ctx)
    ce_rand, _ = evaluate(_phen(cfg, lin.vocab, D_rand, device), test)
    log(f"  random genome at birth: {ce_rand:.4f}")

    rep = regress(G, ctx, W, RegressConfig(n_fit=args.n_fit, lr=args.lr), log=log)
    with torch.no_grad():
        D = G.decode_all(ctx)
    ce_birth, strata_birth = evaluate(_phen(cfg, lin.vocab, D, device), test)
    share = (L_null - ce_birth) / (L_null - ce_adult)
    log(f"  fit: {rep.steps} steps, mean residual {rep.residual_mean:.4f}, "
        f"{rep.seconds:.0f}s; ratios {rep.ratio_genome:.0f}x / {rep.ratio_genome_latents:.0f}x")
    log(f"  at-birth CE {ce_birth:.4f}  birth_share {share:.3f}")
    ce_shuf, _ = evaluate(_phen(cfg, lin.vocab, shuffled(D, torch.Generator().manual_seed(0)),
                                device), test)
    log(f"  shuffle control {ce_shuf:.4f}")
    er = decoded_erank_report(D, W, [(ROLE_Q, 0), (ROLE_UP, 1), (ROLE_DOWN, 2), (ROLE_O, 3),
                                     (ROLE_EMB, -1)])
    for k, v in er.items():
        log(f"  erank {k}: decoded {v['decoded']:.1f} / trained {v['trained']:.1f} of {v['min_dim']}")

    # d_z = 0 arm
    shape0 = GenomeShape(d_g=args.d_g, H=args.H, d_z=0, d_h=args.d_h)
    tables0 = CodeTables(shape0, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    bank0 = LatentBank(shape0, cfg, lin.vocab, seed=1).to(device)
    ctx0 = DecodeContext(cfg, tables0, bank0, lin.vocab, tied=model.tied)
    G0 = Genome(shape0, seed=1).to(device)
    rep0 = regress(G0, ctx0, W, RegressConfig(n_fit=args.n_fit, lr=args.lr), log=None)
    with torch.no_grad():
        D0 = G0.decode_all(ctx0)
    ce_birth0, _ = evaluate(_phen(cfg, lin.vocab, D0, device), test)
    log(f"  d_z=0 arm: residual {rep0.residual_mean:.4f}, at-birth CE {ce_birth0:.4f}")

    scale = None
    if not args.no_scale and args.stage + 1 < len(lin.stages):
        cfg1 = lin.stage(args.stage + 1)
        lm, ops1 = layer_placement(cfg, cfg1)
        import dataclasses
        cfg1 = dataclasses.replace(cfg1, ops_override=ops1)
        g = torch.Generator().manual_seed(0)
        bank1 = bank.grown(cfg1, lm, g, mode="sample")
        tables1 = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg1.L, device=device)
        ctx1 = DecodeContext(cfg1, tables1, bank1, lin.vocab, tied=model.tied)
        with torch.no_grad():
            D1 = G.decode_all(ctx1)
        grown, _ = grow(model, cfg1, mode="genome", decoded=D1)
        ce_grown, _ = evaluate(grown, test)
        rnd, _ = grow(model, cfg1, mode="random", generator=torch.Generator().manual_seed(1))
        ce_grown_rand, _ = evaluate(rnd, test)
        cp, _ = grow(model, cfg1, mode="copy", generator=torch.Generator().manual_seed(1))
        ce_grown_copy, _ = evaluate(cp, test)
        full1, _ = evaluate(_phen(cfg1, lin.vocab, D1, device), test)
        scale = {"stage_next": cfg1.summary(), "grown_genome_ce": ce_grown,
                 "grown_random_ce": ce_grown_rand, "grown_copy_ce": ce_grown_copy,
                 "decode_only_next_stage_ce": full1}
        log(f"  K-SCALE grown to {cfg1.summary()} (no gates): genome {ce_grown:.4f}  "
            f"copy {ce_grown_copy:.4f}  random {ce_grown_rand:.4f}  pure decode {full1:.4f}")

    torch.save({"genome": G.state_dict(), "bank": bank.state_dict(),
                "shape": shape.__dict__}, out / "genome.pt")
    report = manifest({
        "run": "R4-BIRTH", "lineage": lin.name, "stage": args.stage,
        "checkpoint": args.checkpoint, "shape": shape.__dict__,
        "adult_ce": ce_adult, "L_null": L_null, "random_genome_ce": ce_rand,
        "birth_ce": ce_birth, "birth_share": share, "birth_strata": strata_birth,
        "shuffle_ce": ce_shuf, "regress": rep.as_dict(), "erank": er,
        "dz0": {"residual_mean": rep0.residual_mean, "birth_ce": ce_birth0,
                "regress": rep0.as_dict()},
        "scale": scale, "corpus": corpus.manifest(), "eval_test_digest": test.digest,
    })
    write_json(out / "birth.json", report)
    log(f"wrote {out / 'birth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
