#!/usr/bin/env python
"""R4 INIT: the scale-profile birth against the free initialisation (K-INIT).

SPEC: 12 §12.6
SPEC: 12 §12.8 (K-INIT)

    python scripts/anlage_init.py --checkpoint runs/anlage/noise-tinystories-s0/seed0.pt --seeds 1 2 3 4

`s` is fit to the vector roles of a trained adult (12.3); a birth from `s` decodes every
vector role and leaves every matrix at the innate initialisation of seed `k`. Paired with
the plain free initialisation of the same seed, both trained for a short budget (10% of the
primordium's), the contrast is the value of the inherited scale profile at birth and after
the first steps. The adult whose profile is fit is never one of the paired seeds.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402
from anlage_analyse import paired  # noqa: E402

import torch  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import GenomeShape  # noqa: E402
from anlage.genome import DecodeContext  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, load_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from anlage.regress import RegressConfig, express_scale_profile, fit_scale_profile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--checkpoint", default="runs/anlage/noise-tinystories-s0/seed0.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-fit", type=int, default=500)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--noise", default="runs/anlage/noise-tinystories-s0/noise.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(0)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"init-{lin.name}-s0")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    shape = GenomeShape()
    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    adult = load_phenotype(args.checkpoint, device=device)
    G, bank, rep = fit_scale_profile(adult, shape, tables, rc=RegressConfig(n_fit=args.n_fit, lr=2e-3), log=log)
    log(f"R4 INIT  {lin.name} {cfg.summary()}  s fit on {args.checkpoint}: residual per role "
        f"{ {k: round(v, 4) for k, v in rep.residual.items()} }  mean {rep.residual_mean:.4f}  {rep.seconds:.0f}s")
    ctx = DecodeContext(cfg, tables, bank, lin.vocab, tied=True)
    sigma_e = json.loads(pathlib.Path(args.noise).read_text())["sigma_e"]
    tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon, lr_emb=args.lr_emb,
                     lr_provenance="R1 (R2 not run)")
    rows = []
    for seed in args.seeds:
        free = Phenotype(cfg, lin.vocab, seed=seed).to(device)
        W_free = {k: v.detach().clone() for k, v in free.weights().items()}
        W_s = express_scale_profile(G, ctx, W_free)
        born = Phenotype(cfg, lin.vocab, weights=W_s).to(device)
        ce0_free, _ = evaluate(free, test)
        ce0_s, _ = evaluate(born, test)
        r_free = train(free, pool, test, tc, order_seed=seed, compile_mode=None, log=None)
        r_s = train(born, pool, test, tc, order_seed=seed, compile_mode=None, log=None)
        rows.append({"seed": seed, "birth_free": ce0_free, "birth_s": ce0_s, "final_free": r_free.final_ce,
                     "final_s": r_s.final_ce, "curve_free": r_free.curve, "curve_s": r_s.curve})
        log(f"  seed {seed}: at birth free {ce0_free:.4f} / s {ce0_s:.4f};  after {args.tokens:,} tokens "
            f"free {r_free.final_ce:.4f} / s {r_s.final_ce:.4f}  (d {r_s.final_ce - r_free.final_ce:+.4f})")
        del free, born
        torch.cuda.empty_cache()
    c_birth = paired([r["birth_free"] for r in rows], [r["birth_s"] for r in rows])
    c_final = paired([r["final_free"] for r in rows], [r["final_s"] for r in rows])
    k_init = bool(c_final["mean"] is not None and c_final["mean"] <= -3 * sigma_e
                  and c_final.get("ci_high") is not None and c_final["ci_high"] < 0)
    log(f"  at birth: s - free = {c_birth['mean']:+.4f}"
        + (f" [{c_birth['ci_low']:+.4f}, {c_birth['ci_high']:+.4f}]" if c_birth.get("ci_low") is not None else ""))
    log(f"  after {args.tokens:,} tokens: s - free = {c_final['mean']:+.4f}"
        + (f" [{c_final['ci_low']:+.4f}, {c_final['ci_high']:+.4f}]" if c_final.get("ci_low") is not None else "")
        + f" = {c_final['mean'] / sigma_e:+.1f} sigma_e(R1)  K-INIT {'PASS' if k_init else 'FIRES'}")
    rec = {"run": "R4-INIT", "lineage": lin.name, "config": cfg.summary(), "checkpoint": args.checkpoint,
           "shape": shape.__dict__, "fit": rep.as_dict(), "tokens": args.tokens, "batch": args.batch,
           "seeds": args.seeds, "rows": rows, "contrast_birth": c_birth, "contrast_final": c_final,
           "sigma_e": sigma_e, "k_init_pass": k_init}
    write_json(out / "init.json", manifest(rec))
    log(f"wrote {out / 'init.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
