#!/usr/bin/env python
"""R4b SLEEP: a stage-0 lifetime with in-lifetime consolidation, paired with R1's seed.

SPEC: 05 §5.7
SPEC: 08 §8.5

    python scripts/anlage_sleep.py --seed 0 --every 100 --n-refit 100 --eta-anchor 0.01

The lifetime is R1's seed-`s` lifetime under CRN (same initialisation, pool and
order), with `anlage.sleep.Sleeper` attached. Reported:

  * lifetime CE against R1's seed-`s` CE (K-ANCHOR: the pull must not cost CE);
  * the drift/residual/CE(D) record at every sleep;
  * the final regression and the at-birth CE of `D(Gamma, Z)` (K-BIRTH);
  * the overhead of sleeping (seconds asleep / seconds awake).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402

import torch  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import GenomeShape  # noqa: E402
from anlage.genome import DecodeContext, Genome  # noqa: E402
from anlage.latents import LatentBank  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, save_phenotype  # noqa: E402
from anlage.recipe import TrainConfig  # noqa: E402
from anlage.regress import RegressConfig, regress  # noqa: E402
from anlage.sleep import Sleeper  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--d-g", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d-z", type=int, default=16)
    ap.add_argument("--d-h", type=int, default=256)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--n-refit", type=int, default=100)
    ap.add_argument("--refit-lr", type=float, default=2e-3)
    ap.add_argument("--eta-anchor", type=float, default=0.01)
    ap.add_argument("--mode", default="anchor", choices=["anchor", "regenerate", "none"])
    ap.add_argument("--regen-eps", type=float, default=0.5)
    ap.add_argument("--final-fit", type=int, default=1000)
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--noise", default="runs/anlage/noise-tinystories-s0/noise.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda"
    lin = get_lineage(args.lineage)
    cfg = lin.stage(args.stage)
    shape = GenomeShape(d_g=args.d_g, H=args.H, d_z=args.d_z, d_h=args.d_h)
    tag = args.tag or f"{args.mode}-e{args.every}-r{args.n_refit}-a{args.eta_anchor:g}-dg{args.d_g}H{args.H}z{args.d_z}"
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"sleep-{lin.name}-s{args.stage}-{tag}-seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    pool = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    tc = TrainConfig(tokens=args.tokens, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon,
                     lr_emb=args.lr_emb, lr_provenance="R1 (R2 not run)")
    paired = None
    npath = pathlib.Path(args.noise)
    if npath.exists():
        noise = json.loads(npath.read_text())
        if (noise.get("tokens") == args.tokens and noise.get("config") == cfg.summary()
                and noise.get("pool_seed") == args.pool_seed and args.seed < len(noise["final_test_ce"])):
            paired = {"free_ce": noise["final_test_ce"][args.seed], "sigma_e": noise["sigma_e"]}

    model = Phenotype(cfg, lin.vocab, seed=args.seed)
    G = Genome(shape, seed=1).to(device)
    bank = LatentBank(shape, cfg, lin.vocab, seed=1).to(device)
    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    ctx = DecodeContext(cfg, tables, bank, lin.vocab, tied=True)
    L_null = float(torch.log(torch.tensor(float(lin.vocab))))

    def eval_decode(D):
        return evaluate(Phenotype(cfg, lin.vocab, tied=True, weights=D).to(device), test)[0]

    sleeper = Sleeper(G, ctx, model.to(device), tc, every=args.every, n_refit=args.n_refit,
                      eta_anchor=args.eta_anchor, mode=args.mode, regen_eps=args.regen_eps,
                      rc=RegressConfig(lr=args.refit_lr), eval_decode=eval_decode, log=log)
    log(f"R4b SLEEP  {lin.name} {cfg.summary()}  shape {shape}  tokens {args.tokens:,}  seed {args.seed}  "
        f"mode {args.mode}  every {args.every}  n_refit {args.n_refit}  eta_anchor {args.eta_anchor}  "
        f"paired {paired}")
    res = train(model, pool, test, tc, order_seed=args.seed, compile_mode=None,
                hooks={"before_step": sleeper.before_step}, log=log,
                eval_every_tokens=args.tokens // 5)
    ce_life = res.final_ce
    ce_fit, _ = evaluate(model, fit)
    log(f"  lifetime: test {ce_life:.4f}  fit {ce_fit:.4f}  {res.tokens_per_s:,.0f} tok/s  "
        f"asleep {sleeper.seconds_asleep:.0f}s of {res.seconds:.0f}s")
    # the final regression, warm: the genome has tracked the weights all along
    W = {k: v.detach() for k, v in model.weights().items()}
    rep = regress(G, ctx, W, RegressConfig(n_fit=args.final_fit, lr=args.refit_lr), log=log)
    with torch.no_grad():
        D = G.decode_all(ctx)
    ce_birth = eval_decode(D)
    birth_share = (L_null - ce_birth) / (L_null - ce_life) if L_null > ce_life else float("nan")
    log(f"  final fit residual {rep.residual_mean:.4f}  at-birth CE {ce_birth:.4f}  "
        f"birth_share {birth_share:.3f}  (trained {ce_life:.4f}, null {L_null:.4f})")
    if paired:
        d = ce_life - paired["free_ce"]
        log(f"  K-ANCHOR contrast: sleep - free = {d:+.4f} nats = {d / paired['sigma_e']:+.1f} sigma_e")
    save_phenotype(model, out / "adult.pt")
    torch.save({"genome": G.state_dict(), "bank": bank.state_dict(), "shape": shape.__dict__,
                "cfg": dataclasses.asdict(cfg)}, out / "genome.pt")
    rec = {"run": "R4b-SLEEP", "lineage": lin.name, "stage": args.stage, "config": cfg.summary(),
           "shape": shape.__dict__, "tokens": args.tokens, "seed": args.seed, "pool_seed": args.pool_seed,
           "train_config": dataclasses.asdict(tc), "sleep": sleeper.as_dict(),
           "lifetime": res.as_dict(), "fitness_ce": ce_fit, "paired": paired,
           "final_regress": rep.as_dict(), "birth_ce": ce_birth, "birth_share": birth_share,
           "L_null": L_null, "overhead": sleeper.seconds_asleep / max(res.seconds - sleeper.seconds_asleep, 1e-9)}
    write_json(out / "sleep.json", manifest(rec))
    log(f"wrote {out / 'sleep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
