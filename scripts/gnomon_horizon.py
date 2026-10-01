"""How short can a fitness lifetime be and still rank genomes?

SPEC: 06 §6.2, SPEC: 08 §8.3 -- the evolutionary budget is set by this, so it is
measured rather than assumed.
"""
import argparse, dataclasses, json, math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.eval.share import heritability
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--steps", type=int, nargs="+", default=[50, 100, 150])
ap.add_argument("--genomes", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="runs/gnomon/horizon.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

base = get_config(a.config); shape = GenomeShape()
root = f"data/ts_{base.vocab}"
tr, fi = Shard.load(root, "train"), Shard.load(root, "fitness")
rows = []
for N in a.steps:
    cfg = dataclasses.replace(base, N_life=N, batch=a.batch)
    r = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=fi,
                       device="cuda", compile_mode=None, n_eval_batches=4)
    P, t0 = [], time.time()
    for g in range(a.genomes):
        dec = Decoder(shape, seed=1000 + g)
        P.append([r.run(dec, seed=200 + s).final for s in range(a.seeds)])
    h = heritability(P, n_resamples=3000)
    dt = time.time() - t0
    per = dt / (a.genomes * a.seeds)
    rows.append({"N_life": N, "batch": a.batch, "h2": h.h2, "ci_low": h.ci_low,
                 "ci_high": h.ci_high, "sigma_P": h.sigma_P, "V_G": h.V_G,
                 "V_E": h.V_E, "passes": h.passes, "sec_per_lifetime": per,
                 "best": min(h.means), "worst": max(h.means), "P": P})
    print(f"N_life {N:4d} batch {a.batch:4d} | {per:5.1f}s/lifetime | "
          f"h2 {h.h2:.3f} [{h.ci_low:.3f}, {h.ci_high:.3f}] "
          f"sigma_P {h.sigma_P:.3f} | best {min(h.means):.4f} "
          f"worst {max(h.means):.4f} | {'PASS' if h.passes else 'FAIL'}",
          flush=True)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, rows)
print(f"\nwrote {a.out}")
