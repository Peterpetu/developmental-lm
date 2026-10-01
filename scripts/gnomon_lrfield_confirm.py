"""Confirmation of §10.7's effect at replication that can carry a claim.

SPEC: 10 §10.7. The 7-field sweep found `freq^-0.25` beating uniform by 0.067
nats at n = 4 -- about one standard error, which is a pattern worth confirming
and not a result worth reporting. This is the same three fields, seed-paired,
at n = 12, with a BCa interval on the paired difference.
"""
import argparse, dataclasses, json, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, N_RULE, get_config
from gnomon.data.stream import Shard
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner
from gnomon.utils.stats import bca_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=0.09)
ap.add_argument("--genomes", type=int, default=4)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="runs/gnomon/lrfield-confirm.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = dataclasses.replace(get_config("m-pico"), N_life=a.steps, batch=a.batch,
                          lr=a.lr)
shape = GenomeShape()
tr, te = Shard.load("data/ts_4096", "train"), Shard.load("data/ts_4096", "test")
counts = np.bincount(np.asarray(tr.tokens[:20_000_000]).astype(np.int64),
                     minlength=cfg.vocab).astype(np.float64) + 1.0
freq = counts / counts.sum()
FIELDS = {"uniform": np.ones(cfg.vocab),
          "freq^-0.25": freq ** -0.25,
          "freq^-0.10": freq ** -0.10}
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)
pairs = [(g, s) for g in range(a.genomes) for s in range(a.seeds)]
print(f"m-pico  N_life {a.steps}  lr {a.lr}  {len(pairs)} paired lifetimes per field\n")

res = {}
for name, f in FIELDS.items():
    f = np.asarray(f, float); f = f / float((freq * f).sum())
    over = torch.zeros(cfg.vocab, N_RULE, device="cuda")
    over[:, 1] = torch.tensor(f, dtype=torch.float32, device="cuda")
    vals = []
    for g, s in pairs:
        vals.append(runner.run(Decoder(shape, seed=1000 + g), seed=200 + s,
                               coeff_override=(0.0, 1.0, 0.0, 0.0),
                               lr_field={"zeta": over}).final)
    res[name] = vals
    print(f"{name:12s} {statistics.mean(vals):.4f} +/- "
          f"{statistics.stdev(vals):.4f}", flush=True)

print()
for name in ("freq^-0.25", "freq^-0.10"):
    d = [u - v for u, v in zip(res["uniform"], res[name])]   # >0 means field wins
    r = bca_bootstrap(d, lambda x: sum(x) / len(x), n_resamples=6000,
                      alternative="greater")
    print(f"uniform - {name:11s}  mean {statistics.mean(d):+.4f} nats  "
          f"95% BCa [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  "
          f"{'REAL' if r.ci_low > 0 else 'not distinguishable from zero'}")
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {"lr": a.lr, "res": res})
print(f"\nwrote {a.out}")
