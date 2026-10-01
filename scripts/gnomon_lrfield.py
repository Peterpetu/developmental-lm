"""The ceiling of a per-identity learning-rate field.

SPEC: 10 §10.7. The one part of §10.2's rule that §10.5b's learning-rate control
does not rule out is a *per-identity* scale. Its ceiling is measurable with no
evolution at all: if a hand-designed field cannot beat a uniform one, the genome
has nothing to find there either.

**Normalisation is the whole design of this script.** A first version scaled each
field to arithmetic mean 1, which is *not* a fair comparison of shapes: a
heavy-tailed field with arithmetic mean 1 puts most of its mass far below 1, so
the tokens that dominate the loss all learn more slowly and the field loses for
a reason that has nothing to do with its shape. Fields are therefore normalised
so that the **occurrence-weighted** mean is 1,

    sum_v  freq_v * field_v  =  1

which matches the expected step size per token *seen*, which is the quantity the
loss actually responds to. Both normalisations are reported so the difference is
visible rather than asserted.
"""

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, N_RULE, get_config
from gnomon.data.stream import Shard
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=0.09)
ap.add_argument("--genomes", type=int, default=2)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--norm", default="occurrence",
                choices=["occurrence", "arithmetic"])
ap.add_argument("--out", default="runs/gnomon/lrfield.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = dataclasses.replace(get_config("m-pico"), N_life=a.steps, batch=a.batch,
                          lr=a.lr)
shape = GenomeShape()
tr = Shard.load("data/ts_4096", "train")
te = Shard.load("data/ts_4096", "test")

counts = np.bincount(np.asarray(tr.tokens[:20_000_000]).astype(np.int64),
                     minlength=cfg.vocab).astype(np.float64) + 1.0
freq = counts / counts.sum()

# Exponents of frequency, from strongly favouring rare tokens to strongly
# favouring common ones, plus two shape controls.
FIELDS = {
    "uniform            ": np.ones(cfg.vocab),
    "freq^-0.50 (rare+) ": freq ** -0.50,
    "freq^-0.25         ": freq ** -0.25,
    "freq^-0.10         ": freq ** -0.10,
    "freq^+0.10         ": freq ** 0.10,
    "freq^+0.25 (common+)": freq ** 0.25,
    "random lognormal   ": np.random.default_rng(0).lognormal(0.0, 0.7, cfg.vocab),
}

runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)
print(f"m-pico  N_life {a.steps}  batch {a.batch}  lr {a.lr}  "
      f"{len(FIELDS)} fields x {a.genomes} genomes x {a.seeds} seeds")
print(f"normalisation: {a.norm} mean = 1 "
      f"({'sum_v freq_v * field_v' if a.norm == 'occurrence' else 'mean(field)'})\n")
print(f"{'field':21s} {'max/min':>9s} {'arith mn':>9s} {'mean':>8s} "
      f"{'sd':>7s} {'vs uniform':>11s}")

rows, base = [], None
for name, f in FIELDS.items():
    f = np.asarray(f, dtype=np.float64)
    f = f / (float((freq * f).sum()) if a.norm == "occurrence" else float(f.mean()))
    over = torch.zeros(cfg.vocab, N_RULE, device="cuda")
    over[:, 1] = torch.tensor(f, dtype=torch.float32, device="cuda")
    finals = []
    for gi in range(a.genomes):
        dec = Decoder(shape, seed=1000 + gi)
        for s in range(a.seeds):
            finals.append(runner.run(dec, seed=200 + s,
                                     coeff_override=(0.0, 1.0, 0.0, 0.0),
                                     lr_field={"zeta": over}).final)
    m = statistics.mean(finals)
    sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
    if base is None:
        base = m
    rows.append({"field": name.strip(), "mean": m, "sd": sd, "finals": finals,
                 "spread": float(f.max() / f.min()),
                 "arithmetic_mean": float(f.mean())})
    print(f"{name} {f.max()/f.min():9.1f} {f.mean():9.3f} {m:8.4f} {sd:7.4f} "
          f"{m - base:+11.4f}{'  <-- better' if m < base - 1e-9 else ''}",
          flush=True)

best = min(rows, key=lambda r: r["mean"])
print(f"\nbest field: {best['field']} at {best['mean']:.4f}   "
      f"uniform {rows[0]['mean']:.4f}   delta {best['mean'] - rows[0]['mean']:+.4f}")
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {"lr": a.lr, "norm": a.norm, "rows": rows})
print(f"wrote {a.out}")
