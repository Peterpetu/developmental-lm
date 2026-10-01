"""Is there anything in the rule space worth evolving toward?

SPEC: 10 §10.6. Prior to asking whether evolution can *find* a better learning
rule is the question of whether one exists in the space at all. This pins
(A, B, C, D) globally -- the same for every identity, so no decoding is involved
-- and searches. If no point beats Adam, a decoded rule has nothing to find and
the mechanism is honest but pointless, which is a result and is reported as one.
"""
import argparse, dataclasses, json, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner

# (A, B, C, D) with a name and the hypothesis each point tests.
POINTS = [
    ("ADAM              ", (0.0, 1.0, 0.00, 0.0)),
    ("no momentum       ", (1.0, 0.0, 0.00, 0.0)),
    ("half/half         ", (0.5, 0.5, 0.00, 0.0)),
    ("momentum x1.5     ", (0.0, 1.5, 0.00, 0.0)),
    ("momentum x2.0     ", (0.0, 2.0, 0.00, 0.0)),
    ("momentum x0.5     ", (0.0, 0.5, 0.00, 0.0)),
    ("decay -0.03       ", (0.0, 1.0, -0.03, 0.0)),
    ("decay +0.03       ", (0.0, 1.0, 0.03, 0.0)),
    ("neuromod +0.5     ", (0.0, 1.0, 0.00, 0.5)),
    ("neuromod +1.0     ", (0.0, 1.0, 0.00, 1.0)),
    ("neuromod -0.5     ", (0.0, 1.0, 0.00, -0.5)),
    ("pure neuromod     ", (0.0, 0.0, 0.00, 1.0)),
    ("grad + neuromod   ", (0.6, 0.6, 0.00, 0.5)),
    ("momentum + decay  ", (0.0, 1.3, -0.03, 0.0)),
    ("all on            ", (0.3, 1.2, -0.02, 0.4)),
    ("all on, negative D", (0.3, 1.2, -0.02, -0.4)),
]

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--genomes", type=int, default=2)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--out", default="runs/gnomon/rulespace.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = dataclasses.replace(get_config(a.config), N_life=a.steps, batch=a.batch,
                          lr=a.lr)
shape = GenomeShape()
root = f"data/ts_{cfg.vocab}"
tr, te = Shard.load(root, "train"), Shard.load(root, "test")
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)
print(f"{cfg.name}  N_life {cfg.N_life}  batch {cfg.batch}  lr {cfg.lr}  "
      f"{len(POINTS)} points x {a.genomes} genomes x {a.seeds} seeds\n")
print(f"{'point':20s} {'(A, B, C, D)':26s} {'mean':>8s} {'sd':>7s} {'vs Adam':>9s}")

rows, t0, base = [], time.time(), None
for name, pt in POINTS:
    finals = []
    for gi in range(a.genomes):
        dec = Decoder(shape, seed=1000 + gi)
        for s in range(a.seeds):
            finals.append(runner.run(dec, seed=200 + s,
                                     coeff_override=pt).final)
    m = statistics.mean(finals)
    sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
    if base is None:
        base = m
    rows.append({"name": name.strip(), "point": pt, "finals": finals,
                 "mean": m, "sd": sd})
    print(f"{name} {str(pt):26s} {m:8.4f} {sd:7.4f} {m-base:+9.4f}"
          f"{'  <-- better' if m < base - 1e-9 else ''}", flush=True)

best = min(rows, key=lambda r: r["mean"])
print(f"\nbest point: {best['name']} {best['point']} at {best['mean']:.4f}")
print(f"Adam:       {rows[0]['mean']:.4f}   delta {best['mean']-rows[0]['mean']:+.4f} nats")
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {"config": cfg.name,
    "N_life": cfg.N_life, "batch": cfg.batch, "lr": cfg.lr, "rows": rows})
print(f"wrote {a.out}")
