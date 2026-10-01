"""R-ADAM vs R-ETA vs R-DPR vs R-DPR-FROZEN, at matched budget.

SPEC: 10 §10.6. The comparison §10.1's defect made impossible: how much of a
lifetime's progress is attributable to a mechanism the genome controls?
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

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--genomes", type=int, default=4)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--arms", nargs="*",
                default=["adam", "eta", "dpr", "dpr-frozen"])
ap.add_argument("--out", default="runs/gnomon/rules.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = dataclasses.replace(get_config(a.config), N_life=a.steps, batch=a.batch,
                          lr=a.lr)
shape = GenomeShape(d_z=a.d_z)
root = f"data/ts_{cfg.vocab}"
tr, te = Shard.load(root, "train"), Shard.load(root, "test")
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)
print(f"{cfg.name}  N_life {cfg.N_life}  batch {cfg.batch}  lr {cfg.lr}  "
      f"{a.genomes} genomes x {a.seeds} seeds x {len(a.arms)} arms\n")

rows, t0 = [], time.time()
for arm in a.arms:
    finals = []
    for gi in range(a.genomes):
        dec = Decoder(shape, seed=1000 + gi)
        for s in range(a.seeds):
            r = runner.run(dec, seed=200 + s, rule=arm)
            finals.append(r.final)
    m = statistics.mean(finals)
    sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
    rows.append({"arm": arm, "finals": finals, "mean": m, "sd": sd,
                 "report": runner.last_rule_report})
    print(f"R-{arm.upper():11s} {m:.4f} +/- {sd:.4f}   "
          f"[{' '.join(f'{v:.3f}' for v in finals)}]   ({time.time()-t0:.0f}s)")

base = [r for r in rows if r["arm"] == "adam"]
if base:
    b = base[0]["mean"]
    print(f"\nagainst R-ADAM ({b:.4f}):")
    for r in rows:
        if r["arm"] != "adam":
            print(f"  R-{r['arm'].upper():11s} {r['mean']-b:+.4f} nats "
                  f"({'better' if r['mean'] < b else 'worse'})")
rep = rows[-1].get("report") or {}
if rep:
    print("\nwhat the genome chose (mean coefficient, last lifetime):")
    for k in sorted(rep):
        print(f"  {k:12s} {rep[k]:+.4f}")
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {"config": cfg.name, "N_life": cfg.N_life, "batch": cfg.batch,
     "lr": cfg.lr, "rows": rows})
print(f"\nwrote {a.out}")
