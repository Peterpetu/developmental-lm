"""The learning rate, measured at the horizon it is used at.

SPEC: 05 §5.4, SPEC: 08 §8.3, SPEC: 10 §10.6

§10.6 records this as a defect in every comparison the project had made:

    `lr` was `0.03` throughout, chosen from a sweep at 2,000 steps and then used
    at 300 and 600. Adam's curve above shows `0.03` is far from optimal at those
    horizons. Every GNOMON-versus-baseline number recorded earlier is therefore
    at an untuned `lr` for at least one side [...] Those runs are re-baselined
    before any of them is reported as a result.

It was recorded and not fixed: `M_PICO.lr` stayed at `3e-3` and the completed
`EVO` run used it. This script is the fix. It sweeps `lr` at a **stated
horizon**, seed-paired across genomes, and reports the turning point with a BCa
interval on the paired difference from the best.

Two properties the design has to have, because §10.6's own sweep had neither:

* **The optimum must not sit at the boundary.** §10.6's grid ran to 0.12 and the
  best value *was* 0.12, so what it established was "at least 0.12", not an
  optimum. The grid here is extended until the curve turns, and the script says
  so explicitly when it does not.
* **The comparison is seed-paired.** The same genomes and the same lifetime
  seeds at every `lr`, so the differences between rates are differences in the
  rate.

    python scripts/gnomon_lrsweep.py --steps 150 --batch 128
    python scripts/gnomon_lrsweep.py --steps 600 --batch 32
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
from gnomon.utils.stats import bca_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--steps", type=int, default=150,
                help="the horizon the rate is being measured AT (SPEC: 10 §10.6)")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--genomes", type=int, default=3)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--rule", default="adam",
                choices=["adam", "eta", "dpr", "dpr-frozen"])
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--token-code", default="freq", choices=["freq", "bijection"])
ap.add_argument("--lrs", type=float, nargs="*", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True

# A geometric grid, wide enough that the curve can turn inside it. §10.6's
# stopped at 0.12 with its best value there, which establishes a lower bound
# and not an optimum.
LRS = a.lrs or [0.015, 0.03, 0.06, 0.09, 0.12, 0.18, 0.25, 0.35, 0.50]

cfg = dataclasses.replace(get_config(a.config), N_life=a.steps, batch=a.batch)
shape = GenomeShape(d_z=a.d_z)
root = f"data/ts_{cfg.vocab}"
tr, fi = Shard.load(root, "train"), Shard.load(root, "fitness")
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=fi,
                        device=a.device, compile_mode=None, n_eval_batches=8,
                        token_code=a.token_code)

pairs = [(g, s) for g in range(a.genomes) for s in range(a.seeds)]
print(f"{cfg.name}  N_life {a.steps}  batch {a.batch}  rule {a.rule}  "
      f"token code {runner.token_code}  d_z {a.d_z}")
print(f"{len(LRS)} rates x {len(pairs)} seed-paired lifetimes = "
      f"{len(LRS) * len(pairs)} lifetimes\n")
print(f"{'lr':>7s} {'mean CE':>9s} {'sd':>7s} {'vs best':>9s}   {'s':>6s}")

t0 = time.time()
res: dict[str, list[float]] = {}
for lr in LRS:
    runner.cfg = dataclasses.replace(cfg, lr=lr)
    vals = []
    for g, s in pairs:
        # The same genome and the same lifetime seed at every rate, so the
        # difference between two rows is the rate and nothing else.
        vals.append(runner.run(Decoder(shape, seed=3000 + g),
                               seed=700 + s, rule=a.rule).final)
    res[f"{lr:g}"] = vals
    print(f"{lr:7.3f} {statistics.mean(vals):9.4f} "
          f"{statistics.stdev(vals) if len(vals) > 1 else 0.0:7.4f} "
          f"{'':>9s}   {time.time() - t0:6.0f}", flush=True)

means = {k: statistics.mean(v) for k, v in res.items()}
best = min(means, key=means.get)
print(f"\n{'lr':>7s} {'mean CE':>9s} {'vs best':>9s}  {'95% BCa on the paired diff':>30s}")
rows = []
for k, v in res.items():
    d = [x - y for x, y in zip(v, res[best])]          # paired: this lr - best
    if k == best:
        print(f"{float(k):7.3f} {means[k]:9.4f} {0.0:+9.4f}   (the best)")
        rows.append({"lr": float(k), "mean": means[k], "delta": 0.0,
                     "ci": [0.0, 0.0], "finals": v})
        continue
    r = bca_bootstrap(d, lambda x: sum(x) / len(x), n_resamples=6000,
                      alternative="two-sided")
    print(f"{float(k):7.3f} {means[k]:9.4f} {means[k] - means[best]:+9.4f}   "
          f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
          f"{'  (indistinguishable)' if r.ci_low <= 0 <= r.ci_high else ''}")
    rows.append({"lr": float(k), "mean": means[k],
                 "delta": means[k] - means[best],
                 "ci": [r.ci_low, r.ci_high], "finals": v})

i = LRS.index(float(best))
print(f"\nbest lr = {best} at N_life = {a.steps}")
if i in (0, len(LRS) - 1):
    print(f"  *** THE OPTIMUM IS AT THE EDGE OF THE GRID. This measures a bound, "
          f"not a turning point; extend --lrs and re-run before quoting it "
          f"(SPEC: 10 §10.6).")
else:
    print(f"  the curve turns inside the grid: {LRS[i-1]:g} -> {best} -> "
          f"{LRS[i+1]:g} gives {means[f'{LRS[i-1]:g}']:.4f} -> "
          f"{means[best]:.4f} -> {means[f'{LRS[i+1]:g}']:.4f}")

out = a.out or f"runs/gnomon/lrsweep-{a.config}-n{a.steps}-b{a.batch}.json"
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
write_json(out, {
    "config": a.config, "N_life": a.steps, "batch": a.batch, "rule": a.rule,
    "d_z": a.d_z, "token_code": runner.token_code,
    "provenance": runner.token_provenance,
    "genomes": a.genomes, "seeds": a.seeds,
    "best_lr": float(best), "edge_of_grid": i in (0, len(LRS) - 1),
    "rows": rows,
})
print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")
