"""Is the rule space's structure real, or is `B` just a learning rate?

SPEC: 10 §10.6. The sweep found `B = 2` beating `B = 1` by 0.171 nats. But `B`
multiplies the whole displacement, so `B = 2` at `lr` is *exactly* `B = 1` at
`2 lr`: the apparent structure along `B` may be nothing but a mis-set learning
rate. The same confound applies to `D`, whose term adds `E[nu] * D` to the
effective scale.

This control varies `lr` for Adam over the range those points imply, and
compares each rule point against the Adam run of matched **effective** scale.
Anything left over after that is structure the rule has and a learning rate does
not.
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

NU_MEAN = 0.33          # measured in runs/gnomon/rules.json

# (label, (A,B,C,D), lr). `eff` is the expected scale of the displacement
# relative to Adam at the same lr: A + B + D * E[nu].
TRIALS = [
    ("Adam            lr 0.015", (0.0, 1.0, 0.0, 0.0), 0.015),
    ("Adam            lr 0.030", (0.0, 1.0, 0.0, 0.0), 0.030),
    ("Adam            lr 0.040", (0.0, 1.0, 0.0, 0.0), 0.040),
    ("Adam            lr 0.045", (0.0, 1.0, 0.0, 0.0), 0.045),
    ("Adam            lr 0.060", (0.0, 1.0, 0.0, 0.0), 0.060),
    ("Adam            lr 0.090", (0.0, 1.0, 0.0, 0.0), 0.090),
    ("Adam            lr 0.120", (0.0, 1.0, 0.0, 0.0), 0.120),
    ("B=2.0           lr 0.030", (0.0, 2.0, 0.0, 0.0), 0.030),
    ("B=1.5           lr 0.030", (0.0, 1.5, 0.0, 0.0), 0.030),
    ("D=+1.0          lr 0.030", (0.0, 1.0, 0.0, 1.0), 0.030),
    ("D=-0.5          lr 0.030", (0.0, 1.0, 0.0, -0.5), 0.030),
    ("A=.6 B=.6 D=.5  lr 0.030", (0.6, 0.6, 0.0, 0.5), 0.030),
    # The sharpest test of the neuromodulator: same *effective* scale as Adam at
    # lr 0.030, with the gating moved into D. If gating carries signal, this
    # beats Adam at 0.030; if D is only a scale, it ties.
    ("D=+1.0 B=.67    lr 0.030", (0.0, 0.67, 0.0, 1.0), 0.030),
    ("D=-1.0 B=1.33   lr 0.030", (0.0, 1.33, 0.0, -1.0), 0.030),
]

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--genomes", type=int, default=2)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--out", default="runs/gnomon/rulecontrol.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

base_cfg = dataclasses.replace(get_config("m-pico"), N_life=a.steps,
                               batch=a.batch)
shape = GenomeShape()
tr = Shard.load("data/ts_4096", "train")
te = Shard.load("data/ts_4096", "test")
print(f"m-pico  N_life {a.steps}  batch {a.batch}  "
      f"{len(TRIALS)} trials x {a.genomes} genomes x {a.seeds} seeds")
print(f"E[nu] taken as {NU_MEAN} from the generation-0 report\n")
print(f"{'trial':26s} {'eff lr':>8s} {'mean':>8s} {'sd':>7s}")

rows, t0 = [], time.time()
for label, pt, lr in TRIALS:
    cfg = dataclasses.replace(base_cfg, lr=lr)
    runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                            device="cuda", compile_mode=None, n_eval_batches=8)
    finals = []
    for gi in range(a.genomes):
        dec = Decoder(shape, seed=1000 + gi)
        for s in range(a.seeds):
            finals.append(runner.run(dec, seed=200 + s, coeff_override=pt).final)
    m = statistics.mean(finals)
    sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
    eff = lr * (pt[0] + pt[1] + pt[3] * NU_MEAN)
    rows.append({"label": label.strip(), "point": pt, "lr": lr, "eff_lr": eff,
                 "finals": finals, "mean": m, "sd": sd})
    print(f"{label:26s} {eff:8.4f} {m:8.4f} {sd:7.4f}", flush=True)

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {"rows": rows})
print(f"\nwrote {a.out}")
