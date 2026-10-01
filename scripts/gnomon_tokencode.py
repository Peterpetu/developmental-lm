"""Does a frequency-ordered token code help? SPEC: 02 §2.9.

§10.7 measured that a per-token step scaled as `freq^-0.25` beats a uniform one
by 0.062 nats. A genome cannot decode such a field from a code carrying no
frequency information, and §2.4's seeded bijection carries none (R^2 = 0.004
against 0.488 for §2.9's Gray-of-relative-rank). This asks whether restoring the
signal restores the gain -- seed-paired, both codes, same genomes.
"""
import argparse, dataclasses, json, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard, counts_digest, token_counts
from gnomon.genome import codes as C
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner
from gnomon.utils.stats import bca_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=0.09)
ap.add_argument("--genomes", type=int, default=4)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--rule", default="adam", choices=["adam", "eta", "dpr"])
ap.add_argument("--out", default="runs/gnomon/tokencode.json")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = dataclasses.replace(get_config("m-pico"), N_life=a.steps, batch=a.batch,
                          lr=a.lr)
shape = GenomeShape()
tr, te = Shard.load("data/ts_4096", "train"), Shard.load("data/ts_4096", "test")
counts = token_counts(tr, cfg.vocab)
rank = C.frequency_rank(counts)
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)
print(f"counts over {int(counts.sum() - counts.size):,} tokens, "
      f"digest {counts_digest(counts)}, {int((counts == 1.0).sum())} unseen types")
pairs = [(g, s) for g in range(a.genomes) for s in range(a.seeds)]
print(f"m-pico  N_life {a.steps}  lr {a.lr}  rule {a.rule}  "
      f"{len(pairs)} paired lifetimes per code\n")

res = {}
for name, rk in (("bijection (§2.4)", None), ("freq rank (§2.9)", rank)):
    # `set_token_code` rather than reaching into the decoder: switching the code
    # invalidates every compiled graph, and the runner is the only thing that
    # knows that. Reaching past it left a graph bound to the previous code.
    runner.set_token_code(rk)
    vals = [runner.run(Decoder(shape, seed=1000 + g), seed=200 + s, rule=a.rule).final
            for g, s in pairs]
    res[name] = vals
    print(f"{name:20s} {statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}",
          flush=True)

d = [u - v for u, v in zip(res["bijection (§2.4)"], res["freq rank (§2.9)"])]
r = bca_bootstrap(d, lambda x: sum(x) / len(x), n_resamples=6000,
                  alternative="greater")
print(f"\nbijection - freq-rank  mean {statistics.mean(d):+.4f} nats  "
      f"95% BCa [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")
print(f"  {'the frequency-ordered code is better' if r.ci_low > 0 else 'not distinguishable from zero'}")
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {"rule": a.rule, "lr": a.lr, "res": res})
print(f"wrote {a.out}")
