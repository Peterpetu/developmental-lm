"""Smoke run: does a genomically-bottlenecked model train at all?

SPEC: 05 §5.1 -- one lifetime at M-pico, reported honestly.
"""
import argparse, dataclasses, math, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.train import lifetime

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--genome-seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
# SPEC: 10 §10.6 -- the four lifetime arms. `--no-eta` was v0.9's switch and
# was still being passed to a `lifetime()` that had not accepted it since
# v1.0, so this script raised `TypeError` on every invocation.
ap.add_argument("--rule", default="dpr",
                choices=["adam", "eta", "dpr", "dpr-frozen"])
ap.add_argument("--token-code", default="freq", choices=["freq", "bijection"])
ap.add_argument("--no-anneal", action="store_true")
a = ap.parse_args()

cfg = get_config(a.config)
if a.steps:
    cfg = dataclasses.replace(cfg, N_life=a.steps)
shape = GenomeShape(d_z=a.d_z)
root = f"data/ts_{cfg.vocab}"
train, fit = Shard.load(root, "train"), Shard.load(root, "fitness")
dec = Decoder(shape, seed=a.genome_seed).to(a.device)

print(f"config {cfg.name}  d_z {a.d_z}  device {a.device}  steps {cfg.N_life}  "
      f"rule {a.rule}  token code {a.token_code} (SPEC: 02 §2.9)")
print(f"train shard {train.n_tokens:,} tokens | fitness shard {fit.n_tokens:,}")
r = lifetime(dec, cfg, shape, seed=a.seed, train_shard=train, eval_shard=fit,
             device=a.device, progress=True, rule=a.rule,
             token_code=a.token_code, anneal=not a.no_anneal)
print(f"\n  genome {r.n_genome:,} | soma {r.n_soma:,} | decoded {r.n_decoded:,} "
      f"| compression {r.n_decoded/r.n_genome:,.0f}x")
print(f"  ln V   {math.log(cfg.vocab):.4f}")
print(f"  birth  {r.birth:.4f}")
print(f"  final  {r.final:.4f}   (learned {r.birth - r.final:+.4f} nats)")
print(f"  late slope {r.slope:.4f} nats / decade")
print(f"  {r.seconds:.1f}s  ({r.seconds/cfg.N_life*1000:.1f} ms/step)")
