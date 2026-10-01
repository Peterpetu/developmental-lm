"""KG-HERIT: does the genome carry the function?

SPEC: 06 §6.2 -- the gate that fires on the smoke run, before any evolutionary
machinery is built. If it fires, the constitution does not hold and the
architecture is redesigned.
"""
import argparse, dataclasses, json, math, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.eval.share import birth_share, genome_share, heritability
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--genomes", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--no-eta", action="store_true")
ap.add_argument("--no-anneal", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
cfg = get_config(a.config)
if a.steps:
    cfg = dataclasses.replace(cfg, N_life=a.steps)
shape = GenomeShape(d_z=a.d_z)
root = f"data/ts_{cfg.vocab}"
tr, fi = Shard.load(root, "train"), Shard.load(root, "fitness")
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=fi, device=a.device)

print(f"KG-HERIT  config {cfg.name}  d_z {a.d_z}  N_life {cfg.N_life}  "
      f"{a.genomes} genomes x {a.seeds} seeds = {a.genomes*a.seeds} lifetimes")
print(f"  genome {shape.n_params():,} reals | decoded {cfg.n_decoded():,} | "
      f"compression {cfg.n_decoded()/shape.n_params():,.0f}x")

P, births, t0, rows = [], [], time.time(), []
for g in range(a.genomes):
    dec = Decoder(shape, seed=1000 + g)
    row, brow = [], []
    for s in range(a.seeds):
        r = runner.run(dec, seed=200 + s, use_eta=not a.no_eta,
                       anneal=not a.no_anneal)
        row.append(r.final); brow.append(r.birth)
        rows.append({"genome": g, "seed": s, "final": r.final, "birth": r.birth,
                     "slope": r.slope, "seconds": r.seconds})
    P.append(row); births.append(brow)
    print(f"  genome {g:2d}  final {['%.4f' % v for v in row]}  "
          f"mean {sum(row)/len(row):.4f}   ({time.time()-t0:5.0f}s)")

h = heritability(P, n_resamples=4000)
print("\n" + h.report())

flat = [v for row in P for v in row]
print(f"\nbest genome mean {min(h.means):.4f} | worst {max(h.means):.4f} | "
      f"spread {max(h.means)-min(h.means):.4f} nats")
print(f"within-genome sd (seed noise) {math.sqrt(h.V_E):.4f} nats")
print(f"between-genome sd            {math.sqrt(h.V_G):.4f} nats")
L_null = sum(b for br in births for b in br) / len(flat)
print(f"\nbirth CE (mean over all)     {L_null:.4f}   ln V = {math.log(cfg.vocab):.4f}")
print(f"birth_share at generation 0  "
      f"{birth_share(L_null, math.log(cfg.vocab), sum(flat)/len(flat)):+.4f}"
      f"   (expected ~0: a random genome knows nothing)")

if a.out:
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, {
        "config": cfg.name, "d_z": a.d_z, "N_life": cfg.N_life,
        "n_genomes": a.genomes, "n_seeds": a.seeds,
        "use_eta": not a.no_eta, "anneal": not a.no_anneal,
        "h2": h.h2, "ci_low": h.ci_low, "ci_high": h.ci_high,
        "V_G": h.V_G, "V_E": h.V_E, "sigma_P": h.sigma_P,
        "passes": h.passes, "P": P, "births": births, "rows": rows,
    })
    print(f"\nwrote {a.out}")
print(f"\ntotal {time.time()-t0:.0f}s")
