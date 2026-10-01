"""KG-SCALE: does a genome evolved small decode usefully large?

SPEC: 02 §2.7 -- the gnomon claim. Because `|Gamma|` depends on the code widths
and not on `d_model`, `L` or `V`, one genome decodes phenotypes of any size up
to the addressable maximum with no change to its shape. Whether the result is
*better than a random genome of the same shape* is the empirical question, and
this script is the only thing that can answer it.

Paired by lifetime seed: the transferred and random genomes see identical soma
initialisations and identical data order, so the difference is the genome.
"""
import argparse, dataclasses, json, math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.genome.arch import ArchHead
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner
from gnomon.qd.archive import Archive
from gnomon.utils.stats import bca_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--archive", required=True, help="an evolutionary run's archive.pt")
ap.add_argument("--to", default="m-nano", help="target configuration")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = get_config(a.to)
if a.steps: cfg = dataclasses.replace(cfg, N_life=a.steps)
if a.batch: cfg = dataclasses.replace(cfg, batch=a.batch)
shape = GenomeShape(d_z=a.d_z)
arch_store = Archive.load(a.archive)
elite = arch_store.best()
print(f"transferring the elite of {a.archive}")
print(f"  fitness at source {elite.fitness:.4f}  arch [{elite.arch_summary}]  "
      f"generation {elite.generation}")
print(f"  target {cfg.name}: d_model {cfg.d_model} L {cfg.L} vocab {cfg.vocab} "
      f"N_life {cfg.N_life} batch {cfg.batch}")

root = f"data/ts_{cfg.vocab}"
tr, te = Shard.load(root, "train"), Shard.load(root, "test")
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)

def arms():
    """The transferred genome, and a random one of identical shape."""
    w = {k[2:]: v for k, v in elite.state.items() if k.startswith("w.")}
    ah = ArchHead(shape); ah.load_state_dict(
        {k[2:]: v for k, v in elite.state.items() if k.startswith("a.")})
    yield "transferred", w, ah.decode_arch(cfg)
    for j in range(3):
        rw = Decoder(shape, seed=90210 + j).state_dict()
        rah = ArchHead(shape, seed=90210 + j)
        yield f"random-{j}", rw, rah.decode_arch(cfg)
    # A random genome forced onto the *elite's* architecture, so the comparison
    # separates "the genome transferred" from "the architecture transferred".
    yield "random-on-elite-arch", Decoder(shape, seed=555).state_dict(), \
          ah.decode_arch(cfg)

rows, t0 = [], time.time()
for name, w, arch in arms():
    runner.set_arch(arch)
    fin, birth = [], []
    for s in range(a.seeds):
        r = runner.run(w, seed=4000 + s)
        fin.append(r.final); birth.append(r.birth)
    rows.append({"arm": name, "arch": arch.summary(), "final": fin,
                 "birth": birth, "mean_final": sum(fin)/len(fin),
                 "mean_birth": sum(birth)/len(birth),
                 "n_decoded": arch.n_decoded(cfg)})
    print(f"  {name:22s} birth {rows[-1]['mean_birth']:.4f}  "
          f"final {rows[-1]['mean_final']:.4f}  "
          f"[{' '.join(f'{v:.3f}' for v in fin)}]  ({time.time()-t0:.0f}s)")

tx = rows[0]
rand = [r for r in rows if r["arm"].startswith("random-") and r["arm"] != "random-on-elite-arch"]
pooled = [v for r in rand for v in r["final"]]
delta = sum(pooled)/len(pooled) - tx["mean_final"]
same_arch = [r for r in rows if r["arm"] == "random-on-elite-arch"][0]
print(f"\ntransferred {tx['mean_final']:.4f} vs random genomes "
      f"{sum(pooled)/len(pooled):.4f}   delta {delta:+.4f} nats")
print(f"transferred {tx['mean_final']:.4f} vs a random genome on the *same* "
      f"architecture {same_arch['mean_final']:.4f}   "
      f"delta {same_arch['mean_final'] - tx['mean_final']:+.4f} nats")
paired = [b - t for b, t in zip(same_arch["final"], tx["final"])]
if len(paired) >= 5:
    r = bca_bootstrap(paired, lambda v: sum(v)/len(v), n_resamples=4000,
                      alternative="greater")
    print(f"paired (same architecture, same seeds): mean {sum(paired)/len(paired):+.4f} "
          f"95% BCa [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")
    print(f"KG-SCALE: {'the genome transfers' if r.ci_low > 0 else 'FIRES -- no transfer'}")
out = a.out or f"runs/gnomon/transfer-{a.to}.json"
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
write_json(out, {"source": a.archive, "target": cfg.name, "rows": rows})
print(f"wrote {out}")
