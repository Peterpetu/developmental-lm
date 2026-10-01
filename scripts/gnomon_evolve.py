"""Run one evolutionary loop. SPEC: 08 §8.2.

    python scripts/gnomon_evolve.py --gens 40 --pop 24 --seed 0
    python scripts/gnomon_evolve.py --arm A-RAND      # the matched control
"""
import argparse, dataclasses, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.qd.evolve import EvolveConfig, evolve

ARMS = {
    "GNOMON":   {},
    "A-RAND":   {"random_search": True},
    "A-NOARCH": {"freeze_arch": True},
    "A-NOX":    {"no_crossover": True},
    "A-1D":     {"one_descriptor": True},
    "R-ADAM":   {"rule": "adam"},
    "R-ETA":    {"rule": "eta"},
    "R-DPR-FROZEN": {"rule": "dpr-frozen"},
    "A-ANNEAL": {"anneal": False},
    "A-FIXSIG": {"fixed_sigma": True},
    # SPEC: 08 §8.5 -- the paper's own algorithm, reported as an upper bound and
    # never as GNOMON's result. Its artifacts are labelled `A-LAMARCK`.
    "A-LAMARCK": {"lamarck": True},
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-pico")
ap.add_argument("--arm", default="GNOMON", choices=sorted(ARMS))
ap.add_argument("--gens", type=int, default=40)
ap.add_argument("--pop", type=int, default=24)
ap.add_argument("--init", type=int, default=32)
ap.add_argument("--n-seed", type=int, default=2)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch", type=int, default=128)
# SPEC: 10 §10.6 -- measured at *this* horizon, which is not the
# configuration's own: §08.3 sets the fitness horizon to N_life=150 at
# batch 128, so `M_PICO.lr` (measured at 600/32) is the wrong value here.
# runs/gnomon/lrsweep-m-pico-n150-b128.json: 0.09 and 0.12 are
# indistinguishable, the curve turns inside the grid, and 0.003 -- what
# the completed EVO run used -- is worse than 0.015 by +0.4262 nats.
ap.add_argument("--lr", type=float, default=0.09)
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
# SPEC: 02 §2.9 -- `bijection` is the control the section measured against,
# not a tuning choice. It exists so the comparison can be re-run.
ap.add_argument("--token-code", default="freq", choices=["freq", "bijection"])
# SPEC: 07 §7.1 -- `cudagraph` captures the step and replays it as one
# launch: 1.60x measured at M-pico, and ~1 s to bind per architecture
# against `torch.compile`'s ~30 s. It is opt-in because it also changes
# which CUDA kernels run, so a run under it is bit-reproducible against
# itself and not against an eager one (gnomon/lifetime/graph.py).
ap.add_argument("--compile-mode", default=None,
                choices=[None, "cudagraph", "reduce-overhead", "default"])
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
cfg = dataclasses.replace(
    get_config(a.config), N_life=a.steps, batch=a.batch, lr=a.lr,
    lr_provenance=(f"measured at N_life=150, batch=128 "
                   f"(runs/gnomon/lrsweep-m-pico-n150-b128.json); "
                   f"this run is at N_life={a.steps}, batch={a.batch}")
    if (a.steps, a.batch) == (150, 128) else
    f"SET ON THE COMMAND LINE to {a.lr}; not measured at "
    f"N_life={a.steps}, batch={a.batch}")
shape = GenomeShape(d_z=a.d_z)
out = a.out or f"runs/gnomon/evo-{a.arm}-s{a.seed}"
pathlib.Path(out).mkdir(parents=True, exist_ok=True)

ev = EvolveConfig(n_init=a.init, n_pop=a.pop, n_seed=a.n_seed, G_gen=a.gens,
                  seed=a.seed, **ARMS[a.arm])
root = f"data/ts_{cfg.vocab}"
tr, fi = Shard.load(root, "train"), Shard.load(root, "fitness")

write_json(pathlib.Path(out, "config.json"), {
    "arm": a.arm, "config": dataclasses.asdict(cfg),
    "shape": dataclasses.asdict(shape), "evolve": dataclasses.asdict(ev),
    "token_code": a.token_code, "compile_mode": a.compile_mode,
    "lr": cfg.lr, "lr_provenance": cfg.lr_provenance,
})

archive = evolve(cfg, shape, ev, train_shard=tr, eval_shard=fi,
                 device=a.device, out_dir=out, token_code=a.token_code,
                 compile_mode=a.compile_mode,
                 log=lambda s: print(s, flush=True))
print(f"\nwrote {out}/archive.pt and {out}/history.json")
