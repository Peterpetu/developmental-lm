"""FRONTIER: `d_z` is set by measurement, not by choice.

SPEC: 06 §6.5, SPEC: 09 §9.1

`d_z` is the capacity dial and the project's central empirical object. Larger
`d_z` gives the soma more room, which buys language quality and simultaneously
lets the soma wash out the differences between genomes -- driving `h2`, and
therefore evolvability, toward zero. The two effects trade off monotonically.

**The rule is fixed before the sweep runs and is executed by this script, not by
a reader looking at the table**: the operating point is the **largest** `d_z`
whose `h2` lower confidence bound still clears `KG_HERIT_LOWER`. Reporting the
whole frontier is required whatever the outcome; selecting the operating point
after seeing it is legitimate precisely because the rule was fixed first.

`d_z = 0` is a real configuration and the purest form of the architecture:
identity is then entirely innate, the soma holds only `y`, and almost nothing is
learned within a lifetime. It is the left endpoint and it is expected to be a
poor language model. Reporting it is not optional.

Four quantities per point, exactly §6.5's list:

``L*``            mean held-out CE after a full lifetime, on the fitness split.
``h2``            §6.2's ANOVA estimate with its 95% BCa interval.
``share``         §6.3's co-adapted randomise-and-measure probe.
``birth_share``   §6.4's at-birth quantity, against `L_null = ln V`.

`share_retrain` (§6.3.1) is **not** among them and that is not an oversight:
this is a generation-0 measurement, where the evolved genome *is* a random
genome and the statistic is identically zero by construction. §6.1 is the
reason, and computing it here would produce a number that means nothing.

    python scripts/gnomon_frontier.py --config m-nano --genomes 12 --seeds 3
"""
import argparse, dataclasses, json, math, pathlib, statistics, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.eval.share import (KG_HERIT_LOWER, KG_HERIT_POINT, birth_share,
                               genome_share, heritability, predicted_response)
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner

# SPEC: 06 §6.5 -- fixed before the sweep. `0` first, so a crash mid-sweep still
# leaves the endpoint the section calls non-optional.
D_Z_GRID = (0, 2, 4, 8, 16, 32, 64)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m-nano")
ap.add_argument("--genomes", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=None,
                help="override N_life; the horizon is stated in the output")
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--d-z", type=int, nargs="*", default=list(D_Z_GRID))
ap.add_argument("--rule", default="adam",
                choices=["adam", "eta", "dpr", "dpr-frozen"])
ap.add_argument("--token-code", default="freq", choices=["freq", "bijection"])
ap.add_argument("--n-probe", type=int, default=5,
                help="draws per side of §6.3's probe (the section says 5)")
ap.add_argument("--out", default="runs/gnomon/frontier.json")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
base = get_config(a.config)
over = {k: v for k, v in (("N_life", a.steps), ("batch", a.batch), ("lr", a.lr))
        if v is not None}
cfg = dataclasses.replace(base, **over)
root = f"data/ts_{cfg.vocab}"
tr, fi = Shard.load(root, "train"), Shard.load(root, "fitness")
L_null = math.log(cfg.vocab)

print(f"FRONTIER  {cfg.name}  N_life {cfg.N_life}  batch {cfg.batch}  "
      f"lr {cfg.lr}  rule {a.rule}  token code {a.token_code}")
# SPEC: 10 §10.6 -- a rate is only meaningful with the horizon it was measured
# at, so the run says which it has. An unmeasured one is loud about it: that is
# the whole point of recording provenance rather than only the number.
print(f"  lr provenance: {cfg.lr_provenance}")
if cfg.lr_provenance.startswith("NOT MEASURED"):
    print(f"  *** this run uses an UNMEASURED learning rate. §10.6 records "
          f"using one as a defect in every comparison it touches. Pass --lr "
          f"with a value from scripts/gnomon_lrsweep.py, or accept that the "
          f"result is reported as being at an untuned rate.")
print(f"d_z in {a.d_z}   {a.genomes} genomes x {a.seeds} seeds = "
      f"{a.genomes * a.seeds} lifetimes per point, "
      f"{len(a.d_z) * a.genomes * a.seeds} in all")
print(f"L_null = ln V = {L_null:.4f}\n")

t0 = time.time()
points = []
for d_z in a.d_z:
    shape = GenomeShape(d_z=d_z)
    runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=fi,
                            device=a.device, compile_mode=None,
                            n_eval_batches=8, token_code=a.token_code)
    # `P[g][s]`: the fully crossed design §6.2 requires. Genomes are the same
    # draws at every `d_z` (`seed=5000+g`) and so are the lifetime seeds, so a
    # difference between two points on the frontier is `d_z` and nothing else.
    P, births = [], []
    for g in range(a.genomes):
        row = []
        for s in range(a.seeds):
            r = runner.run(Decoder(shape, seed=5000 + g), seed=900 + s,
                           rule=a.rule)
            row.append(r.final)
            births.append(r.birth)
        P.append(row)
        print(f"  d_z {d_z:3d}  genome {g + 1:2d}/{a.genomes}  "
              f"{statistics.mean(row):.4f}   ({time.time() - t0:6.0f}s)",
              flush=True)

    h = heritability(P)
    L_star = statistics.mean(x for row in P for x in row)
    L_birth = statistics.mean(births)

    # SPEC: 06 §6.3 -- the co-adapted probe, on the *last* lifetime's state:
    # `Z*` is still in the runner and `Gamma*` is the genome it was trained
    # against, which is exactly the pair the probe is defined over.
    gamma_star = runner.export_genome()
    L_gamma_rand = []
    for k in range(a.n_probe):
        runner.load_genome(Decoder(shape, seed=90_000 + k).state_dict())
        L_gamma_rand.append(runner.evaluate())
    runner.load_genome(gamma_star)
    L_z_rand = []
    for k in range(a.n_probe):
        runner.reset_soma(seed=80_000 + k)
        L_z_rand.append(runner.evaluate())

    probe = genome_share(L_star, L_gamma_rand, L_z_rand)
    bs = birth_share(L_birth, L_null, L_star)
    points.append({
        "d_z": d_z, "L_star": L_star, "L_birth": L_birth,
        "h2": h.h2, "ci_low": h.ci_low, "ci_high": h.ci_high,
        "V_G": h.V_G, "V_E": h.V_E, "sigma_P": h.sigma_P,
        "R_predicted": predicted_response(h),
        "share": probe["share"], "birth_share": bs,
        "passes_KG_HERIT": h.passes, "P": P,
        "n_genome": runner.model.n_genome(), "n_soma": runner.model.n_soma(),
    })
    print(f"  d_z {d_z:3d}  L* {L_star:.4f}  h2 {h.h2:.3f} "
          f"[{h.ci_low:.3f}, {h.ci_high:.3f}]  share {probe['share']:+.3f}  "
          f"birth_share {bs:+.3f}\n", flush=True)

print(f"\n{'d_z':>5s} {'L*':>8s} {'h2':>7s} {'95% BCa':>18s} {'sigma_P':>8s} "
      f"{'share':>7s} {'birth':>7s} {'|Z|':>9s}  KG-HERIT")
for p in points:
    print(f"{p['d_z']:5d} {p['L_star']:8.4f} {p['h2']:7.3f} "
          f"[{p['ci_low']:6.3f}, {p['ci_high']:6.3f}] {p['sigma_P']:8.4f} "
          f"{p['share']:+7.3f} {p['birth_share']:+7.3f} {p['n_soma']:9,d}  "
          f"{'PASS' if p['passes_KG_HERIT'] else 'FAIL'}")

# SPEC: 06 §6.5 -- the rule, applied by the script rather than by a reader.
admissible = [p for p in points if p["ci_low"] >= KG_HERIT_LOWER]
print(f"\nthe rule, fixed before the sweep: the operating point is the LARGEST "
      f"d_z whose h2 lower bound clears {KG_HERIT_LOWER}")
if admissible:
    chosen = max(admissible, key=lambda p: p["d_z"])
    print(f"  admissible: d_z in {[p['d_z'] for p in admissible]}")
    print(f"  OPERATING POINT: d_z = {chosen['d_z']}  "
          f"(h2 {chosen['h2']:.3f}, lower bound {chosen['ci_low']:.3f}, "
          f"L* {chosen['L_star']:.4f})")
else:
    chosen = None
    print(f"  *** NO d_z IS ADMISSIBLE. Every point's h2 lower bound is below "
          f"{KG_HERIT_LOWER}, so `KG-HERIT` fires across the whole frontier and "
          f"§06.2's consequence applies: the constitution does not hold "
          f"empirically at this configuration. Reported as measured.")

# The trade-off §6.5 calls the project's central empirical object, made visible
# rather than asserted: quality should improve with d_z and h2 should fall.
if len(points) > 1:
    q = points[0]["L_star"] - points[-1]["L_star"]
    e = points[0]["h2"] - points[-1]["h2"]
    print(f"\nacross the frontier: L* moves {q:+.4f} nats "
          f"({'better' if q > 0 else 'worse'} at large d_z), "
          f"h2 moves {-e:+.3f}")

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
write_json(a.out, {
    "config": cfg.name, "N_life": cfg.N_life, "batch": cfg.batch, "lr": cfg.lr,
    "rule": a.rule, "token_code": a.token_code, "L_null": L_null,
    "n_genomes": a.genomes, "n_seeds": a.seeds,
    "KG_HERIT_POINT": KG_HERIT_POINT, "KG_HERIT_LOWER": KG_HERIT_LOWER,
    "points": points,
    "operating_point": chosen["d_z"] if chosen else None,
    "seconds": time.time() - t0,
})
print(f"\nwrote {a.out}   ({time.time() - t0:.0f}s)")
