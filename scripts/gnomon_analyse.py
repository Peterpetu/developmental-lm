"""The kill criteria of §09.2, evaluated from a completed run.

SPEC: 07 §7.4, SPEC: 08 §8.4, SPEC: 09 §9.2

``KG-FLAT``   no significant downward trend in `best` across generations.
``KG-QD``     MAP-Elites indistinguishable from random search (TOST).
``KG-RANK``   the archive's elites show decoded effective rank collapsing.
``KG-SHARE``  `share_retrain` below 0.50 (§01.6), when the run recorded it.

`KG-HERIT` lives in `gnomon_herit.py` because §06.2 puts it on the smoke run,
before the machinery this script analyses exists; `KG-SCALE` in
`gnomon_transfer.py`; `KG-NOFREE` is `G-NOFREE`, a test rather than a run.

    python scripts/gnomon_analyse.py runs/gnomon/evo-GNOMON-s*/history.json \
        --against runs/gnomon/evo-A-RAND-s*/history.json
"""
import argparse, glob, json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from gnomon.eval.share import KG_SHARE_FLOOR, share_retrain
from gnomon.utils.stats import (bca_bootstrap, benjamini_hochberg, mann_kendall,
                                tost)

ap = argparse.ArgumentParser()
ap.add_argument("history", nargs="+")
ap.add_argument("--against", nargs="*", default=[])
ap.add_argument("--margin", type=float, default=0.05,
                help="TOST equivalence margin in nats (SPEC: 09 §9.2)")
a = ap.parse_args()

def load(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            out.append((p, json.loads(pathlib.Path(p).read_text())))
    return out

runs = load(a.history)
ctrl = load(a.against)
if not runs:
    sys.exit(f"no histories matched {a.history}")

print(f"{'run':46s} {'gens':>5s} {'best@0':>8s} {'best@end':>9s} "
      f"{'delta':>8s} {'MK p':>8s} {'cov':>5s} {'QD':>8s}")
p_values, deltas, finals = [], [], []
for path, h in runs:
    best = [r["best"] for r in h]
    mk = mann_kendall(best, alternative="decreasing")     # a *decreasing* loss
    d = best[-1] - best[0]
    p_values.append(mk.p_value); deltas.append(d); finals.append(best[-1])
    print(f"{path[:46]:46s} {len(h)-1:5d} {best[0]:8.4f} {best[-1]:9.4f} "
          f"{d:+8.4f} {mk.p_value:8.4f} {h[-1]['coverage']:5.2f} "
          f"{h[-1]['qd_score']:8.2f}")

print()
if len(p_values) > 1:
    rej, adj = benjamini_hochberg(p_values, alpha=0.05)
    print(f"Mann-Kendall over {len(p_values)} run-seeds, BH-corrected: "
          f"{sum(rej)}/{len(rej)} significant   adj p = "
          f"{[f'{v:.4f}' for v in adj]}")
    flat = not any(rej)
else:
    flat = p_values[0] >= 0.05
    print(f"single run-seed: Mann-Kendall p = {p_values[0]:.4f} "
          f"(§09.6 rule 3: one seed is stated as a limitation)")
# SPEC: 09 §9.2. `KG-FLAT` is defined over the *completed* run with three
# run-seeds. Declaring it on a partial or single-seed history would be a verdict
# the design does not license -- and `best` is a step function, so Mann-Kendall
# has little power on it until the run is long: at generation 6 of 20 it stood at
# p = 0.11 while `mean_elite` was already at p = 0.0002.
n_seeds, min_gens = len(runs), min(len(h) - 1 for _, h in runs)
if n_seeds < 3 or min_gens < 20:
    print(f"KG-FLAT: NOT EVALUABLE -- needs 3 run-seeds x 20 generations, "
          f"have {n_seeds} x {min_gens}")
    print(f"  provisional: mean delta {np.mean(deltas):+.4f} nats on `best`; "
          f"see the per-series table below, which is the informative one")
else:
    print(f"KG-FLAT: {'FIRES -- evolution is inert' if flat else 'not fired'}"
          f"   (mean delta {np.mean(deltas):+.4f} nats over the run)")

print()
print(f"{'series':12s} {'gen0':>9s} {'now':>9s} {'delta':>9s} {'MK p':>8s} {'per gen':>9s}")
for key, alt in (("best", "decreasing"), ("mean_elite", "decreasing"),
                 ("qd_score", "increasing"), ("coverage", "increasing"),
                 ("mean_erank", "decreasing"), ("mean_birth", "decreasing")):
    for path, h in runs:
        v = [r.get(key) for r in h]
        if any(x is None for x in v):
            # SPEC: 07 §7.4 -- a run written before v1.1 has no `erank` series
            # at all, because `Member.erank` was computed and never recorded.
            # Say so rather than reporting a criterion over absent data.
            print(f"{key:12s}      -- not recorded in {pathlib.Path(path).parent.name}")
            continue
        mk = mann_kendall(v, alternative=alt)
        print(f"{key:12s} {v[0]:9.4f} {v[-1]:9.4f} {v[-1]-v[0]:+9.4f} "
              f"{mk.p_value:8.4f} {(v[-1]-v[0])/max(len(v)-1,1):+9.4f}")

# ----------------------------------------------------------------- KG-RANK
# SPEC: 07 §7.4, SPEC: 09 §9.2. The decoder degenerating is a different failure
# from evolution being inert: `best` can improve while the decoded weights
# collapse toward rank 2, which is §03.1's additive regime and the thing the
# two-hidden-layer decoder exists to avoid. It is a *decreasing* trend test on
# the archive's elites, over the same generations.
print()
rank_p = []
for path, h in runs:
    v = [r.get("mean_erank") for r in h]
    if any(x is None for x in v):
        print(f"KG-RANK: NOT EVALUABLE for {path} -- no `mean_erank` series "
              f"(run predates v1.1; re-run to record it)")
        continue
    mk = mann_kendall(v, alternative="decreasing")
    rank_p.append(mk.p_value)
    print(f"KG-RANK  {pathlib.Path(path).parent.name:28s} erank/min(d) "
          f"{v[0]:.4f} -> {v[-1]:.4f}   MK p {mk.p_value:.4f}")
if rank_p:
    rej, adj = (benjamini_hochberg(rank_p, alpha=0.05) if len(rank_p) > 1
                else ([rank_p[0] < 0.05], rank_p))
    fires = any(rej)
    print(f"KG-RANK: {'FIRES -- the decoder is degenerating' if fires else 'not fired'}"
          f"   ({sum(rej)}/{len(rej)} run-seeds show a significant decline)")

# ---------------------------------------------------------------- KG-SHARE
# SPEC: 01 §1.6, SPEC: 06 §6.3.1. Reported when the run recorded the three
# losses it needs; `scripts/gnomon_frontier.py` and `gnomon_herit.py` write
# them. A run that did not is reported as not evaluable, never as passing.
print()
shares = []
for path, h in runs:
    side = pathlib.Path(path).with_name("share.json")
    if not side.exists():
        continue
    d = json.loads(side.read_text())
    r = share_retrain(d["L_star"], d["L_random_trained"], d["L_birth"])
    shares.append(r["share_retrain"])
    print(f"KG-SHARE {pathlib.Path(path).parent.name:28s} "
          f"share_retrain {r['share_retrain']:+.4f}   {r['KG_SHARE']}")
if not shares:
    print(f"KG-SHARE: NOT EVALUABLE -- no `share.json` beside these histories "
          f"(floor {KG_SHARE_FLOOR}); run scripts/gnomon_frontier.py to write one")

# The registered prediction, §09.3.
R_pred = 1.271 * 0.398 * 0.446
gens = max(len(h) - 1 for _, h in runs)
obs = -np.mean(deltas) / max(gens, 1)
print(f"\nregistered upper bound  R = i h2 sigma_P = {R_pred:.4f} nats/generation")
print(f"observed mean response                    = {obs:.4f} nats/generation")
print(f"  -> {'within' if obs <= R_pred else 'EXCEEDS'} the bound"
      f"{'' if obs <= R_pred else ' -- diagnose a leak (§09.3)'}")

if ctrl:
    print()
    cf = [h[-1]["best"] for _, h in ctrl]
    cq = [h[-1]["qd_score"] for _, h in ctrl]
    qf = [h[-1]["qd_score"] for _, h in runs]
    print(f"GNOMON  final best {np.mean(finals):.4f} (n={len(finals)})   "
          f"QD {np.mean(qf):.2f}")
    print(f"A-RAND  final best {np.mean(cf):.4f} (n={len(cf)})   "
          f"QD {np.mean(cq):.2f}")
    print(f"  delta best {np.mean(cf) - np.mean(finals):+.4f} nats "
          f"(positive = MAP-Elites wins)")
    if len(finals) >= 2 and len(cf) >= 2:
        r = tost(finals, cf, a.margin, paired=False)
        print(f"  TOST at margin {a.margin}: "
              f"{'EQUIVALENT -- KG-QD FIRES' if r.equivalent else 'not equivalent'}"
              f"   90% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")
    else:
        print("  TOST needs >= 2 seeds per arm; reported as a point comparison "
              "only, and §09.6 rule 3 requires that to be stated")
