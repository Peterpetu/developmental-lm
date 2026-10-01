#!/usr/bin/env python
"""Evaluate the gates and kill criteria of docs/anlage/08 §8.3 from artifacts on disk.

SPEC: 08 §8.3
SPEC: 08 §8.6

    python scripts/anlage_analyse.py [--runs runs/anlage]

Every criterion prints one of PASS / FIRES / INCONCLUSIVE / NOT EVALUABLE. A
criterion whose run does not exist is NOT EVALUABLE, never a pass. Paired
contrasts use the seeds the runs share; `sigma_e` comes from the R1 file of
the configuration and is never inherited from another.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gnomon.utils.stats import bca_bootstrap, tost  # noqa: E402


def _load(p: pathlib.Path):
    return json.loads(p.read_text()) if p.exists() else None


def paired(a: list[float], b: list[float]) -> dict:
    """`b - a` paired over seeds. SPEC: 08 §8.2.

    BCa over seeds at `n >= 5` (the vendored bootstrap's own floor); a paired
    Student-t 95% interval at `3 <= n < 5`, labelled; the mean alone below.
    """
    d = [y - x for x, y in zip(a, b)]
    out = {"n": len(d), "mean": statistics.fmean(d) if d else None, "method": None}
    if len(d) >= 5:
        res = bca_bootstrap(d, statistics.fmean, n_resamples=4000, alternative="two-sided")
        out.update({"ci_low": res.ci_low, "ci_high": res.ci_high, "method": "bca"})
    elif len(d) >= 3:
        from scipy import stats as _st
        m, s = statistics.fmean(d), statistics.stdev(d)
        h = _st.t.ppf(0.975, len(d) - 1) * s / len(d) ** 0.5
        out.update({"ci_low": m - h, "ci_high": m + h, "method": "t"})
    return out


def verdict_better(contrast: dict, sigma_e: float, k: float = 3.0) -> str:
    """'b better than a by >= k sigma_e' means mean(b - a) <= -k sigma_e and the CI excludes 0."""
    if contrast["mean"] is None:
        return "NOT EVALUABLE"
    if contrast["n"] < 3:
        return ("FIRES?" if contrast["mean"] > -k * sigma_e else "PASS?") + " (n<3, no interval)"
    excludes_zero = contrast["ci_high"] < 0 or contrast["ci_low"] > 0
    if contrast["mean"] <= -k * sigma_e and excludes_zero:
        return "PASS"
    if contrast["mean"] > -k * sigma_e and (contrast["ci_low"] > -k * sigma_e):
        return "FIRES"
    return "INCONCLUSIVE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=str(REPO / "runs" / "anlage"))
    ap.add_argument("--lineage", default="tinystories")
    args = ap.parse_args()
    runs = pathlib.Path(args.runs)
    lin = args.lineage
    lines = []

    noise = _load(runs / f"noise-{lin}-s0" / "noise.json")
    if noise is None:
        lines.append("G-NOISE      NOT EVALUABLE -- runs/anlage/noise-<lineage>-s0/noise.json absent; "
                     "nothing at this configuration may be reported")
        print("\n".join(lines))
        return 1
    sigma_e = float(noise["sigma_e"])
    lines.append(f"G-NOISE      PASS  sigma_e = {sigma_e:.4f} nats ({noise['sigma_e_df']} df), "
                 f"mean test CE {noise['mean_test_ce']:.4f} over {noise['seeds']} seeds")

    # K-BIRTH / K-SCALE from the latest birth report
    births = sorted(runs.glob(f"birth-{lin}-s0-*/birth.json"))
    if not births:
        lines.append("K-BIRTH      NOT EVALUABLE -- no birth report")
        lines.append("K-SCALE      NOT EVALUABLE -- no birth report")
    else:
        b = _load(births[-1])
        res_ok = all(v < 0.5 for v in b["regress"]["residual"].values())
        share = b["birth_share"]
        v = ("PASS" if share >= 0.5 else ("FIRES" if res_ok else "INCONCLUSIVE (residual > 0.5 on some role)"))
        lines.append(f"K-BIRTH      {v}  birth_share {share:.3f}  at-birth {b['birth_ce']:.4f}  "
                     f"adult {b['adult_ce']:.4f}  shuffle {b['shuffle_ce']:.4f}  random genome "
                     f"{b['random_genome_ce']:.4f}  mean residual {b['regress']['residual_mean']:.3f}  "
                     f"[{births[-1].parent.name}]")
        sc = b.get("scale")
        if sc:
            d = sc["grown_genome_ce"] - sc["grown_random_ce"]
            v = "PASS" if d <= -3 * sigma_e else ("FIRES" if d > -3 * sigma_e else "INCONCLUSIVE")
            lines.append(f"K-SCALE      {v}  grown genome {sc['grown_genome_ce']:.4f} vs random "
                         f"{sc['grown_random_ce']:.4f} vs copy {sc['grown_copy_ce']:.4f} (one seed, no gates)")
        else:
            lines.append("K-SCALE      NOT EVALUABLE -- no grown decode in the birth report")

    # K-GROW
    grows = sorted(runs.glob(f"grow-{lin}-s0to1*/grow.json"))
    if not grows:
        lines.append("K-GROW       NOT EVALUABLE -- no growth run")
    else:
        g = _load(grows[-1])
        r = g["results"]
        treat = "prior" if any(k.startswith("prior/") for k in r) else "genome"
        seeds = [s for s in g["seeds"] if all(f"{a}/seed{s}" in r for a in (treat, "copy"))]
        gen = [r[f"{treat}/seed{s}"]["final_ce"] for s in seeds]
        cp = [r[f"copy/seed{s}"]["final_ce"] for s in seeds]
        c = paired(cp, gen)
        v = verdict_better(c, sigma_e)
        lines.append(f"K-GROW       {v}  {treat} - copy = {c['mean']:+.4f} over {c['n']} seeds"
                     + (f"  {c['method']} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]" if "ci_low" in c else "")
                     + f"  (fires unless {treat} better by 3 sigma_e = {3 * sigma_e:.4f})  [{grows[-1].parent.name}]")
        if all(f"flat/seed{s}" in r for s in seeds) and seeds:
            fl = [r[f"flat/seed{s}"]["final_ce"] for s in seeds]
            c2 = paired(fl, cp)
            lines.append(f"H-GROW(2)    {verdict_better(c2, sigma_e, k=0.0)}  copy - flat = {c2['mean']:+.4f}"
                         + (f"  {c2['method']} [{c2['ci_low']:+.4f}, {c2['ci_high']:+.4f}]" if "ci_low" in c2 else ""))
        for arm in g["arms"]:
            vals = g["summary"][arm]["final_ce"]
            lines.append(f"   {arm:8s} {statistics.fmean(vals):.4f}  {['%.4f' % x for x in vals]}")

    # K-POP
    pops = sorted(runs.glob(f"pop-{lin}-s0-*/pop.json"))
    if not pops:
        lines.append("K-POP        NOT EVALUABLE -- no population run")
    else:
        p = _load(pops[-1])
        c1 = paired(p["single_test"], p["merge_test"])
        c2 = paired(p["best_sibling_test"], p["merge_test"])
        v1 = verdict_better(c1, sigma_e)
        v2 = verdict_better(c2, sigma_e, k=0.0)
        fires = v1.startswith("FIRES") and v2.startswith("FIRES")
        lines.append(f"K-POP        {'FIRES' if fires else ('PASS' if v1 == 'PASS' else 'INCONCLUSIVE')}  "
                     f"merge - single = {c1['mean']:+.4f} ({v1}); merge - best sibling = {c2['mean']:+.4f} ({v2})  "
                     f"n={c1['n']}  [{pops[-1].parent.name}]")

    # K-ANCHOR (R4b): the assimilated lifetime against its paired free run
    sleeps = sorted(runs.glob(f"sleep-{lin}-s0-*/sleep.json"))
    if not sleeps:
        lines.append("K-ANCHOR     NOT EVALUABLE -- no sleep run")
    else:
        sl = _load(sleeps[-1])
        if sl.get("paired"):
            d = sl["lifetime"]["final_ce"] - sl["paired"]["free_ce"]
            lines.append(f"K-ANCHOR     {'FIRES' if d > 0 else 'PASS'}  sleep - free = {d:+.4f} nats "
                         f"({d / sigma_e:+.1f} sigma_e, n=1 screening)  at-birth of the final decode {sl['birth_ce']:.2f}  "
                         f"[{sleeps[-1].parent.name}]")
        else:
            lines.append("K-ANCHOR     INCONCLUSIVE -- sleep run not paired with a free run")

    # ---- §12: the programme genome -------------------------------------------------------
    lives = sorted(runs.glob(f"live-{lin}-s0*/live.json"))
    if not lives:
        lines.append("G-LIVE-mu    NOT EVALUABLE -- no liveness screen (scripts/anlage_live.py)")
    else:
        lv = _load(lives[-1])
        alive = {a: v["alive_3sigma"] for a, v in lv["live_mu"].items()}
        best_alpha = max(lv["live_mu"], key=lambda a: max(x for x in lv["live_mu"][a]["ratio_strata"] if x == x))
        lines.append(f"G-LIVE-mu    {'PASS' if any(alive.values()) else 'FIRES'}  alive at alpha: "
                     f"{ {a: bool(v) for a, v in alive.items()} }  max stratum spread {max(x for x in lv['live_mu'][best_alpha]['ratio_strata'] if x == x):.1f} "
                     f"sigma_e(q) at alpha {best_alpha}; test spread "
                     f"{ {a: round(v['spread_test'], 4) for a, v in lv['live_mu'].items()} }  "
                     f"(protocol sigma_test {lv['sigma_test']:.4f})  [{lives[-1].parent.name}]")
        for m, v in lv.get("live_eta", {}).items():
            lines.append(f"   G-LIVE-eta lr x{m}: {v['d_test']:+.4f} nats ({v['d_test_sigma']:+.1f} sigma of the protocol)")
    mixes = sorted(runs.glob(f"mix-{lin}-s0*/mix.json"))
    if not mixes:
        lines.append("K-MIX        NOT EVALUABLE -- no mixture run (scripts/anlage_mix.py)")
    else:
        mx = _load(mixes[-1])
        c = mx["contrasts"].get("star")
        if c is None:
            lines.append("K-MIX        NOT EVALUABLE -- no star arm")
        else:
            cc = {"n": c["n"], "mean": c["mean"], "ci_low": c.get("ci_low"), "ci_high": c.get("ci_high")}
            v = verdict_better(cc, sigma_e) if cc["ci_low"] is not None else ("FIRES?" if c["mean"] > -3 * sigma_e else "PASS?")
            lines.append(f"K-MIX        {v}  star - corpus = {c['mean']:+.4f} over {c['n']} seeds"
                         + (f"  {c['method']} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]" if c.get("ci_low") is not None else "")
                         + f"  (fires unless star better by 3 sigma_e = {3 * sigma_e:.4f})  [{mixes[-1].parent.name}]")
    pops2 = sorted(runs.glob(f"pop2-{lin}-s0*/pop.json"))
    if pops2:
        pp = _load(pops2[-1])
        for a, r in pp["per_alpha"].items():
            lines.append(f"K-POP(a{a})  {'PASS' if r.get('k_pop_pass') else 'FIRES'}  child - best sibling {r['delta_vs_best']:+.4f} "
                         f"({r['delta_vs_best_sigma']:+.1f} sigma), child - SINGLE {r.get('delta_vs_single', float('nan')):+.4f} "
                         f"({r.get('delta_vs_single_sigma', float('nan')):+.1f} sigma); {r['merge']['chosen']}  "
                         f"n=1 screening  [{pops2[-1].parent.name}]")
    gens = {g.parent.name: _load(g) for g in runs.glob(f"gen*-{lin}-*/generation.json")}
    if not gens:
        lines.append("K-LOOP       NOT EVALUABLE -- no generation run (scripts/anlage_generation.py)")
    else:
        by_gen = {}
        for name, g in gens.items():
            by_gen.setdefault(int(g["gen"]), {})["single" if g["single"] else "pop"] = (name, g)
        for gi in sorted(by_gen):
            pair = by_gen[gi]
            if "single" in pair and "pop" in pair:
                d = pair["pop"][1]["adult_ce"] - pair["single"][1]["adult_ce"]
                v = "PASS" if d <= -3 * sigma_e else ("INCONCLUSIVE" if d < 0 else "FIRES")
                lines.append(f"K-LOOP(g{gi})  {v}  ANLAGE-P adult - SINGLE-WARM adult = {d:+.4f} nats ({d / sigma_e:+.1f} sigma_e, n=1 screening)  "
                             f"adults {pair['pop'][1]['adult_ce']:.4f} vs {pair['single'][1]['adult_ce']:.4f}  "
                             f"[{pair['pop'][0]} vs {pair['single'][0]}]")
            else:
                have = ", ".join(f"{k}: adult {v[1]['adult_ce']:.4f}" for k, v in pair.items())
                lines.append(f"K-LOOP(g{gi})  NOT EVALUABLE -- one arm only ({have})")
    inits = sorted(runs.glob(f"init-{lin}-s0*/init.json"))
    if not inits:
        lines.append("K-INIT       NOT EVALUABLE -- no init run (scripts/anlage_init.py)")
    else:
        ij = _load(inits[-1])
        cf, cb = ij["contrast_final"], ij["contrast_birth"]
        v = verdict_better(cf, ij["sigma_e"]) if cf.get("ci_low") is not None else "INCONCLUSIVE"
        lines.append(f"K-INIT       {v}  s-birth - free after {ij['tokens']:,} tokens = {cf['mean']:+.4f}"
                     + (f"  {cf['method']} [{cf['ci_low']:+.4f}, {cf['ci_high']:+.4f}]" if cf.get("ci_low") is not None else "")
                     + f"  (at birth {cb['mean']:+.4f}); sigma_e of the run's eval version {ij['sigma_e']:.4f}  [{inits[-1].parent.name}]")
    lines.append("K-SOTA       NOT EVALUABLE -- the Pile lineage has not run")
    lines.append("K-BIRTH/K-SCALE are closed by section 11 (fired; the weight genome is not load-bearing)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
