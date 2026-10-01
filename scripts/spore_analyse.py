"""Every verdict of docs/spore/02-protocol.md, recomputed from the artifacts in runs/spore/.

    python scripts/spore_analyse.py

A criterion whose deciding run does not exist prints NOT EVALUATED. Nothing here is edited
by hand; the ledger quotes this script's output.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "spore"
L_NULL = math.log(8192)

B_GD = [(819200, 4.306, "train"), (1638400, 3.205, "train"), (2457600, 2.855, "train"),
        (10002432, 2.2255, "test"), (20004864, 1.9975, "test"), (25000000, 1.857, "test, 4 seeds"),
        (201015296, 1.804, "test")]
"""The backprop ceiling (02 §2.2): the 13M transformer `s0:d256/L8`, from
`runs/bottleneck/free-tinystories-s0-seed0/loop.json` and `docs/anlage/RESULTS.md` R1."""


def gd_at(tokens: float) -> float:
    """Log-linear interpolation of B-GD's curve at `tokens` (an interpolation, labelled as such)."""
    pts = [(math.log(t), c) for t, c, _ in B_GD]
    x = math.log(tokens)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def load(name: str) -> dict | None:
    p = RUNS / name / "life.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return None if r.get("failed") or not r.get("complete") else r


def counts(name: str) -> dict | None:
    p = RUNS / name / "counts.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stage(r: dict, name: str, key: str = "test_ce"):
    s = r.get("stages", {}).get(name)
    return None if s is None else s.get(key)


def fmt(x, nd=4):
    return "   -  " if x is None else f"{x:.{nd}f}"


def evolution_summary() -> list[str]:
    """K-EVO: the (mu + lambda) loop's record from runs/spore/evo-*/history.json."""
    out = []
    for h in sorted(RUNS.glob("evo-*/history.json")):
        H = json.loads(h.read_text(encoding="utf-8"))
        gens = H.get("generations", [])
        if not gens:
            continue
        base = gens[0]["population"]
        base_rec = next((p for p in base if p["genome"] == H["base"]), base[0])
        out.append(f"{h.parent.name}: {H.get('n_lives', '?')} lives, {len(gens)} generations, {H.get('seconds', 0) / 3600:.2f} h; "
                   f"base fitness {base_rec['fitness_ce']:.4f} test {base_rec['test_ce']:.4f}")
        for g_ in gens:
            out.append(f"   gen {g_['gen']:2d}: elite fitness {g_['elite_fitness']:.4f}  test {g_['elite_test']:.4f}  "
                       f"(pop {min(p['fitness_ce'] for p in g_['population']):.4f}..{max(p['fitness_ce'] for p in g_['population']):.4f})")
        e = H.get("elite")
        if e:
            moved = {k: (H["base"][k], e["genome"][k]) for k in H["base"] if H["base"][k] != e["genome"][k]}
            out.append(f"   elite genome moved: {moved}")
            out.append(f"   K-EVO (screen): elite - base = {e['test_ce'] - base_rec['test_ce']:+.4f} on the test set at the screening budget "
                       f"(the pre-registered reading is the elite re-scored at 5M against sem05-5M; see the ledger)")
    return out


def main() -> int:
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _main()
    text = buf.getvalue()
    text += "\n== K-EVO\n" + "\n".join(evolution_summary() or ["NOT EVALUATED (no evo-*/history.json)"]) + "\n"
    print(text, end="")
    if "--markdown" in sys.argv:
        md = REPO / "docs" / "spore" / "RESULTS.md"
        import time
        md.write_text("# SPORE — the results register\n\nRendered by `python scripts/spore_analyse.py --markdown` on "
                      + time.strftime("%Y-%m-%d %H:%M") + " from `runs/spore/`; every number carries its artifact "
                      "(`runs/spore/<run>/life.json`, `counts.json`, `evo-*/history.json`). Regenerate after every run; "
                      "never edit by hand. Verdicts: `docs/spore/02-protocol.md`; the readings and their reasons: "
                      "`03-evidence-ledger.md`.\n\n```\n" + text + "```\n", encoding="utf-8")
        print(f"wrote {md}")
    return 0


def _main() -> int:
    runs = {p.parent.name: load(p.parent.name) for p in sorted(RUNS.glob("*/life.json"))}
    runs = {k: v for k, v in runs.items() if v is not None}
    print("== lives (test CE per stage; final = the stage the FITNESS set chose)")
    print(f"{'run':<22s} {'tokens':>9s} {'F':>6s} {'M':>5s} {'birth':>7s} {'lin':>7s} {'soft':>7s} {'refin':>7s} {'final':>7s} {'fit':>7s} {'stage':<14s} {'s':>5s}")
    for k, r in runs.items():
        print(f"{k:<22s} {r['tokens_seen'] / 1e6:>8.2f}M {r['F_final']:>6d} {r['M_units']:>5d} "
              f"{fmt(stage(r, 'birth')):>7s} {fmt(stage(r, 'ridge-linear')):>7s} {fmt(stage(r, 'ridge-softmax')):>7s} "
              f"{fmt(stage(r, 'refined')):>7s} {r['test_ce']:>7.4f} {r['fitness_ce']:>7.4f} {r['final_stage']:<14s} {r['timings']['total']:>5.0f}")
    print()
    print("== nulls and the ceiling")
    for name in ("counts-05M", "counts-5M", "counts-25M"):
        c = counts(name)
        if c:
            print(f"{name:<12s} unigram {c['unigram']['test_ce']:.4f}  bigram {c['bigram']['test_ce']:.4f}  trigram {c['trigram']['test_ce']:.4f}  "
                  f"(lam2 {c['lam2']:.2f}, lam3 {c['lam3']:.2f}; {c['tokens_seen'] / 1e6:.1f}M tokens)")
    print("B-GD (13M transformer): " + ", ".join(f"{t / 1e6:.1f}M -> {c} ({w})" for t, c, w in B_GD))
    print()
    # ---- G-NOISE
    BASE = "sem05-5M"   # the organism of record from 03 §P-S.16 on (semantic base, E = 1)
    seeds = [runs[k] for k in (BASE, "sem05-5M-s1", "sem05-5M-s2") if k in runs]
    sigma = None
    if len(seeds) >= 2:
        vals = [r["test_ce"] for r in seeds]
        sigma = st.stdev(vals)
        lin = [stage(r, "ridge-linear") for r in seeds]
        print(f"G-NOISE  PASS  sigma_e = {sigma:.4f} nats ({len(vals) - 1} df) over atom seeds; test CE {', '.join(f'{v:.4f}' for v in vals)}; "
              f"linear-head sigma {st.stdev(lin):.4f}")
    else:
        print(f"G-NOISE  NOT EVALUATED (fewer than 2 seeds of {BASE})")
    bar = 3 * sigma if sigma else None
    base = runs.get(BASE)
    # ---- R-BIRTH
    if base and stage(base, "birth") is not None:
        b = stage(base, "birth")
        verdict = "PASS" if (bar is None and b < L_NULL) or (bar is not None and b < L_NULL - bar) else "FAIL"
        print(f"R-BIRTH  {verdict}  innate memory-only predictor {b:.4f} vs null {L_NULL:.4f} (delta {b - L_NULL:+.4f}; tau_m {base['stages']['birth']['tau_m']:.2f})")
    else:
        print("R-BIRTH  NOT EVALUATED")
    # ---- R-PARADIGM / K-SOTA
    for name, cname, tok in (("sem05-E3-5M", "counts-5M", 5e6), ("sem05-E3-25M", "counts-25M", 25e6)):
        r, c = runs.get(name), counts(cname)
        if r is None:
            print(f"R-PARADIGM@{tok / 1e6:.0f}M  NOT EVALUATED ({name} missing)")
            continue
        line = f"R-PARADIGM@{tok / 1e6:.0f}M  {name} test {r['test_ce']:.4f}"
        if c:
            d = r["test_ce"] - c["trigram"]["test_ce"]
            v = "PASS" if (bar is not None and d < -bar) or (bar is None and d < 0) else "FAIL"
            line += f"  vs trigram {c['trigram']['test_ce']:.4f} (delta {d:+.4f}) -> {v}"
        gd = gd_at(tok)
        line += f"  | K-SOTA gap to B-GD at {tok / 1e6:.0f}M ({gd:.3f}, interpolated): {r['test_ce'] - gd:+.4f}"
        print(line)
    # slopes
    r5, r25 = runs.get("sem05-E3-5M"), runs.get("sem05-E3-25M")
    if r5 and r25:
        s_sp = (r25["test_ce"] - r5["test_ce"]) / math.log2(25 / 5)
        s_gd = (gd_at(25e6) - gd_at(5e6)) / math.log2(25 / 5)
        print(f"K-SOTA slope 5M->25M: SPORE {s_sp:+.4f} nats/doubling vs B-GD {s_gd:+.4f}  -> {'FIRES (flatter)' if s_sp > s_gd else 'does not fire'}")
    # ---- K-HDC
    if base:
        print(f"K-HDC arms vs {BASE} (delta > 0 = the ablation hurts):")
        hits = 0
        n = 0
        for arm in ("s-nobind-5M", "s-nomem-5M", "s-D1024-5M"):
            r = runs.get(arm)
            if r is None:
                print(f"   {arm:<12s} NOT EVALUATED")
                continue
            n += 1
            d = r["test_ce"] - base["test_ce"]
            dl = (stage(r, "ridge-linear") or float('nan')) - (stage(base, "ridge-linear") or float('nan'))
            hit = bar is not None and d >= bar
            hits += int(hit)
            print(f"   {arm:<12s} final {r['test_ce']:.4f} (delta {d:+.4f}; linear-head delta {dl:+.4f}) {'>= 3 sigma' if hit else ('< 3 sigma' if bar else '')}")
        if n == 3 and bar is not None:
            print(f"K-HDC  {'FIRES' if hits < 2 else 'does not fire'}  ({hits} of 3 ablations cost >= 3 sigma_e = {bar:.4f})")
        else:
            print("K-HDC  NOT EVALUATED")
    # ---- K-DEV
    if base:
        go, gr = runs.get("s-growoff-5M"), runs.get("s-growrand-5M")
        if go and gr and bar is not None:
            d_off = go["test_ce"] - base["test_ce"]
            d_rand = gr["test_ce"] - base["test_ce"]
            fires = not (d_off >= bar and d_rand >= bar)
            print(f"K-DEV  {'FIRES' if fires else 'does not fire'}  select {base['test_ce']:.4f} vs off {go['test_ce']:.4f} (delta {d_off:+.4f}) vs random {gr['test_ce']:.4f} (delta {d_rand:+.4f}); bar {bar:.4f}")
            gg = runs.get("s-gridge-5M")
            if gg:
                d_g_off, d_g_rand = gg["test_ce"] - go["test_ce"], gg["test_ce"] - gr["test_ce"]
                ok = d_g_off <= -bar and d_g_rand <= -bar
                print(f"K-DEV (11b, ridge-born units)  {'does not fire' if ok else 'FIRES'}  ridge {gg['test_ce']:.4f} vs off (delta {d_g_off:+.4f}) vs random (delta {d_g_rand:+.4f}); bar {bar:.4f}")
        else:
            print("K-DEV  NOT EVALUATED")
    # ---- K-REFINE
    if base:
        ref = stage(base, "refined")
        lin = stage(base, "ridge-linear")
        if ref is not None and lin is not None and bar is not None:
            d = ref - lin
            print(f"K-REFINE  {'does not fire' if d <= -bar else 'FIRES'}  refined {ref:.4f} vs best ridge head {lin:.4f} (delta {d:+.4f}; bar {bar:.4f})")
        else:
            print("K-REFINE  NOT EVALUATED")
    # ---- G-DET
    for a, b in (("D2048-05M", "det-05M"), ("sem05-05M", "sem05-det-05M"), ("det2a-05M", "det2b-05M"), ("det3a-05M", "det3b-05M")):
        ra, rb = runs.get(a), runs.get(b)
        if ra and rb:
            d = abs(ra["test_ce"] - rb["test_ce"])
            print(f"G-DET  {'PASS' if d <= 1e-4 else 'FAIL'}  {a} {ra['test_ce']:.6f} vs {b} {rb['test_ce']:.6f} (|delta| {d:.2e})")
        else:
            print(f"G-DET  NOT EVALUATED ({a} / {b})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
