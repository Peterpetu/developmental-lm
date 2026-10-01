"""Verdicts from records on disk. SPEC: docs/palimpsest/02-protocol.md."""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Optional

__all__ = ["load_records", "frontier", "verdict_sieve", "verdict_stat", "control_verdict", "report"]

RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs" / "palimpsest"
SIEVE_ORDER = ["copy", "sieve-c2", "sieve-c4", "sieve-c8", "sieve-c16", "scratch"]


def load_records(root: pathlib.Path = RUNS) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("*/record.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            continue
        if r.get("complete"):
            out[p.parent.name] = r
    return out


def _arm_name(r: dict) -> str:
    a = r["cfg"]["arm"]
    if a in ("sieve", "sieve-nodream", "sieve-fresh", "sieve-life", "trunc"):
        a = f"{a}-{r['cfg']['core']}"
    cfg = r["cfg"]
    if a == "ewc":
        return f"ewc-{cfg['lam_ewc']:.0e}"
    if a not in ("copy", "scratch", "iid", "replay", "ewc", "lwf", "shrink_perturb") and cfg.get("dream_start", "mixed") != "mixed":
        a = f"{a}-{cfg['dream_start']}"
    if a.startswith("sieve") and (cfg.get("lam_dream", 1.0) != 1.0 or cfg.get("dream_tokens", 250_000) not in (250_000, 500_000)):
        a = f"{a}-dream{cfg['dream_tokens'] / 1e6:g}Mx{cfg['lam_dream']:g}"
    if a == "copy" and cfg.get("tokens_per_phase", 8_000_000) != 8_000_000:
        a = "copy-m"                                        # the FLOP-matched copy (01 §1.8)
    return a
    if a == "ewc":
        return f"ewc-{r['cfg']['lam_ewc']:.0e}"
    return a


def by_life(records: dict, life: str) -> dict[str, dict[int, dict]]:
    """`arm -> seed -> record` for one life."""
    out: dict[str, dict[int, dict]] = {}
    for name, r in records.items():
        if r.get("life") != life:
            continue
        out.setdefault(_arm_name(r), {})[int(r["seed"])] = r
    return out


def _u(r: dict) -> float:
    return float(r["scores"]["U_K"])


def sigma_u(arms: dict, control: str = "copy") -> Optional[float]:
    rs = arms.get(control, {})
    if len(rs) < 2:
        return None
    return statistics.stdev(_u(r) for r in rs.values())


def paired(arms: dict, a: str, b: str) -> Optional[dict]:
    """Paired (a - b) over the seeds both have; the t interval at n >= 2."""
    if a not in arms or b not in arms:
        return None
    seeds = sorted(set(arms[a]) & set(arms[b]))
    if not seeds:
        return None
    d = [_u(arms[a][s]) - _u(arms[b][s]) for s in seeds]
    m = statistics.mean(d)
    if len(d) >= 2:
        sd = statistics.stdev(d)
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(len(d), 2.0)
        half = t * sd / math.sqrt(len(d))
        return {"mean": m, "lo": m - half, "hi": m + half, "n": len(d), "seeds": seeds}
    return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": 1, "seeds": seeds}


def frontier(arms: dict) -> list[tuple[str, float, float, int]]:
    """`(arm, ratio, mean U_K, n)` over the frontier arms present."""
    out = []
    for a in SIEVE_ORDER:
        if a in arms:
            rs = arms[a]
            r0 = next(iter(rs.values()))
            ratio = 1.0 if a == "copy" else (float("inf") if a == "scratch" else float(r0["ratio"]))
            out.append((a, ratio, statistics.mean(_u(r) for r in rs.values()), len(rs)))
    return out


def verdict_sieve(arms: dict) -> dict:
    """K-SIEVE. SPEC: 02 §2.2."""
    fr = frontier(arms)
    if "copy" not in arms or len(fr) < 3:
        return {"verdict": "NOT EVALUATED", "reason": "copy or fewer than two sieve arms missing", "frontier": fr}
    sieves = [f for f in fr if f[0].startswith("sieve")]
    if not sieves:
        return {"verdict": "NOT EVALUATED", "reason": "no sieve arm", "frontier": fr}
    best = min(fr, key=lambda f: f[2])
    sig = sigma_u(arms)
    out = {"frontier": fr, "argmin": best[0], "sigma_U": sig}
    if best[0] == "copy" or best[0] == "scratch":
        out["verdict"] = "FIRES"
        out["reason"] = f"argmin is {best[0]}"
        return out
    pc = paired(arms, "copy", best[0])
    out["copy_minus_best"] = pc
    if sig is None:
        out["verdict"] = "SCREEN"
        out["reason"] = f"interior argmin {best[0]} at n=1; sigma_U unmeasured (G-NOISE)"
        return out
    bar = 3 * sig
    out["bar"] = bar
    if pc["mean"] >= bar and (pc["n"] >= 2 and pc["lo"] > 0):
        out["verdict"] = "LOAD-BEARING"
    elif pc["mean"] <= 0:
        out["verdict"] = "FIRES"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def verdict_stat(arms: dict) -> dict:
    """K-STAT. SPEC: 02 §2.2."""
    fr = frontier(arms)
    if "copy" not in arms or not any(f[0].startswith("sieve") for f in fr):
        return {"verdict": "NOT EVALUATED", "frontier": fr}
    sig = sigma_u(arms)
    u_copy = statistics.mean(_u(r) for r in arms["copy"].values())
    worst = min((f for f in fr if f[0].startswith("sieve")), key=lambda f: f[2])
    out = {"frontier": fr, "sigma_U": sig, "best_sieve": worst[0]}
    tol = sig if sig is not None else 0.0
    if u_copy <= worst[2] + tol:
        out["verdict"] = "MONOTONE"
    elif sig is not None and (u_copy - worst[2]) >= 3 * sig:
        out["verdict"] = "CONTRADICTS PROPOSITION 1"
    else:
        out["verdict"] = "SCREEN: a sieve arm below copy at n=1" if sig is None else "INCONCLUSIVE"
    return out


def control_verdict(arms: dict, mech: str, control: str, *, better_is: str = "lower") -> dict:
    """A 'not better' criterion: fires unless `mech` beats `control` by 3 sigma. SPEC: 02 §2.3."""
    pc = paired(arms, mech, control)
    if pc is None:
        return {"verdict": "NOT EVALUATED", "mech": mech, "control": control}
    sig = sigma_u(arms)
    out = {"mech": mech, "control": control, "mech_minus_control": pc, "sigma_U": sig}
    if sig is None:
        out["verdict"] = "SCREEN " + ("(mech ahead)" if pc["mean"] < 0 else "(mech behind)")
        return out
    bar = 3 * sig
    out["bar"] = bar
    if pc["mean"] <= -bar and pc["n"] >= 2 and pc["hi"] < 0:
        out["verdict"] = "DID NOT FIRE"
    elif pc["mean"] >= 0:
        out["verdict"] = "FIRES"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def report(records: dict) -> str:
    lines = []
    for life in ("shift", "stat", "shift3", "statcode"):
        arms = by_life(records, life)
        if not arms:
            continue
        lines.append(f"== life {life}")
        for a in sorted(arms):
            for s, r in sorted(arms[a].items()):
                sc = r["scores"]
                fin = "  ".join(f"{e} {c:.4f}" for e, c in sc["final"].items())
                g = f"  G_K {sc['G_K']:.4f}" if "G_K" in sc else ""
                ret = "  ".join(f"R[{k}] {v:+.3f}" for k, v in sc.get("retention", {}).items())
                lines.append(f"  {a:<18s} s{s}  U_K {sc['U_K']:.4f}  Lambda {sc['Lambda']:.4f}{g}  "
                             f"overhead {r['flops'].get('overhead_frac', 0):.2f}  | {fin}  | {ret}")
                if sc.get("core_curve"):
                    cc = "  ".join("{" + ", ".join(f"{e} {c:.2f}" for e, c in (ce or {}).items()) + "}" for ce in sc["core_curve"])
                    lines.append(f"  {'':<18s}     core after each phase: {cc}")
                if r.get("births"):
                    bb = "  ".join("{" + ", ".join(f"{e} {c:.2f}" for e, c in b.get("evals", {}).items()) + "}" for b in r["births"])
                    lines.append(f"  {'':<18s}     at birth: {bb}")
        if life in ("shift", "shift3"):
            ks = verdict_sieve(arms)
            lines.append(f"  K-SIEVE  {ks['verdict']}  " + json.dumps({k: v for k, v in ks.items() if k != 'frontier'}, default=str))
            lines.append("    frontier: " + ", ".join(f"{a} (r={r:.1f}) {u:.4f} n={n}" for a, r, u, n in ks.get("frontier", [])))
            gk = [(a, statistics.mean(r["scores"]["G_K"] for r in rs.values()), len(rs))
                  for a, rs in sorted(arms.items()) if all("G_K" in r["scores"] for r in rs.values())]
            if gk:
                lines.append("    the genome's own union score G_K: " + ", ".join(f"{a} {g:.4f} n={n}" for a, g, n in gk))
            best = ks.get("argmin") if ks.get("argmin", "").startswith("sieve") else None
            if best is None:
                sv = [f for f in frontier(arms) if f[0].startswith("sieve")]
                best = min(sv, key=lambda f: f[2])[0] if sv else None
            if best:
                core = best.split("-", 1)[1]
                # the ablations are read against the sieve of their own core (they were run at c8)
                def same_core(prefix):
                    cs = [a for a in arms if a.startswith(prefix)]
                    return cs[0] if cs else f"{prefix}{core}"
                abl_core = same_core("trunc-").split("-")[-1] if any(a.startswith("trunc-") for a in arms) else core
                sieve_abl = f"sieve-{abl_core}" if f"sieve-{abl_core}" in arms else best
                for name, mech, ctrl in (("K-REPLAY", best, "replay"), ("K-LWF", best, "lwf"),
                                         ("K-SP", best, "shrink_perturb"), ("K-SCRATCH", best, "scratch"),
                                         ("K-DREAM", sieve_abl, same_core("sieve-nodream-")),
                                         ("K-GROW-C(trunc)", sieve_abl, same_core("trunc-")),
                                         ("K-GROW-C(fresh)", sieve_abl, same_core("sieve-fresh-")),
                                         ("K-SOTA-C", best, "iid"),
                                         ("K-DREAM-LIFE(vs sieve)", f"sieve-life-{core}", best),
                                         ("K-DREAM-LIFE(vs copy-dream)", f"sieve-life-{core}", "copy-dream"),
                                         ("  sieve-life vs copy", f"sieve-life-{core}", "copy"),
                                         ("  sieve-life vs replay", f"sieve-life-{core}", "replay"),
                                         ("  copy-dream vs copy", "copy-dream", "copy")):
                    v = control_verdict(arms, mech, ctrl)
                    lines.append(f"  {name:<16s} {v['verdict']}  " + json.dumps({k: vv for k, vv in v.items() if k != 'verdict'}, default=str))
                for a in sorted(arms):
                    if a.startswith("ewc"):
                        v = control_verdict(arms, best, a)
                        lines.append(f"  K-EWC({a})  {v['verdict']}  " + json.dumps({k: vv for k, vv in v.items() if k != 'verdict'}, default=str))
                # K-GENOME: the core as a model against the copied phenotype
                if best in arms and "copy" in arms:
                    gk = [r["scores"].get("G_K") for r in arms[best].values() if "G_K" in r["scores"]]
                    uc = statistics.mean(_u(r) for r in arms["copy"].values())
                    if gk:
                        lines.append(f"  K-GENOME  {'FIRES' if statistics.mean(gk) >= uc else 'DID NOT FIRE'}  G_K {statistics.mean(gk):.4f} vs U_K(copy) {uc:.4f}")
        if life == "stat":
            kst = verdict_stat(arms)
            lines.append(f"  K-STAT  {kst['verdict']}  " + json.dumps({k: v for k, v in kst.items() if k != 'frontier'}, default=str))
            lines.append("    frontier: " + ", ".join(f"{a} (r={r:.1f}) {u:.4f} n={n}" for a, r, u, n in kst.get("frontier", [])))
    return "\n".join(lines) if lines else "no complete records under runs/palimpsest"
