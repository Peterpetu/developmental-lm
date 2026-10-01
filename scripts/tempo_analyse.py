"""Every verdict of the TEMPO protocol, from the run records and nothing else.

SPEC: 10 §10.1  (two fixed sets per stream; a sigma PER CRITERION from the control arm's seeds
                 on the criterion's own metric; paired CRN contrasts; comparison at equal
                 cumulative FLOPs; no missing or incomplete run is a pass)
SPEC: 10 §10.2  (the gates and kill criteria)
SPEC: 10 §10.5  (the one verdict rule)

    python scripts/tempo_analyse.py [--runs runs/tempo] [--config ts] [--json out.json]

A run directory is `<run>-<arm>[-<tag>]-<config>-s<seed>` holding `record.json` (lives) or
`result.json` (R0, R2, R10's baseline, G-ACT). **A record counts only when it says
`"complete": true`**, whichever file it is in.

Everything a criterion needs goes through one function, `criterion()`, which applies 10 §10.5
once. The second adversarial round found this layer deciding verdicts by hand, criterion by
criterion: K-HORIZON fired on a positive point estimate with no bar and no interval; a missing
arm read as "did not fire"; incomplete runs counted; a stream whose reward failed to record
had its *loss in nats* silently substituted for its reward and reported as a ten-nat
regression; K-XFER counted a stream that was never run as a loss (11 §P-C.37). One rule, one
implementation, and a test file that feeds it synthetic records with known answers.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from typing import Callable, Optional

import numpy as np
from scipy import stats

from tempo_common import REPO, RUNS, write_json
from tempo.criticality import BAND, SE_GATE, needs_control

CONFIG_BASES = ("ts", "tsk2", "tsk1", "tsk1dx512", "tsk1dc512", "tsk1wide", "pile")
"""The configuration bases `scripts/tempo_common.py::config_by_name` accepts."""
CONFIG_RE = (r"(?:" + "|".join(sorted(CONFIG_BASES, key=len, reverse=True))
             + r")(?:-(?:dg|dx|dc|r|K)\d+)*")
"""A base, then any number of `-<key><int>` modifiers. Longest base first, so `tsk1wide` is not
matched as `tsk1` with a stray tail. The modifiers' shape is distinctive enough that a config
containing hyphens cannot be confused with the optional `tag` group (11 §P-D.16)."""
NAME_RE = re.compile(r"^(?P<run>R\d+|codec|GACT)-(?P<arm>[^-]+)(?:-(?P<tag>[^-]+))??"
                     r"-(?P<config>" + CONFIG_RE + r")-s(?P<seed>\d+)$")
REWARD_STREAMS = ("task", "control", "vision", "speech")
XFER_STREAMS = ("control", "vision", "speech", "pattern")
NOT_EVALUATED = "NOT EVALUATED"
INCONCLUSIVE = "INCONCLUSIVE"
FIRED = "FIRED"
DID_NOT_FIRE = "DID NOT FIRE"
MIN_SEEDS = 2
"""SPEC: 10 §10.5 -- fewer than two paired seeds is NOT EVALUATED, not INCONCLUSIVE: the
contrast was not run, which is a different statement from "run and unclear"."""


# =============================================================================
# Loading
# =============================================================================

def load_runs(root: pathlib.Path) -> dict:
    """`(run, arm, config) -> {seed: record}` over the complete runs only."""
    runs: dict = {}
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        mc = re.match(r"^codec-(?P<config>" + CONFIG_RE + r")-s(?P<seed>\d+)$", d.name)
        if mc:
            blob = _complete(d / "result.json")
            if blob is not None:
                runs.setdefault(("codec", mc["config"], mc["config"]), {})[int(mc["seed"])] = blob
            continue
        m = NAME_RE.match(d.name)
        if not m:
            continue
        blob = _complete(d / "record.json")
        if blob is None:
            blob = _complete(d / "result.json")
        if blob is None:
            continue
        key = (m["run"], m["arm"] + (f"-{m['tag']}" if m["tag"] else ""), m["config"])
        runs.setdefault(key, {})[int(m["seed"])] = blob
    return runs


def _complete(path: pathlib.Path) -> Optional[dict]:
    """A record, if it exists, parses, and says it finished. A `result.json` is held to the
    same standard as a `record.json`: the first form checked `complete` only on records, so a
    partially written `result.json` was read as a finished run (11 §P-C.37)."""
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return blob if isinstance(blob, dict) and blob.get("complete") is True else None


# =============================================================================
# Metrics
# =============================================================================

def _f(x) -> Optional[float]:
    """A finite float, or None. Records carry `null` where a non-finite number was written
    (`gnomon.utils.report.clean`), so every read goes through here."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def stream_score(ev: Optional[dict], stream: str) -> Optional[float]:
    """A stream's evaluation as a loss, so that lower is better everywhere.

    A reward-bearing stream is scored by its **greedy reward** and enters as its negative. If
    that reward is missing or non-finite the answer is `None` -- *not* the stream's NLL. The
    first form fell through to `fresh` when `reward` was `null`, which put a loss in nats where
    a negative reward belonged: on one synthetic case it reported a 10-nat regression that was
    entirely a change of units (11 §P-C.37).
    """
    if ev is None:
        return None
    if stream in REWARD_STREAMS:
        r = _f(ev.get("reward"))
        return None if r is None else -r
    return _f(ev.get("fresh"))


def episodes(rec: dict) -> list[dict]:
    return [e for e in rec.get("episodes", []) if "L_check" in e]


def final_test(rec: dict, stream: str = "text") -> Optional[float]:
    for e in reversed(rec.get("episodes", [])):
        ev = (e.get("eval_final") or {}).get(stream)
        if ev is not None:
            return stream_score(ev, stream)
    return None


def curve(rec: dict, stream: str = "text") -> list[tuple[float, float]]:
    """`(F_cum, score)` at the end of every episode, ascending in `F_cum`."""
    out = []
    for e in episodes(rec):
        f = _f(e.get("F_cum"))
        s = stream_score((e.get("eval_final") or {}).get(stream), stream)
        if f is not None and s is not None:
            out.append((f, s))
    out.sort(key=lambda p: p[0])
    return out


def score_at(rec: dict, F_star: Optional[float], stream: str = "text") -> Optional[float]:
    """The score at cumulative compute `F_star`, by linear interpolation of the curve. `None`
    when the curve does not reach `F_star`, when it has no points, or when `F_star` is below
    its first point (10 §10.1 compares at a budget both arms actually reached)."""
    if F_star is None:
        return None
    c = curve(rec, stream)
    if not c or F_star < c[0][0] - 1e-9 or F_star > c[-1][0] * (1 + 1e-9):
        return None
    for (F0, s0), (F1, s1) in zip(c[:-1], c[1:]):
        if F0 <= F_star <= F1:
            w = 0.0 if F1 == F0 else (F_star - F0) / (F1 - F0)
            return s0 + w * (s1 - s0)
    return c[-1][1]


def common_budget(*arms: dict) -> Optional[float]:
    """The largest budget every complete run of every arm reached. `None` when any run has no
    curve at all -- an arm that recorded no `F_cum` cannot be compared at equal compute, and
    dropping it silently would compare the rest at a budget it never had to meet."""
    Fs = []
    for a in arms:
        for r in a.values():
            c = curve(r)
            if not c:
                return None
            Fs.append(c[-1][0])
    return min(Fs) if Fs else None


def rewards(rec: dict) -> list[float]:
    out = [_f(e.get("reward")) for e in rec.get("episodes", [])]
    return [v for v in out if v is not None]


def profiles(rec: dict) -> list[dict]:
    out = []
    for e in rec.get("episodes", []):
        if _f(e.get("Lambda")) is None:
            continue
        out.append({"Lambda": _f(e["Lambda"]), "n_blocks": e.get("n_blocks"),
                    "lambda_block": _f(e.get("lambda_block")), "lambda_block_se": _f(e.get("lambda_block_se")),
                    "Lambda_se": _f(e.get("Lambda_se")),
                    "mix": e.get("lambda_mix", {}), "ffn": e.get("lambda_ffn", {}),
                    "mix_se": e.get("lambda_mix_se", {}), "ffn_se": e.get("lambda_ffn_se", {})})
    return out


def early_score(rec: dict, stream: str) -> Optional[float]:
    """R7/R9: the first evaluation of the run's episode, a tenth of the way in. The run is one
    episode by construction (10 §10.3); if a record ever has more, the *first* episode's first
    evaluation is the one the criterion means."""
    evs = [e for e in rec.get("evals", []) if stream in e]
    if not evs:
        return None
    first_ep = min(e.get("episode", 0) for e in evs)
    same = sorted((e for e in evs if e.get("episode", 0) == first_ep), key=lambda e: e.get("n", 0))
    return stream_score(same[0][stream], stream) if same else None


# =============================================================================
# The one verdict rule. SPEC: 10 §10.5
# =============================================================================

def paired(a: dict, b: dict, f: Callable) -> Optional[dict]:
    """Paired by seed: the mean of `f(a) - f(b)` with its t-interval."""
    diffs = []
    for s in sorted(set(a) & set(b)):
        va, vb = f(a[s]), f(b[s])
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    if not diffs:
        return None
    n = len(diffs)
    mean = float(np.mean(diffs))
    if n >= 2:
        sd = float(np.std(diffs, ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    else:
        sd, half = float("nan"), float("nan")
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def sigma_of(arm: dict, f: Callable) -> Optional[float]:
    """The spread, over an arm's seeds, of one metric. `None` below two usable values."""
    vals = [f(r) for r in arm.values()]
    vals = [v for v in vals if v is not None]
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else None


def verdict(contrast: Optional[dict], bar: Optional[float], *, fires_when: str) -> str:
    """10 §10.5, applied once. Lower is better for every metric."""
    if contrast is None or contrast["n"] < MIN_SEEDS or bar is None or not math.isfinite(bar):
        return NOT_EVALUATED
    m, lo, hi = contrast["mean"], contrast["lo"], contrast["hi"]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return NOT_EVALUATED
    if fires_when == "not_better":
        if m <= -bar and hi < 0:
            return DID_NOT_FIRE
        return FIRED if m >= 0 else INCONCLUSIVE
    if fires_when == "worse":
        if m >= bar and lo > 0:
            return FIRED
        return DID_NOT_FIRE if hi < bar else INCONCLUSIVE
    raise ValueError(fires_when)


def criterion(name: str, mech: dict, ctrl: dict, metric: Callable, *, fires_when: str,
              sigma_arm: Optional[dict] = None, sigma_metric: Optional[Callable] = None,
              note: str = "") -> dict:
    """One criterion, end to end: the paired contrast, the bar from the **control arm's** spread
    on **this criterion's metric**, and the verdict. Everything a criterion needs is here, so a
    criterion cannot quietly use a different rule from its neighbour."""
    if not mech or not ctrl:
        return {"verdict": NOT_EVALUATED, "reason": "an arm is missing or incomplete",
                "name": name, "note": note}
    c = paired(mech, ctrl, metric)
    sig = sigma_of(sigma_arm if sigma_arm is not None else ctrl, sigma_metric or metric)
    bar = 3 * sig if sig is not None else None
    v = verdict(c, bar, fires_when=fires_when)
    out = {"name": name, "contrast": c, "sigma": sig, "bar": bar, "verdict": v,
           "fires_when": fires_when, "note": note}
    if v == NOT_EVALUATED:
        out["reason"] = ("fewer than two paired seeds" if c is None or c["n"] < MIN_SEEDS
                         else "the control arm's sigma is unmeasured")
    return out


def fmt(c: Optional[dict]) -> str:
    if c is None:
        return "n/a"
    return f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n']}"


def fbar(b: Optional[float]) -> str:
    return "n/a" if b is None or not math.isfinite(b) else f"{b:.4f}"


def line(c: dict) -> str:
    if c["verdict"] == NOT_EVALUATED:
        return f"{c['name']:26s} {NOT_EVALUATED} ({c.get('reason', 'no arm')})"
    return (f"{c['name']:26s} {fmt(c['contrast'])}  bar {fbar(c['bar'])} -> {c['verdict']}"
            + (f"   [{c['note']}]" if c["note"] else ""))


def combine(parts: list[dict]) -> str:
    """One verdict from several sub-contrasts of the same criterion (K-GENE over its arms,
    K-MORPH over its controls). A demonstrated FIRED is a fire even if another arm is missing:
    the criterion reads "does not beat **each** of the arms", and it demonstrably does not."""
    vs = [p["verdict"] for p in parts]
    if FIRED in vs:
        return FIRED
    if NOT_EVALUATED in vs:
        return NOT_EVALUATED
    if INCONCLUSIVE in vs:
        return INCONCLUSIVE
    return DID_NOT_FIRE


# =============================================================================
# The report
# =============================================================================

def analyse(runs: dict, config: str = "ts") -> dict:
    out: dict = {"config": config, "verdicts": {}, "numbers": {}}
    lines: list[str] = []

    def R(run, arm):
        return runs.get((run, arm, config), {})

    def record(c: dict, key: Optional[str] = None) -> dict:
        out["verdicts"][key or c["name"]] = c
        lines.append(line(c))
        return c

    def at_equal_F(mech: dict, ctrl: dict, stream: str = "text"):
        F = common_budget(mech, ctrl)
        return (lambda r: score_at(r, F, stream)), F

    # ---------------------------------------------------------------- R0
    r0 = runs.get(("codec", config, config)) or {}
    if r0:
        res = r0[min(r0)]
        out["numbers"]["R0"] = res
        lines.append(f"R0 SPAN  ceiling {res['ceiling_nll_per_token']:.5f} nats/token "
                     f"(I1 {'ok' if res['I1'] else 'FAILS'}); noise gap {res['noise_gap_per_token']:.5f} "
                     f"(I2 {'ok' if res['I2'] else 'FAILS'})")
    else:
        lines.append(f"R0 SPAN  {NOT_EVALUATED}")

    # ---------------------------------------------------------------- R1
    r1 = R("R1", "nolat")
    sigma_e = sigma_of(r1, final_test) if r1 else None
    if r1:
        tests = [final_test(r) for r in r1.values()]
        tests = [t for t in tests if t is not None]
        last = [r["episodes"][-1] for r in r1.values() if r.get("episodes")]
        out["numbers"]["R1"] = {
            "tests": tests, "sigma_e": sigma_e, "mean": float(np.mean(tests)) if tests else None,
            "tokens_per_s": [_f(e.get("tokens_per_s")) for e in last],
            "peak_mem_gb": [_f(e.get("peak_mem_gb")) for e in last],
            "gstat": [e.get("gstat_ok") for e in last],
            "Lambda": [_f(e.get("Lambda")) for e in last],
        }
        tps = [v for v in out["numbers"]["R1"]["tokens_per_s"] if v is not None]
        mem = [v for v in out["numbers"]["R1"]["peak_mem_gb"] if v is not None]
        lines.append(f"R1 NOISE  n={len(tests)} test {np.mean(tests) if tests else float('nan'):.4f} "
                     f"sigma_e {sigma_e if sigma_e is not None else float('nan'):.4f} "
                     f"tok/s {np.median(tps) if tps else float('nan'):.0f} "
                     f"mem {max(mem) if mem else float('nan'):.2f} GB")
    else:
        lines.append(f"R1 NOISE  {NOT_EVALUATED} (G-NOISE: nothing below is reportable)")

    lat = R("R1", "lat01")
    if lat and r1:
        c = paired(lat, r1, final_test)
        out["numbers"]["R1_latent_term"] = {"contrast_lat01_minus_nolat": c}
        lines.append(f"R1 latent term  lambda_lat=0.1 - 0 {fmt(c)}  (positive: the term hurts)")

    # ---------------------------------------------------------------- K-CONT
    r2_tok = R("R2", "token")
    metric_cont = lambda r: final_test(r) if "episodes" in r else _f(r.get("test_nll_targets_K_to_TK"))
    c = criterion("K-CONT", r1, r2_tok, metric_cont, fires_when="worse",
                  sigma_arm=r1, sigma_metric=final_test,
                  note=(f"training-FLOP ratio {np.mean([_f(r.get('flop_ratio')) or float('nan') for r in r2_tok.values()]):.3f}"
                        if r2_tok else ""))
    record(c)

    # ---------------------------------------------------------------- R3 and the K-CRIT trigger
    r3 = R("R3", "a_gamma0")
    trigger_fires = None
    if r3:
        profs = [p for r in r3.values() for p in profiles(r)]
        blocks = [p["lambda_block"] for p in profs if p["lambda_block"] is not None]
        gst = [e.get("gstat_ok") for r in r3.values() for e in r.get("episodes", [])]
        trigger_fires = needs_control(profs, BAND, SE_GATE) if profs else None
        out["numbers"]["R3"] = {
            "lambda_block_min": min(blocks) if blocks else None,
            "lambda_block_max": max(blocks) if blocks else None,
            "band": BAND, "se_gate": SE_GATE, "n_profiles": len(profs),
            "gstat_pass_rate": float(np.mean([bool(g) for g in gst])) if gst else None,
            "K-CRIT triggered": trigger_fires,
        }
        lines.append(f"R3 STAT/CRIT  per-block Lambda in "
                     f"[{min(blocks) if blocks else float('nan'):+.4f}, "
                     f"{max(blocks) if blocks else float('nan'):+.4f}] band {BAND} gate {SE_GATE} se: "
                     f"K-CRIT {'triggered' if trigger_fires else 'not needed (the profile stays in the band)'}; "
                     f"G-STAT pass rate {np.mean([bool(g) for g in gst]) if gst else float('nan'):.2f}")
    else:
        lines.append(f"R3 STAT/CRIT  {NOT_EVALUATED}")

    # ---------------------------------------------------------------- K-GENE and its arms
    tempo = R("R4", "tempo")
    gene_parts = []
    for arm in ("a_gamma0", "b_homeo", "c_random", "d_free", "e_svf", "f_es", "g_grpo"):
        ctrl = R("R4", arm)
        metric, F = at_equal_F(tempo, ctrl) if (tempo and ctrl) else (final_test, None)
        c = criterion(f"K-GENE vs {arm}", tempo, ctrl, metric, fires_when="not_better",
                      note=f"at F={F:.3e}" if F else "")
        record(c)
        if arm in ("a_gamma0", "b_homeo", "c_random", "d_free"):
            gene_parts.append(c)
    if gene_parts:
        v = combine(gene_parts)
        out["verdicts"]["K-GENE"] = {"name": "K-GENE", "verdict": v,
                                     "arms": {p["name"]: p["verdict"] for p in gene_parts}}
        lines.append(f"K-GENE -> {v}")

    # K-HORIZON: the rewards are positive on average AND the score at equal compute is worse
    ca = R("R4", "a_gamma0")
    if tempo and ca:
        metric, F = at_equal_F(tempo, ca)
        c = criterion("K-HORIZON", tempo, ca, metric, fires_when="worse",
                      note="fires only if the mean per-episode reward is also positive")
        rew = [float(np.mean(rewards(r))) for r in tempo.values() if rewards(r)]
        mean_r = float(np.mean(rew)) if rew else None
        halves = [(float(np.mean(x[:len(x) // 2])), float(np.mean(x[len(x) // 2:])))
                  for x in (rewards(r) for r in tempo.values()) if len(x) >= 2]
        if c["verdict"] == FIRED and not (mean_r is not None and mean_r > 0):
            c["verdict"] = DID_NOT_FIRE                 # worse, but not while claiming reward
            c["note"] += "; the score is worse but the mean reward is not positive"
        c["mean_reward"], c["reward_halves"] = mean_r, halves
        record(c)
    else:
        record({"name": "K-HORIZON", "verdict": NOT_EVALUATED, "reason": "R4 tempo or (a) missing", "note": ""})

    # ---------------------------------------------------------------- K-REV, K-SAL, K-MORPH, K-CRIT
    def equal_F_criterion(name, mech_run, mech_arm, ctrl_run, ctrl_arm, **kw):
        mech, ctrl = R(mech_run, mech_arm), R(ctrl_run, ctrl_arm)
        metric, F = at_equal_F(mech, ctrl) if (mech and ctrl) else (final_test, None)
        return criterion(name, mech, ctrl, metric, fires_when="not_better",
                         note=f"at F={F:.3e}" if F else "", **kw)

    record(equal_F_criterion("K-REV", "R4", "tempo", "R5", "forward"))
    record(equal_F_criterion("K-SAL", "R4", "tempo", "R5", "uniform"))
    morph = [record(equal_F_criterion(f"K-MORPH vs {a}", "R6", "tempo", "R6", a))
             for a in ("schedule", "oneshot", "flat")]
    v = combine(morph)
    out["verdicts"]["K-MORPH"] = {"name": "K-MORPH", "verdict": v,
                                  "arms": {p["name"]: p["verdict"] for p in morph}}
    lines.append(f"K-MORPH -> {v}")

    if trigger_fires:
        record(equal_F_criterion("K-CRIT", "R4", "b_homeo", "R4", "a_gamma0"))
    else:
        reason = "the profile stays in the band" if r3 else "R3 not run"
        out["verdicts"]["K-CRIT"] = {"name": "K-CRIT",
                                     "verdict": "NOT NEEDED" if r3 else NOT_EVALUATED, "reason": reason}
        lines.append(f"K-CRIT  {'NOT NEEDED' if r3 else NOT_EVALUATED} ({reason})")

    # ---------------------------------------------------------------- K-TRANSFER
    record(criterion("K-TRANSFER", R("R7", "tempo-tempo"), R("R7", "tempo-reset"),
                     lambda r: early_score(r, "task"), fires_when="not_better",
                     note="early task greedy reward"))

    # ---------------------------------------------------------------- K-XFER
    t, c_ = R("R9", "tempo-tempo"), R("R9", "tempo-reset")
    per_stream, wins, evaluated = {}, 0, 0
    for st in XFER_STREAMS:
        c = criterion(f"K-XFER {st}", t, c_, lambda r, st=st: early_score(r, st),
                      fires_when="not_better", note="early score, bar from the reset arm")
        per_stream[st] = c
        lines.append(line(c))
        if c["verdict"] != NOT_EVALUATED:
            evaluated += 1
            wins += int(c["verdict"] == DID_NOT_FIRE)
    if evaluated < len(XFER_STREAMS):
        v = NOT_EVALUATED                       # a stream that was not run is not a loss
    else:
        v = DID_NOT_FIRE if wins >= 2 else FIRED
    out["verdicts"]["K-XFER"] = {"name": "K-XFER", "verdict": v, "pairs_won": wins,
                                 "pairs_evaluated": evaluated, "per_stream": per_stream}
    lines.append(f"K-XFER -> {v} ({wins} of {evaluated} evaluated, {len(XFER_STREAMS)} required)")

    # ---------------------------------------------------------------- R8 and K-BLEND
    blend = R("R8", "tempo")
    if blend:
        by: dict[str, list] = {}
        for r in blend.values():
            for st in list(REWARD_STREAMS) + ["text", "pattern", "interleave"]:
                v0 = final_test(r, st)
                if v0 is not None:
                    by.setdefault(st, []).append(v0)
        out["numbers"]["R8"] = {st: float(np.mean(v)) for st, v in by.items()}
        lines.append("R8 BLEND  per-stream final score: "
                     + ", ".join(f"{k} {v:.3f}" for k, v in sorted(out["numbers"]["R8"].items())))
        detail, worse, missing = {}, 0, []
        for st in sorted(by):
            spec = R("R8", f"tempo-only{st}")
            if not spec:
                missing.append(st)
                continue
            c = criterion(f"K-BLEND {st}", blend, spec, lambda r, st=st: final_test(r, st),
                          fires_when="worse")
            detail[st] = c
            lines.append(line(c))
            worse += int(c["verdict"] == FIRED)
        if missing:
            v = NOT_EVALUATED                   # 10 §10.2: without the specific arms, never a pass
            reason = f"stream-specific arms missing: {missing}"
        else:
            v = FIRED if worse > 1 else DID_NOT_FIRE
            reason = ""
        out["verdicts"]["K-BLEND"] = {"name": "K-BLEND", "verdict": v, "streams_worse": worse,
                                      "missing": missing, "per_stream": detail, "reason": reason}
        lines.append(f"K-BLEND -> {v}" + (f" ({reason})" if reason else f" ({worse} streams worse)"))
    else:
        out["verdicts"]["K-BLEND"] = {"name": "K-BLEND", "verdict": NOT_EVALUATED, "reason": "R8 missing"}
        lines.append(f"K-BLEND  {NOT_EVALUATED}")

    # ---------------------------------------------------------------- G-ACT
    gact = R("GACT", "mlp")
    if gact:
        # paired by seed, the agent's return against the MLP's, in the "worse" direction with
        # both entered as losses (negated returns)
        mlp = {s: {"v": -_f(r.get("mlp_return"))} for s, r in gact.items() if _f(r.get("mlp_return")) is not None}
        agent = {s: {"v": -_f(r.get("agent_return"))} for s, r in gact.items() if _f(r.get("agent_return")) is not None}
        c = criterion("G-ACT", agent, mlp, lambda r: r["v"], fires_when="worse",
                      sigma_arm=mlp, sigma_metric=lambda r: r["v"],
                      note="the agent's shared output against a dedicated MLP")
        c["mlp_returns"] = [-v["v"] for v in mlp.values()]
        c["agent_returns"] = [-v["v"] for v in agent.values()]
        record(c)
    else:
        record({"name": "G-ACT", "verdict": NOT_EVALUATED, "reason": "no G-ACT run", "note": ""})

    # ---------------------------------------------------------------- K-SOTA
    record(criterion("K-SOTA", R("R10", "tempo"), R("R10", "token"), metric_cont,
                     fires_when="worse", sigma_arm=R("R10", "token"),
                     sigma_metric=lambda r: _f(r.get("test_nll_targets_K_to_TK")),
                     note="against the FLOP-matched token baseline at the same budget"))

    out["lines"] = lines
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--config", default=None, help="one configuration; default: every one present")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    runs = load_runs(pathlib.Path(args.runs))
    configs = [args.config] if args.config else sorted({k[2] for k in runs}) or ["ts"]
    reports = {}
    for cfg in configs:
        rep = analyse(runs, cfg)
        reports[cfg] = rep
        print(f"\n================ {cfg} ================")
        print("\n".join(rep["lines"]))
    if args.json:
        write_json(pathlib.Path(args.json), reports if len(reports) > 1 else reports[configs[0]])


if __name__ == "__main__":
    main()
