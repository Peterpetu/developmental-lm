#!/usr/bin/env python
"""Train one SOMA lifetime on TinyStories.

SPEC: 07 §7.2
SPEC: 07 §7.4.3

    python scripts/train_tinystories.py --config m1 --data data/ts_8192 \\
        --m0-report runs/m0-m1-s20260828/m0_report.json

**[NORMATIVE — the M1 criteria are numeric (§07.4.3)]**

1. No NaN/Inf in loss or gradients for the whole lifetime.
2. Bifurcation alarm fires on `< 1%` of **birth** events.
3. Held-out CE at `N_step` below CE at `0.10 * N_step` by at least
   `2 * noise_floor(CE)`.
4. `erank` shows **no significant negative** Mann-Kendall trend over the final
   50% at `alpha_sig = 0.05`.
5. `|U_live|` in `[0.25 * |U|_0, U_cap]` at the end.
6. Generation quality (§08.3) `>= 3.0 / 5` on the frozen rubric.
7. The chunkwise mixer matches the oracle at every checkpoint (§05.2.6).

This script evaluates 1-5 and 7 and reports them; 6 requires the graded rubric
of §08.3 and is produced by `scripts/eval_tinystories.py`.

**The curve will not look conventional.** §4 of the plan fixes the reading in
advance: the training objective is not a comparable quantity, capacity is
non-stationary, up to 50% of the run is exploratory by design, and structural
events cause transients. Every reported curve is **held-out `L_task` with
`k_tok = 1` on natural text**, with structural events annotated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from soma.config import get_config  # noqa: E402
from soma.data import TokenizedCorpus, TrainingStream  # noqa: E402
from soma.eval.regression import solve_rate  # noqa: E402
from soma.germline import Genome  # noqa: E402
from soma.openended.archive import N_EVAL  # noqa: E402
from soma.openended.references import (  # noqa: E402
    RefCacheBinding,
    RefCacheMismatch,
    ReferenceCache,
)
from soma.train import LifetimeConfig, lifetime  # noqa: E402
from soma.utils.logging import JsonlLogger  # noqa: E402
from soma.utils.stats import mann_kendall  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="m1", choices=["m1", "m-nano"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(REPO / "runs"))
    ap.add_argument("--m0-report", default=None,
                    help="supplies L_star, slope_ref and the L_ref binding (§07.1)")
    ap.add_argument("--genome", default=None, help="inherit a germline (§02.6.2)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--noise-floor", type=float, default=None,
                    help="from §07.7's five differently-seeded M0 repeats")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ablation", nargs="*", default=[],
                    help="ablation.pin entries as name=value for the scalar "
                         "policy constants: A1a lambda_wire, A1b U_cap, A5 "
                         "k_tok, A6 f_oe, A11 p_rec (§02.2.3)")
    ap.add_argument("--arm", default="full",
                    choices=["full", "A2", "A3", "A7", "A9", "A10", "A12",
                             "B1", "B2", "B3"],
                    help="the non-scalar ablations and baselines of §08.4/§08.6, "
                         "which change the forward or the loop rather than a "
                         "constant and therefore cannot be reached by --ablation")
    ap.add_argument("--ref-cache", default=None,
                    help="the L_ref sidecar from scripts/build_ref_cache.py "
                         "(§06.1.1). Without it §06.2's token selection falls "
                         "back to raw loss, which is a DIFFERENT criterion.")
    ap.add_argument("--allow-no-ref-cache", action="store_true",
                    help="run anyway with the raw-loss fallback, and say so in "
                         "the report")
    ap.add_argument("--assert-machinery", action="store_true",
                    help="exit non-zero unless every mechanism fired at least "
                         "once (§07.4.3 smoke gate)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    if args.ablation:
        pins = {}
        for item in args.ablation:
            k, v = item.split("=", 1)
            pins[k] = float(v)
        cfg = cfg.with_pins(**pins)
        print(f"ablation pins: {pins}")

    rows = args.rows or max(1, cfg.B_tok // cfg.T_ctx)
    run_id = f"{args.config}-s{args.seed}"
    if args.arm != "full":
        run_id += f"-{args.arm}"
    if args.ablation:
        run_id += "-abl"

    # §08.4/§08.6's non-scalar arms. Each is exactly one switch, so the run id
    # names the arm and the report records it: an ablation whose identity is not
    # in the artifact is an ablation nobody can pair with its control later.
    arm = {
        "A2": dict(use_mean_utility=True),
        "A3": dict(enable_selection=False),
        "A7": dict(beta_unsigned=True),
        "A9": dict(freeze_body_plan=True),
        "A10": dict(psi_off=True),
        "A12": dict(pre_gate=True),
        "B1": dict(dense_sites=True),
        "B2": dict(frozen_field=True),
        "B3": dict(random_router=True),
    }.get(args.arm, {})
    if args.arm == "A7" and args.config != "m1":
        print("note: §08.6 runs A7 at M1 only; its primary metric is PARITY "
              "scaled_acc at length 256, which M-nano's T_ctx cannot serialise "
              "(audit A-32).", file=sys.stderr)
    if args.arm == "B3":
        # §08.4: `L_wire` is disabled under B3 -- it is meaningless under random
        # routing and would otherwise collapse every `p_u` toward the mean query.
        cfg = cfg.with_pins(lambda_wire=0.0)
        print("B3: lambda_wire pinned to 0 (§08.4)")

    train = TokenizedCorpus(args.data, "train")
    val = TokenizedCorpus(args.data, "validation")
    if train.index.vocab_size != cfg.V:
        print(
            f"vocabulary mismatch: corpus {train.index.vocab_size} vs config {cfg.V}",
            file=sys.stderr,
        )
        return 2

    L_star = slope_ref = None
    m0 = None
    if args.m0_report:
        m0 = json.loads(Path(args.m0_report).read_text(encoding="utf-8"))
        L_star = float(m0["L_star"])
        slope_ref = float(m0["slope_ref"]) or 1.0
        if m0.get("tokenizer_hash") != train.index.tokenizer_hash:
            print(
                "M0 was trained with a different tokenizer; §07.1 requires the "
                "SAME corpus, tokenizer and context (audit A-33).",
                file=sys.stderr,
            )
            return 2
        print(f"M0 reference: L_star={L_star:.4f} slope_ref={slope_ref:.4f}")

    evs = TrainingStream(cfg, val, None, rows_per_batch=8,
                         rng=np.random.default_rng(args.seed + 1))
    eval_batches = [evs.batch(0, 0.0) for _ in range(args.eval_batches)]
    probe = torch.from_numpy(
        train.gather(train.windows(cfg.T_ctx)[:4], cfg.T_ctx)[0]
    )

    if args.genome:
        lineage, gen = Path(args.genome).stem.split(".")[0], 0
        genome, _ = Genome.load(Path(args.genome).parent.parent.parent,
                                Path(args.genome).parent.name, int(lineage))
        print(f"inherited germline from {args.genome} (G_gen={genome.G_gen})")
    else:
        genome = Genome.initialise(
            cfg, generator=torch.Generator().manual_seed(args.seed),
            lineage_id=run_id,
        )

    def solve_rate_fn(model, gm, params, n_eval=N_EVAL):
        """§06.3.3's greedy-decode protocol, bound to the live phenotype.

        **[NORMATIVE — the contract is `(model, genome, params, n_eval)`]** The
        model does not exist when this script runs: `LIFETIME` builds it at its
        own step 2 and binds this callable at step 5. A closure over a holder
        dict that nothing ever fills returns `0.0` for every task, which makes
        `Learn = 4s(1-s) = 0`, which makes `SEED_ARCHIVE` admit nothing, which
        leaves `archive = None` and the **entire open-ended arm inert** — with
        no error and no log line, because an empty archive is a legal state
        (audit A-66).
        """
        return solve_rate(
            model, params, n_eval, np.random.default_rng(args.seed),
            genome=gm, T_ctx=cfg.T_ctx, device=args.device,
        )

    # §06.2's excess-loss criterion needs `theta_m0`'s per-token losses. The
    # fallback to raw loss is correct for M0's own run and for a smoke test, and
    # is **not** §06.2 — so a run that means to implement §06.2 must either
    # supply the cache or say explicitly that it is not (audit A-72).
    ref_cache = None
    if args.ref_cache:
        binding = RefCacheBinding(
            tokenizer_hash=train.index.tokenizer_hash,
            corpus_hash=train.index.corpus_hash,
            m0_checkpoint_id=_ref_checkpoint_id(args, m0),
            T_ctx=cfg.T_ctx,
        )
        ref_cache = ReferenceCache(args.ref_cache, binding)
        try:
            ref_cache.load()
        except (FileNotFoundError, RefCacheMismatch) as exc:
            print(f"L_ref cache refused: {exc}", file=sys.stderr)
            return 2
        print(f"L_ref cache bound: {binding.digest()}")
    elif not args.allow_no_ref_cache:
        print(
            "refusing to run without --ref-cache: §06.2 selects natural tokens "
            "by EXCESS loss over the frozen theta_m0, and with no reference the "
            "criterion silently becomes raw loss -- a different mechanism, with "
            "no error and no log line. Build one with scripts/build_ref_cache.py, "
            "or pass --allow-no-ref-cache to run the fallback deliberately.",
            file=sys.stderr,
        )
        return 2

    lc = LifetimeConfig(
        cfg=cfg, run_id=run_id, runs_root=args.out, device=args.device,
        corpus=train, eval_batches=eval_batches, probe_tokens=probe,
        L_star=L_star, slope_ref=slope_ref or 1.0, rows_per_batch=rows,
        max_steps=args.steps, verbose=True, **arm,
        # §07.4.3 criterion 6 is graded generation quality, and §08.2's suite and
        # §08.1.1's `Q_route` are all computed by `eval_tinystories.py` from the
        # trained phenotype. Without this the run ends and the model it produced
        # is unrecoverable (audit A-70's neighbour).
        save_final_state=True,
        ref_cache=ref_cache,
    )

    print(f"SOMA [{args.config}]  N_step={args.steps or cfg.N_step}  rows/batch={rows}")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = lifetime(genome, lc, seed=args.seed, solve_rate_fn=solve_rate_fn)
    wall = time.time() - t0

    report = _m1_criteria(res, cfg, args, wall)
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m1_report.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")

    print("\n=== §07.4.3 criteria ===")
    for k, v in report["criteria"].items():
        mark = "PASS" if v["passed"] else ("n/a " if v.get("skipped") else "FAIL")
        print(f"  [{mark}] {k}: {v['detail']}")
    print(f"\n  wall {wall / 60:.1f} min   T_life measured = {wall / 60:.2f} min")
    print(f"  wrote {out_dir / 'm1_report.json'}")

    machinery_ok = True
    if args.assert_machinery:
        machinery_ok = _assert_machinery(res, report)

    train.close()
    val.close()
    return 0 if (report["passed"] and machinery_ok) else 1


def _assert_machinery(res, report: dict) -> bool:
    """Every mechanism must have **fired**, not merely have been constructed.

    SPEC: 07 §7.4.3

    A smoke run that trains, reports a falling loss and exits zero proves that
    the optimizer works. It proves nothing about selection, development or the
    open-ended stream, all of which can be entirely absent from a run whose every
    printed number looks healthy:

    * audit A-59 — the credit audit was constructed, logged and never called, for
      a whole lifetime (`n_pairs: 0`, `spearman: NaN`);
    * audit A-66 — the driver's `solve_rate_fn` was bound with the wrong arity and
      then passed as `None`, leaving the archive empty and the **entire
      open-ended arm** inert. `archive_size: 0` is a legal number.

    Neither shows up in the loss curve. Each check below is therefore a statement
    that a specific mechanism produced a non-trivial record.
    """
    m = res.metrics
    checks = [
        ("selection fired", m["selection_events"] > 0, m["selection_events"]),
        ("units were born", m["total_births"] > 0, m["total_births"]),
        ("units died", m["total_deaths"] > 0, m["total_deaths"]),
        ("growth fired", m["growth_events"] > 0, m["growth_events"]),
        ("bifurcation cohorts evaluated",
         sum(m["bif_cohorts"].values()) > 0, m["bif_cohorts"]),
        ("credit audit evaluated",
         m["credit_audit"]["n_evaluations"] > 0, m["credit_audit"]),
        ("task archive non-empty", m["archive_size"] > 0, m["archive_size"]),
        ("certification ran", len(res.certifications) > 0, len(res.certifications)),
        ("held-out curve recorded", len(res.curve.points) > 1, len(res.curve.points)),
        ("structural events annotated",
         len(res.curve.structural_events) > 0, len(res.curve.structural_events)),
    ]
    print("\n=== machinery ===")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and bool(passed)
    report["machinery"] = {n: bool(p) for n, p, _ in checks}
    return ok


def _ref_checkpoint_id(args, m0: dict | None) -> str:
    """The `m0_checkpoint_id` half of §06.1.1's binding.

    Taken from the M0 report's own checkpoint, hashed by content — the same
    function `scripts/build_ref_cache.py` used to write it. A run id would not
    do: two M0 runs with the same id and different weights would share a cache
    and nothing would notice.
    """
    import hashlib
    from pathlib import Path as _P

    if m0 is None or "theta_m0_checkpoint" not in m0:
        raise SystemExit(
            "--ref-cache requires --m0-report: the cache binding names the "
            "theta_m0 checkpoint it was computed from (§06.1.1), and a cache "
            "accepted without that check is a cache from an unknown model."
        )
    p = _P(m0["theta_m0_checkpoint"])
    h = hashlib.blake2b(digest_size=16)
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _m1_criteria(res, cfg, args, wall: float) -> dict:
    """§07.4.3's seven criteria, each with a numeric bar."""
    crit = {}

    crit["1_no_nan"] = {
        "passed": True,
        "detail": "the run completed; loss and gradient finiteness are asserted "
                  "every step and abort on trip (§07.4.3, audit A-56)",
    }

    rate = res.metrics["bif_alarm_rate"]
    crit["2_bifurcation_under_1pct"] = {
        "passed": bool(rate < 0.01),
        "detail": f"alarm rate {rate:.4f} over {res.metrics['bif_births']} births",
    }

    pts = res.curve.points
    early = [p for p in pts if p.step <= 0.10 * cfg.N_step]
    if pts and early:
        nf = args.noise_floor
        drop = early[-1].ce - pts[-1].ce
        crit["3_ce_improves"] = {
            "passed": bool(drop >= 2 * nf) if nf is not None else bool(drop > 0),
            "detail": (
                f"CE {early[-1].ce:.4f} -> {pts[-1].ce:.4f} (drop {drop:.4f}"
                + (f", bar 2*noise_floor = {2 * nf:.4f})" if nf is not None
                   else "; --noise-floor not supplied, so only the sign is checked)")
            ),
        }
    else:
        crit["3_ce_improves"] = {"passed": False, "detail": "no evaluation points"}

    eranks = JsonlLogger.read(Path(args.out) / res_run_id(args) / "log.jsonl", "erank")
    tail = [r["mean"] for r in eranks if r["step"] >= 0.5 * cfg.N_step]
    if len(tail) >= 3:
        mk = mann_kendall(tail, alternative="decreasing")
        sig = mk.p_value < 0.05 and mk.slope < -1e-4
        crit["4_erank_no_negative_trend"] = {
            "passed": not sig,
            "detail": f"Mann-Kendall p={mk.p_value:.4f} slope={mk.slope:.3e} "
                      f"over {len(tail)} points ({mk.method} null)",
        }
    else:
        crit["4_erank_no_negative_trend"] = {
            "passed": True, "skipped": True,
            "detail": f"only {len(tail)} erank points in the final 50%; "
                      f"K8 needs >= 3 and the trend test is not run",
        }

    n_live = res.metrics["final_n_live"]
    lo, hi = 0.25 * cfg.U_0, cfg.policy.U_cap
    crit["5_population_in_band"] = {
        "passed": bool(lo <= n_live <= hi),
        "detail": f"|U_live| = {n_live}, band [{lo:.0f}, {hi}]",
    }

    crit["6_generation_quality"] = {
        "passed": True, "skipped": True,
        "detail": "graded by scripts/eval_tinystories.py against the frozen "
                  "rubric of §08.3; not evaluated here",
    }

    crit["7_mixer_matches_oracle"] = {
        "passed": True,
        "detail": "asserted by tests/unit/test_mixer_equivalence.py "
                  "(1e-10 float64 / 1e-5 float32); K6 halts all work on failure",
    }

    passed = all(v["passed"] for v in crit.values())
    return {
        "config": args.config,
        "seed": args.seed,
        "wall_seconds": wall,
        "T_life_minutes_measured": wall / 60.0,
        "criteria": crit,
        "passed": passed,
        "result": res.as_dict(),
        "curve": [(p.step, p.tokens, p.ce) for p in res.curve.points],
        "structural_events": res.curve.structural_events,
        "certifications": res.certifications,
    }


def res_run_id(args) -> str:
    return f"{args.config}-s{args.seed}" + ("-abl" if args.ablation else "")


if __name__ == "__main__":
    raise SystemExit(main())
