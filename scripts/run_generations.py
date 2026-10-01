#!/usr/bin/env python
"""The M3 decisive experiment: the generation curve at M-nano.

SPEC: 07 §7.5
SPEC: 08 §8.6
SPEC: 10 §10.1

    python scripts/run_generations.py --data data/ts_4096 --generations 20 \
        --seeds 1 2 3 4 5 --out runs/mnano-gen

**What the experiment asks.** Does a germline that is *selected across
generations* produce lifetimes that adapt **faster** than a germline that is
merely re-randomised each generation? The primary statistic is **adaptation
efficiency** — time-to-criterion `n_crit` and early-phase slope — not final
loss, and not birth performance.

**Why not birth performance.** `AUDIT-v1.md` A-2: every tensor that touches data
is soma and is re-randomised each lifetime, so inheriting gene *index sets*
cannot improve step-0 loss. `BirthScore ~ 0` is the **prediction**, and it is
retained as a Weismann-barrier diagnostic: a significantly non-zero value
indicates leakage across the barrier, not success.

**The two arms.** `main` inherits and selects; `a4` (§08.6 ablation A4) resets
the germline to random each generation with the ES perturbation scale matched,
so it measures ordinary recipe drift with no inheritance. Both arms run the same
seeds, the same lifetimes and the same budget; the only difference is whether
generation `g` starts from generation `g-1`.

**K1 fires iff *both* pre-registered tests fail** — a conjunction, not a
disjunction (§07.5; spec v1's §07.5 said "either" and §10 said "or", and the two
readings decide the project's central verdict differently).
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
from soma.generations import (  # noqa: E402
    analyse_arm_difference,
    analyse_curves,
    budget_report,
    run_generations,
)
from soma.germline.es import EsHyper  # noqa: E402
from soma.openended.archive import N_EVAL  # noqa: E402
from soma.train import LifetimeConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="m-nano", choices=["m-nano", "m1"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(REPO / "runs" / "generations"))
    ap.add_argument("--generations", type=int, default=20,
                    help="G_gen; §07.5 asks for >= 20 at M-nano")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="§07.5 requires n_seed >= 5 for alpha_sig = 0.05")
    ap.add_argument("--arms", nargs="+", default=["main", "a4"],
                    choices=["main", "a4"])
    ap.add_argument("--steps", type=int, default=None,
                    help="override N_step per lifetime (budget control)")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--L-star", type=float, default=None,
                    help="§07.1: the M0 criterion loss; from --m0-report if absent")
    ap.add_argument("--m0-report", default=None)
    ap.add_argument("--l-star-fraction", type=float, default=0.50,
                    help="§07.1 (audit A-81): take L_star as the M0 baseline's "
                         "held-out CE at this fraction of the matched token "
                         "budget. 1.0 is the full budget and is NOT the default "
                         "-- see the note in --help output and §07.1.")
    ap.add_argument("--slope-ref", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--assert-complete", action="store_true",
                    help="exit non-zero unless every arm ran every seed to G_gen "
                         "and the curve is interpretable (§07.5 censoring rule)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    rows = args.rows or max(1, cfg.B_tok // cfg.T_ctx)

    L_star = args.L_star
    slope_ref = args.slope_ref
    if args.m0_report:
        m0 = json.loads(Path(args.m0_report).read_text(encoding="utf-8"))
        slope_ref = float(m0["slope_ref"]) or 1.0
        by_frac = m0.get("L_star_by_fraction") or {}
        key = f"{args.l_star_fraction:.2f}"
        if key in by_frac:
            L_star = float(by_frac[key])
            print(f"L_star = {L_star:.4f} (M0 at {args.l_star_fraction:.0%} of the "
                  f"matched budget; full-budget L_star is {float(m0['L_star']):.4f})")
        elif args.l_star_fraction >= 1.0:
            L_star = float(m0["L_star"])
        else:
            print(
                f"the M0 report has no L_star at fraction {key}; it carries "
                f"{sorted(by_frac)}. §07.1 (audit A-81) selects L_star from that "
                f"curve, so re-run scripts/run_m0.py to produce it rather than "
                f"substituting the full-budget value, which censors every "
                f"lifetime at M-nano.",
                file=sys.stderr,
            )
            return 2
    if L_star is None:
        print(
            "refusing to run without L_star: §07.1 defines the criterion loss as "
            "the M0 baseline's held-out CE at a MATCHED token budget, and a "
            "time-to-criterion measured against an invented threshold is not the "
            "primary statistic of anything. Pass --m0-report or --L-star.",
            file=sys.stderr,
        )
        return 2

    train = TokenizedCorpus(args.data, "train")
    val = TokenizedCorpus(args.data, "validation")
    if train.index.vocab_size != cfg.V:
        print(f"vocabulary mismatch: corpus {train.index.vocab_size} vs config {cfg.V}",
              file=sys.stderr)
        return 2

    evs = TrainingStream(cfg, val, None, rows_per_batch=8,
                         rng=np.random.default_rng(args.seeds[0] + 1))
    eval_batches = [evs.batch(0, 0.0) for _ in range(args.eval_batches)]
    probe = torch.from_numpy(train.gather(train.windows(cfg.T_ctx)[:8], cfg.T_ctx)[0])

    def make_lifetime_config(run_id: str, seed: int) -> LifetimeConfig:
        return LifetimeConfig(
            cfg=cfg, run_id=run_id, runs_root=args.out, device=args.device,
            corpus=train, eval_batches=eval_batches, probe_tokens=probe,
            L_star=L_star, slope_ref=slope_ref, rows_per_batch=rows,
            max_steps=args.steps, verbose=False, log_to_disk=True,
        )

    def solve_rate_fn(model, genome, params, n_eval=N_EVAL):
        return solve_rate(
            model, params, n_eval, np.random.default_rng(args.seeds[0]),
            genome=genome, T_ctx=cfg.T_ctx, device=args.device,
        )

    hyper = EsHyper()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    t0 = time.time()
    for arm in args.arms:
        kind = "reset each generation" if arm == "a4" else "inherited"
        print(f"\n=== arm {arm} ({kind}) ===")
        curves[arm] = run_generations(
            cfg, make_lifetime_config,
            n_generations=args.generations,
            seeds=args.seeds,
            arm=arm,
            hyper=hyper,
            reset_each_generation=(arm == "a4"),
            solve_rate_fn=solve_rate_fn,
            out_dir=str(out_dir),
            verbose=True,
        )
    wall = time.time() - t0

    n_lifetimes = len(args.arms) * len(args.seeds) * args.generations * hyper.P_es
    report = {
        "config": args.config,
        "arms": args.arms,
        "seeds": args.seeds,
        "generations": args.generations,
        "wall_seconds": wall,
        "n_lifetimes": n_lifetimes,
        "T_life_minutes_measured": (wall / 60.0) / max(1, n_lifetimes),
        "verdicts": {},
    }
    for arm, cs in curves.items():
        verdict = analyse_curves(cs, rng=np.random.default_rng(20260828))
        report["verdicts"][arm] = verdict.as_dict()
        print(f"\n  [{arm}] b_criterion={verdict.b_criterion:+.4f} "
              f"(sig={verdict.b_criterion_significant}) "
              f"b_slope={verdict.b_slope:+.4f} (sig={verdict.b_slope_significant}) "
              f"censored={verdict.censored_fraction:.2f} "
              f"interpretable={verdict.interpretable}")

    report["budget"] = budget_report(
        report["T_life_minutes_measured"], cfg,
        n_seed=len(args.seeds), n_generations=args.generations,
        P_es=hyper.P_es,
    )
    # **[NORMATIVE — K1 is the arm difference, audit A-82]** Each arm's own
    # curve is reported, but the verdict is `main - a4`, paired by seed. Reading
    # K1 off `main` alone declares success on an inheritance effect of zero when
    # both arms drift identically, and spends half the M3 budget on a control
    # that never enters the decision.
    if "main" in curves and "a4" in curves:
        d = analyse_arm_difference(
            curves["main"], curves["a4"], rng=np.random.default_rng(20260828)
        )
        report["inheritance_effect"] = d.as_dict()
        report["K1_fires"] = bool(d.fires)
        print(f"\n  [main - a4] b_criterion={d.b_criterion:+.4f} "
              f"(sig={d.b_criterion_significant}) b_slope={d.b_slope:+.4f} "
              f"(sig={d.b_slope_significant}) -> K1 fires = {d.fires}")
    else:
        report["K1_fires"] = None
        report["K1_not_evaluated"] = (
            "K1 is decided on main - a4 (§07.5, audit A-82); this run has "
            f"arms {sorted(curves)}. An arm's absolute curve is confounded by "
            "recipe drift and is not interpretable alone."
        )
        print(f"\n  K1 NOT evaluated: arms {sorted(curves)}; the verdict needs "
              f"both 'main' and 'a4'.")

    path = out_dir / "generation_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {path}")
    print(f"  measured T_life = {report['T_life_minutes_measured']:.2f} min "
          f"over {n_lifetimes} lifetimes")

    train.close()
    val.close()

    if args.assert_complete:
        ok = True
        for arm, cs in curves.items():
            if len(cs) != len(args.seeds):
                print(f"arm {arm}: {len(cs)} seeds, expected {len(args.seeds)}",
                      file=sys.stderr)
                ok = False
            for c in cs:
                if len(c.records) != args.generations:
                    print(f"arm {arm} seed {c.seed}: {len(c.records)} generations, "
                          f"expected {args.generations}", file=sys.stderr)
                    ok = False
            if not report["verdicts"][arm]["interpretable"]:
                frac = report["verdicts"][arm]["censored_fraction"]
                print(f"arm {arm}: censored fraction {frac:.2f} > 0.5; §07.5 makes "
                      f"the curve uninterpretable and L_star must be re-derived at "
                      f"the measured D_tok (audit A-34).", file=sys.stderr)
                ok = False
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
