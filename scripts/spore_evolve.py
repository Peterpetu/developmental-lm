"""Selection across lifetimes. SPEC: docs/spore/01-model.md §1.8, 02 K-EVO.

    python scripts/spore_evolve.py --name evo-5M --budget 5000000 --pop 8 --elites 2 --generations 10 --wall-hours 2
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from spore_common import RUNS, add_genome_args, genome_from_args, load_all, log_to, manifest, write_record  # noqa: E402

import torch  # noqa: E402

from spore.evolve import evolve  # noqa: E402
from spore.life import LifeConfig, live  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--elites", type=int, default=2)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--wall-hours", type=float, default=2.0)
    ap.add_argument("--evo-seed", type=int, default=0)
    ap.add_argument("--growth", default="select", choices=["select", "random", "off", "ridge"])
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--score-tokens", type=int, default=65536)
    ap.add_argument("--refine-lr", type=float, default=1e-2)
    add_genome_args(ap)
    args = ap.parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    base = genome_from_args(args)
    lin, corpus, ev_fit, ev_test = load_all()
    lives = out / "lives"
    lives.mkdir(exist_ok=True)

    def evaluate(g):
        cfg = LifeConfig(budget=args.budget, offset=args.offset, growth=args.growth, tf32=args.tf32, score_tokens=args.score_tokens, refine_lr=args.refine_lr)
        try:
            rec = live(g, cfg, shard=corpus.train, ev_fit=ev_fit, ev_test=ev_test, V=lin.vocab, log=log)
        except (RuntimeError, FloatingPointError, ValueError) as exc:      # an infeasible genome (memory, a failed solve)
            log(f"life FAILED for {g.digest()}: {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
            return {"fitness_ce": float("inf"), "test_ce": float("inf"), "failed": f"{type(exc).__name__}: {exc}"}
        write_record(lives / f"{g.digest()}.json", rec)
        torch.cuda.empty_cache()
        return {"fitness_ce": rec["fitness_ce"], "test_ce": rec["test_ce"], "F_final": rec["F_final"],
                "final_stage": rec["final_stage"], "seconds": rec["timings"]["total"]}

    hist = evolve(base, evaluate, pop=args.pop, elites=args.elites, generations=args.generations,
                  seed=args.evo_seed, out=out, wall_seconds=args.wall_hours * 3600, log=log)
    hist.update(manifest({"run": args.name, "model": "spore-evo", "budget": args.budget,
                          "eval_test_digest": ev_test.digest, "eval_fit_digest": ev_fit.digest}))
    write_record(out / "history.json", hist)
    log(f"done: {hist['n_lives']} lives, elite fitness {hist['elite']['fitness_ce']:.4f} test {hist['elite']['test_ce']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
