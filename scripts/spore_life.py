"""One SPORE lifetime, recorded. SPEC: docs/spore/01-model.md, 02 §2.5.

    python scripts/spore_life.py --name base-5M --budget 5000000
    python scripts/spore_life.py --name nobind-5M --budget 5000000 --no-binding
    python scripts/spore_life.py --name nomem-5M --budget 5000000 --no-memory
    python scripts/spore_life.py --name growoff-5M --budget 5000000 --growth off
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# cuBLAS chooses split-K reductions by workspace unless this is set before CUDA initialises;
# G-DET (03 §P-S.26/28) needs bit-reproducible GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from spore_common import RUNS, add_genome_args, genome_from_args, load_all, log_to, manifest, write_record  # noqa: E402

import torch  # noqa: E402

from spore.life import LifeConfig, live  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--no-binding", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--growth", default="select", choices=["select", "random", "off", "ridge"])
    ap.add_argument("--readout", default="ridge", choices=["ridge", "sketch"])
    ap.add_argument("--refine", type=int, default=None)
    ap.add_argument("--refine-lr", type=float, default=5e-3)
    ap.add_argument("--score-tokens", type=int, default=2 ** 18)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--n-eval", type=int, default=256)
    add_genome_args(ap)
    args = ap.parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    g = genome_from_args(args)
    cfg = LifeConfig(budget=args.budget, offset=args.offset, batch_windows=args.batch_windows,
                     binding=not args.no_binding, memory=not args.no_memory, growth=args.growth,
                     readout=args.readout, refine=args.refine, refine_lr=args.refine_lr,
                     score_tokens=args.score_tokens, tf32=args.tf32)
    lin, corpus, ev_fit, ev_test = load_all(n_test=args.n_eval, n_fit=args.n_eval)
    log(f"genome {g.digest()} {g.as_dict()}")
    log(f"cfg {cfg}")
    rec = live(g, cfg, shard=corpus.train, ev_fit=ev_fit, ev_test=ev_test, V=lin.vocab, log=log)
    rec.update(manifest({"run": args.name, "model": "spore", "eval_test_digest": ev_test.digest,
                         "eval_fit_digest": ev_fit.digest, "corpus": corpus.manifest()}))
    write_record(out / "life.json", rec)
    log(f"done: final stage {rec['final_stage']}  fitness {rec['fitness_ce']:.4f}  test {rec['test_ce']:.4f}  "
        f"{rec['timings']['total']:.0f} s  peak {rec['peak_mem_gb']:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
