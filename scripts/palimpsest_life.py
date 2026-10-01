"""One life. SPEC: docs/palimpsest/01-model.md §1.6, 02 §2.5.

    python scripts/palimpsest_life.py --name shift-copy-s0 --life shift --arm copy --seed 0
    python scripts/palimpsest_life.py --name shift-sieve-c8-s0 --life shift --arm sieve --core c8 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from anlage_common import log_to, manifest, write_json  # noqa: E402

import torch  # noqa: E402

from palimpsest.envs import Environment, phases_for  # noqa: E402
from palimpsest.life import ARMS, LifeConfig, run_life  # noqa: E402

RUNS = REPO / "runs" / "palimpsest"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--life", default="shift", choices=["shift", "stat", "shift3", "statcode"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--core", default="c8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--distil-tokens", type=int, default=2_000_000)
    ap.add_argument("--dream-tokens", type=int, default=500_000)
    ap.add_argument("--T-dream", type=int, default=256)
    ap.add_argument("--lam-dream", type=float, default=1.0)
    ap.add_argument("--core-init", default="prev", choices=["prev", "shrink"])
    ap.add_argument("--dream-start", default="mixed", choices=["eos", "uniform", "mixed"])
    ap.add_argument("--new-units", default="copy", choices=["copy", "fresh"])
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--eps-gain", type=float, default=0.02)
    ap.add_argument("--lam-ewc", type=float, default=1e3)
    ap.add_argument("--lam-lwf", type=float, default=1.0)
    ap.add_argument("--replay-frac", type=float, default=0.1)
    ap.add_argument("--replay-tokens", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=256)
    args = ap.parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = LifeConfig(arm=args.arm, core=args.core, tokens_per_phase=args.tokens,
                     distil_tokens=args.distil_tokens, dream_tokens=args.dream_tokens, T_dream=args.T_dream,
                     lam_dream=args.lam_dream, core_init=args.core_init, dream_start=args.dream_start, new_units=args.new_units,
                     noise=args.noise, eps_gain=args.eps_gain, lam_ewc=args.lam_ewc, lam_lwf=args.lam_lwf,
                     replay_frac=args.replay_frac, replay_tokens=args.replay_tokens)
    spec = phases_for(args.life, args.seed, K=args.K)
    t0 = time.time()
    envs = {e: Environment(e, n_eval=args.n_eval) for e in spec.envs}
    log(f"environments {list(envs)} loaded in {time.time() - t0:.0f}s; phases {spec.as_list()}")
    rec = run_life(spec, cfg, seed=args.seed, envs=envs, log=log)
    rec["life"] = args.life
    rec.update(manifest({"run": args.name, "model": "palimpsest", "environments": {e: v.manifest() for e, v in envs.items()}}))
    write_json(out / "record.json", rec)
    sc = rec["scores"]
    log(f"done in {rec['seconds']:.0f}s: U_K {sc['U_K']:.4f}  Lambda {sc.get('Lambda', float('nan')):.4f}  "
        + (f"G_K {sc['G_K']:.4f}  " if "G_K" in sc else "") + "final " + "  ".join(f"{e} {c:.4f}" for e, c in sc["final"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
