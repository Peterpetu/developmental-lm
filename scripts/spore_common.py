"""Shared plumbing for the `spore_*` scripts: corpus, evaluation sets, manifests, records.
SPEC: docs/spore/02-protocol.md §2.5."""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from anlage_common import eval_sets, get_lineage, load_corpus, log_to, manifest  # noqa: E402
from spore.genome import Genome  # noqa: E402

__all__ = ["REPO", "RUNS", "manifest", "load_all", "write_record", "genome_from_args", "add_genome_args", "log_to"]

RUNS = REPO / "runs" / "spore"


def load_all(device: str = "cuda", n_test: int = 256, n_fit: int = 256):
    lin = get_lineage("tinystories")
    corpus = load_corpus(lin)
    ev_test, ev_fit = eval_sets(corpus, lin, n_test=n_test, n_fit=n_fit, device=device)
    return lin, corpus, ev_fit, ev_test


def add_genome_args(ap) -> None:
    g = Genome()
    ap.add_argument("--genome-json", default=None, help="a genome dict (file path) overriding the defaults")
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--orders", default=None, help="comma-separated, e.g. 1,2,3,4,6")
    ap.add_argument("--gamma-b", type=float, default=None)
    ap.add_argument("--gamma-e", type=float, default=None)
    ap.add_argument("--n-k", type=int, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--d-k", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--E", type=int, default=None)
    ap.add_argument("--waves", default=None, help="comma-separated fractions, or 'none'")
    ap.add_argument("--k-wave", type=int, default=None)
    ap.add_argument("--cand-mult", type=int, default=None)
    ap.add_argument("--psi", default=None)
    ap.add_argument("--k-w", type=int, default=None)
    ap.add_argument("--eta-ri", type=float, default=None)
    ap.add_argument("--onehot", action="store_true")
    ap.add_argument("--eta-sem", type=float, default=None)
    ap.add_argument("--reservoir", action="store_true")
    ap.add_argument("--lro", action="store_true")
    ap.add_argument("--rho-res", type=float, default=None)
    ap.add_argument("--d-sem", type=int, default=None)
    ap.add_argument("--sem-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0, help="atom seed s_A; the other seeds are offsets of it")


def genome_from_args(args) -> Genome:
    d = Genome().as_dict()
    if args.genome_json:
        d.update(json.loads(pathlib.Path(args.genome_json).read_text(encoding="utf-8")))
    for k in ("D", "gamma_b", "gamma_e", "n_k", "beta", "d_k", "lam", "E", "k_wave", "cand_mult", "psi", "k_w", "eta_ri",
              "eta_sem", "d_sem", "sem_frac", "rho_res"):
        v = getattr(args, k, None)
        if v is not None:
            d[k] = v
    if getattr(args, "onehot", False):
        d["onehot"] = True
    if getattr(args, "reservoir", False):
        d["reservoir"] = True
    if getattr(args, "lro", False):
        d["lro"] = True
    if args.orders:
        d["orders"] = [int(x) for x in args.orders.split(",")]
    if args.waves is not None:
        d["waves"] = [] if args.waves == "none" else [float(x) for x in args.waves.split(",")]
    if args.seed is not None and not args.genome_json:
        d["s_A"], d["s_rho"], d["s_M"], d["s_U"] = args.seed, args.seed + 1, args.seed + 2, args.seed + 3
    return Genome.from_dict(d)


def write_record(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
