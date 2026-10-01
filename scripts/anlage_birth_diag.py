#!/usr/bin/env python
"""Where does the at-birth loss go? Role-wise and layer-wise swaps between the
trained phenotype and its genome decode.

SPEC: 05 §5.6
SPEC: 08 §8.5

    python scripts/anlage_birth_diag.py --genome <birth>/genome.pt --checkpoint seed0.pt

For each role `m` (and each layer `l`) the decoded phenotype `D` is evaluated with
that role's (layer's) trained weights restored (`D <- W` on the role) and, in the
other direction, the trained phenotype with that role decoded (`W <- D`). The
two tables locate the roles whose decode costs the function, which the per-role
Frobenius residual alone does not tell (a residual of 0.5 on a norm gain and on
the embedding are not the same loss).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, write_json  # noqa: E402

import torch  # noqa: E402

from anlage.codes import CodeTables  # noqa: E402
from anlage.config import ROLE_NAMES, GenomeShape, StageConfig  # noqa: E402
from anlage.genome import DecodeContext, Genome  # noqa: E402
from anlage.latents import LatentBank  # noqa: E402
from anlage.lifetime import evaluate  # noqa: E402
from anlage.model import Phenotype, load_phenotype  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    lin = get_lineage(args.lineage)
    corpus = load_corpus(lin)
    test, _ = eval_sets(corpus, lin, device=device)
    blob = torch.load(args.genome, map_location="cpu")
    shape = GenomeShape(**blob["shape"])
    if "cfg" in blob:
        cfg_d = dict(blob["cfg"])
        if cfg_d.get("ops_override") is not None:
            cfg_d["ops_override"] = tuple(cfg_d["ops_override"])
        cfg = StageConfig(**cfg_d)
    else:
        cfg = lin.stage(args.stage)
    G = Genome(shape, seed=1).to(device)
    G.load_state_dict(blob["genome"])
    bank = LatentBank(shape, cfg, lin.vocab).to(device)
    bank.load_state_dict(blob["bank"])
    model = load_phenotype(args.checkpoint, device=device)
    tables = CodeTables(shape, vocab=lin.vocab, rank=corpus.rank, L=cfg.L, device=device)
    ctx = DecodeContext(cfg, tables, bank, lin.vocab, tied=model.tied)
    with torch.no_grad():
        D = G.decode_all(ctx)
    W = {k: v.detach() for k, v in model.weights().items()}
    out = pathlib.Path(args.out or pathlib.Path(args.genome).parent)
    log = log_to(out / "diag.txt")

    def ce_of(weights):
        m = Phenotype(cfg, lin.vocab, tied=model.tied, weights=weights).to(device)
        return evaluate(m, test)[0]

    ce_W, ce_D = ce_of(W), ce_of(D)
    log(f"trained {ce_W:.4f}   decoded {ce_D:.4f}")
    roles = sorted({k[0] for k in W})
    rows = {}
    log("role-wise: D with the role's trained weights restored | W with the role decoded")
    for m in roles:
        keys = [k for k in W if k[0] == m]
        hyb_D = dict(D); hyb_W = dict(W)
        for k in keys:
            hyb_D[k] = W[k]
            hyb_W[k] = D[k]
        a, b = ce_of(hyb_D), ce_of(hyb_W)
        res = float(sum(((D[k] - W[k]).float().pow(2).sum()) for k in keys)
                    / sum(W[k].float().pow(2).sum() for k in keys))
        rows[ROLE_NAMES[m]] = {"restore_in_D": a, "decode_in_W": b, "residual": res}
        log(f"  {ROLE_NAMES[m]:>12}: D<-W {a:7.4f} ({a - ce_D:+.3f})   W<-D {b:7.4f} ({b - ce_W:+.3f})   residual {res:.3f}")
    layers = {}
    log("layer-wise: W with one layer decoded")
    for l in range(cfg.L):
        hyb_W = dict(W)
        for k in W:
            if k[1] == l:
                hyb_W[k] = D[k]
        b = ce_of(hyb_W)
        layers[l] = b
        log(f"  layer {l} ({cfg.op(l)}): W<-D {b:7.4f} ({b - ce_W:+.3f})")
    # the wiring alone: every matrix decoded, the lexicon (EMB/HEAD) and vectors trained
    from anlage.config import ROLE_EMB, ROLE_HEAD_OUT
    from anlage.genome import MAT_INDEX
    hyb = dict(W)
    for k in W:
        if k[0] in MAT_INDEX and k[0] not in (ROLE_EMB, ROLE_HEAD_OUT):
            hyb[k] = D[k]
    ce_wiring = ce_of(hyb)
    hyb = dict(W)
    for k in W:
        if k[0] in (ROLE_EMB, ROLE_HEAD_OUT):
            hyb[k] = D[k]
    ce_lexicon = ce_of(hyb)
    log(f"all wiring matrices decoded, lexicon + vectors trained: {ce_wiring:.4f}")
    log(f"lexicon decoded, everything else trained:               {ce_lexicon:.4f}")
    write_json(out / "diag.json", {"trained": ce_W, "decoded": ce_D, "roles": rows,
                                   "layers": layers, "wiring_decoded": ce_wiring,
                                   "lexicon_decoded": ce_lexicon})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
