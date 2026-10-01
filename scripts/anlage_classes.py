#!/usr/bin/env python
"""R2 CLASSES: the data-class partition and what the free model already does across it.

SPEC: 12 §12.2
SPEC: 12 §12.9

    python scripts/anlage_classes.py [--checkpoint-dir runs/anlage/noise-tinystories-s0]

Writes the class manifest (M1), the classified pool's corpus mixture and digests (M4:
the classified pool holds the same windows as the plain pool), and, for every R1 seed
checkpoint found, the per-class CE on the fitness split: whether the classes differ in
loss at all, and by how much against `sigma_e`, before any mixture is varied.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import (CLASS_NAMES_TS, ClassifiedPool, classify_documents,  # noqa: E402
                         classify_eval_windows)
from anlage.lifetime import per_token_loss  # noqa: E402
from anlage.model import load_phenotype  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--checkpoint-dir", default="runs/anlage/noise-tinystories-s0")
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"
    lin = get_lineage(args.lineage)
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"classes-{lin.name}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    cls_train, man = classify_documents(corpus.train, corpus.rank)
    cls_fit, _ = classify_documents(corpus.fitness, corpus.rank, manifest=man)
    cls_test, _ = classify_documents(corpus.test, corpus.rank, manifest=man)
    log(f"R2 CLASSES  {lin.name}  rule {man['rule']}  length edges {man['length_edges']}  "
        f"rarity median {man['rarity_median']:.4f}  train doc counts {man['doc_counts']}")
    pool = ClassifiedPool(corpus.train, cls_train, man["n_classes"], n_windows=args.pool_windows,
                          T=lin.T, pool_seed=args.pool_seed, device=device)
    plain = corpus.pool(n_windows=args.pool_windows, T=lin.T, pool_seed=args.pool_seed)
    same_windows = pool.digest() == plain.digest()
    mu_c = pool.corpus_mixture
    log(f"  pool windows per class {pool.counts.tolist()}  corpus mixture {[round(x, 4) for x in mu_c.tolist()]}  "
        f"same windows as the plain pool: {same_windows}")
    # token mass per class in the training shard (documents' lengths)
    lengths = np.asarray(corpus.train.lengths, dtype=np.int64)
    mass = np.bincount(cls_train, weights=lengths, minlength=man["n_classes"])
    mass = (mass / mass.sum()).tolist()
    test, fit = eval_sets(corpus, lin, device=device)
    wc_fit = classify_eval_windows(fit, [corpus.fitness, corpus.test], [cls_fit, cls_test])
    wc_test = classify_eval_windows(test, [corpus.fitness, corpus.test], [cls_fit, cls_test])
    log(f"  fitness windows per class {torch.bincount(wc_fit, minlength=6).tolist()}  "
        f"test windows per class {torch.bincount(wc_test, minlength=6).tolist()}")
    per_seed = {}
    ckdir = pathlib.Path(args.checkpoint_dir)
    for ck in sorted(ckdir.glob("seed*.pt")):
        model = load_phenotype(ck, device=device)
        rows = {}
        for name, ev, wc in (("fitness", fit, wc_fit), ("test", test, wc_test)):
            tot = torch.zeros(6, dtype=torch.float64)
            cnt = torch.zeros(6, dtype=torch.float64)
            with torch.no_grad():
                for i in range(0, ev.windows.shape[0], 16):
                    w = ev.windows[i:i + 16]
                    ptl = per_token_loss(model, w).float().cpu()          # (b, T)
                    for j in range(w.shape[0]):
                        c = int(wc[i + j])
                        tot[c] += float(ptl[j].sum())
                        cnt[c] += ptl.shape[1]
            rows[name] = (tot / cnt.clamp_min(1)).tolist()
        per_seed[ck.stem] = rows
        log(f"  {ck.stem}: fitness per class {[round(x, 4) for x in rows['fitness']]}  "
            f"test per class {[round(x, 4) for x in rows['test']]}")
    noise_path = ckdir / "noise.json"
    sigma_e = json.loads(noise_path.read_text())["sigma_e"] if noise_path.exists() else None
    spread = None
    if per_seed:
        M = np.array([per_seed[k]["fitness"] for k in sorted(per_seed)])
        spread = {"class_mean": M.mean(0).tolist(), "class_range": float(M.mean(0).max() - M.mean(0).min()),
                  "seed_sd_per_class": M.std(0, ddof=1).tolist() if M.shape[0] > 1 else None}
        log(f"  class means (fitness) {[round(x, 4) for x in spread['class_mean']]}  range {spread['class_range']:.4f}"
            + (f"  = {spread['class_range'] / sigma_e:.1f} sigma_e" if sigma_e else ""))
    rec = {"run": "R2-CLASSES", "lineage": lin.name, "manifest_classes": man, "names": list(CLASS_NAMES_TS),
           "pool_counts": pool.counts.tolist(), "corpus_mixture": mu_c.tolist(), "token_mass": mass,
           "same_windows_as_plain_pool": same_windows, "pool_digest": pool.digest(),
           "class_digest": pool.class_digest(), "fitness_window_classes": torch.bincount(wc_fit, minlength=6).tolist(),
           "test_window_classes": torch.bincount(wc_test, minlength=6).tolist(), "per_seed": per_seed,
           "sigma_e": sigma_e, "spread": spread}
    write_json(out / "classes.json", manifest(rec))
    log(f"wrote {out / 'classes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
