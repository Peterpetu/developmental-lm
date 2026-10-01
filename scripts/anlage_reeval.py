#!/usr/bin/env python
"""Re-evaluate the saved checkpoints of R1, R2 and R3 on the v2 evaluation sets.

SPEC: 08 §8.2 (evaluation sets, v2)
SPEC: 12 §12.5

    python scripts/anlage_reeval.py

Writes, keeping the v1 files beside them as `*-v1.json`:
  * `runs/anlage/noise-tinystories-s0/noise.json`  -- `sigma_e`, `sigma_e(q)` on v2 from the four R1 adults;
  * `runs/anlage/classes-tinystories/classes.json` -- the per-class CE of the R1 adults on v2;
  * `runs/anlage/live-tinystories-s0/live.json`    -- every R3 arm re-scored on v2, the surrogate
    re-fit with held-one-out selection, and `mu*` re-solved.
The training runs themselves are untouched; only the measurement changes, and the change is
recorded in every manifest (`eval_version`).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import (EVAL_VERSION, REPO, eval_sets, get_lineage, load_corpus, log_to,  # noqa: E402
                           manifest, write_json)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import classify_documents, classify_eval_windows  # noqa: E402
from anlage.lifetime import evaluate, per_token_loss  # noqa: E402
from anlage.model import load_phenotype  # noqa: E402
from anlage.program import TAU, fit_surrogate, solve_trust_region  # noqa: E402
from screen.variance import pooled_within_sd  # noqa: E402


def per_class_ce(model, ev, wc, n_classes):
    tot = torch.zeros(n_classes, dtype=torch.float64)
    cnt = torch.zeros(n_classes, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, ev.windows.shape[0], 16):
            w = ev.windows[i:i + 16]
            ptl = per_token_loss(model, w).float().cpu()
            for j in range(w.shape[0]):
                c = int(wc[i + j])
                tot[c] += float(ptl[j].sum())
                cnt[c] += ptl.shape[1]
    return (tot / cnt.clamp_min(1)).tolist()


def _keep_v1(path: pathlib.Path) -> None:
    """Keep the file's current version beside it as `<stem>-v<version>.json`."""
    if not path.exists():
        return
    cur = json.loads(path.read_text()).get("eval_version", 1)
    old = path.with_name(path.stem + f"-v{cur}.json")
    if not old.exists():
        shutil.copy(path, old)


def main() -> int:
    device = "cuda"
    lin = get_lineage("tinystories")
    corpus = load_corpus(lin)
    test, fit = eval_sets(corpus, lin, device=device)
    assert EVAL_VERSION == 3
    log = log_to(REPO / "runs/anlage/reeval-log.txt")
    cls_train, man = classify_documents(corpus.train, corpus.rank)
    cls_fit, _ = classify_documents(corpus.fitness, corpus.rank, manifest=man)
    cls_test, _ = classify_documents(corpus.test, corpus.rank, manifest=man)
    wc_fit = classify_eval_windows(fit, [corpus.fitness, corpus.test], [cls_fit, cls_test])
    wc_test = classify_eval_windows(test, [corpus.fitness, corpus.test], [cls_fit, cls_test])
    n_C = man["n_classes"]
    log(f"REEVAL v{EVAL_VERSION}  test digest {test.digest}  fitness digest {fit.digest}  "
        f"test classes {torch.bincount(wc_test, minlength=n_C).tolist()}  fitness classes {torch.bincount(wc_fit, minlength=n_C).tolist()}")

    # ---- R1 -------------------------------------------------------------------------------
    ndir = REPO / "runs/anlage/noise-tinystories-s0"
    npath = ndir / "noise.json"
    _keep_v1(npath)
    noise = json.loads(npath.read_text())
    tests, fits, strata_t, strata_f, classes_f = [], [], [], [], []
    for seed in range(noise["seeds"]):
        m = load_phenotype(ndir / f"seed{seed}.pt", device=device)
        ce_t, st = evaluate(m, test)
        ce_f, sf = evaluate(m, fit)
        tests.append(ce_t); fits.append(ce_f); strata_t.append(st); strata_f.append(sf)
        classes_f.append(per_class_ce(m, fit, wc_fit, n_C))
        log(f"  R1 seed {seed}: test {ce_t:.4f}  fit {ce_f:.4f}")
    sd, df = pooled_within_sd([[x] for x in tests]) if False else (float(np.std(tests, ddof=1)), len(tests) - 1)
    S_t = np.array(strata_t, dtype=float); S_f = np.array(strata_f, dtype=float)
    noise.update({"eval_version": EVAL_VERSION, "final_test_ce": tests, "fitness_ce": fits,
                  "sigma_e": sd, "sigma_e_df": df, "mean_test_ce": float(np.mean(tests)),
                  "sigma_e_strata_test": [None if np.isnan(x) else float(x) for x in np.nanstd(S_t, axis=0, ddof=1)],
                  "sigma_e_strata_fit": [None if np.isnan(x) else float(x) for x in np.nanstd(S_f, axis=0, ddof=1)],
                  "stratum_counts_test": test.stratum_counts().tolist(),
                  "stratum_counts_fit": fit.stratum_counts().tolist(),
                  "eval_test_digest": test.digest, "eval_fit_digest": fit.digest,
                  "classes_fitness_per_seed": classes_f})
    write_json(npath, manifest(noise))
    log(f"  sigma_e(v{EVAL_VERSION}) = {sd:.4f} ({df} df)  mean {np.mean(tests):.4f}")

    # ---- R2 -------------------------------------------------------------------------------
    cpath = REPO / "runs/anlage/classes-tinystories/classes.json"
    _keep_v1(cpath)
    cj = json.loads(cpath.read_text())
    M = np.array(classes_f)
    cj.update({"eval_version": EVAL_VERSION, "per_seed": {f"seed{i}": {"fitness": classes_f[i]} for i in range(len(classes_f))},
               "fitness_window_classes": torch.bincount(wc_fit, minlength=n_C).tolist(),
               "test_window_classes": torch.bincount(wc_test, minlength=n_C).tolist(), "sigma_e": sd,
               "spread": {"class_mean": M.mean(0).tolist(), "class_range": float(M.mean(0).max() - M.mean(0).min()),
                          "seed_sd_per_class": M.std(0, ddof=1).tolist()}})
    write_json(cpath, manifest(cj))
    log(f"  R2 class means (fitness, v2) {[round(x, 4) for x in M.mean(0).tolist()]}  range {cj['spread']['class_range']:.4f}")

    # ---- R3 -------------------------------------------------------------------------------
    lpath = REPO / "runs/anlage/live-tinystories-s0/live.json"
    _keep_v1(lpath)
    lv = json.loads(lpath.read_text())
    for a in lv["arms"]:
        m = load_phenotype(a["checkpoint"], device=device)
        ce_t, _ = evaluate(m, test)
        ce_f, strata = evaluate(m, fit)
        a.update({"test_ce": ce_t, "fitness_ce": ce_f, "strata": strata, "classes": per_class_ce(m, fit, wc_fit, n_C)})
    centre = [a for a in lv["arms"] if a["label"].startswith("centre")]
    S = np.array([a["strata"] for a in centre], dtype=float)
    Cc = np.array([a["classes"] for a in centre], dtype=float)
    sigma_q = np.nanstd(S, axis=0, ddof=1); sigma_c = Cc.std(axis=0, ddof=1)
    sigma_test = float(np.std([a["test_ce"] for a in centre], ddof=1))
    log(f"  R3 protocol noise (v2): test {sigma_test:.4f}  strata {[round(float(x), 4) for x in sigma_q]}")
    live = {}
    for alpha in lv["alphas"]:
        recs = [a for a in lv["arms"] if a["label"].startswith(f"a{alpha:g} ")]
        S_a = np.array([r["strata"] for r in recs], dtype=float); C_a = np.array([r["classes"] for r in recs], dtype=float)
        dev_q = np.nanmax(np.abs(S_a - S[0]), axis=0); dev_c = np.abs(C_a - Cc[0]).max(axis=0)
        ratio_q = dev_q / np.where(sigma_q > 0, sigma_q, np.nan); ratio_c = dev_c / np.where(sigma_c > 0, sigma_c, np.nan)
        spread_test = float(np.max(np.abs(np.array([r["test_ce"] for r in recs]) - centre[0]["test_ce"])))
        best = min(recs, key=lambda r: r["test_ce"])
        live[f"{alpha:g}"] = {"max_dev_strata": dev_q.tolist(), "ratio_strata": ratio_q.tolist(),
                              "max_dev_classes": dev_c.tolist(), "ratio_classes": ratio_c.tolist(),
                              "spread_test": spread_test, "spread_test_sigma": spread_test / sigma_test if sigma_test > 0 else None,
                              "alive_3sigma": bool(np.nanmax(ratio_q) >= 3.0),
                              "best_sibling": best["label"], "best_sibling_gain": best["test_ce"] - centre[0]["test_ce"]}
        log(f"  alpha {alpha:g}: stratum ratios {[round(float(x), 1) for x in ratio_q]}  test spread {spread_test:.4f} "
            f"({spread_test / sigma_test:.1f} sigma)  best sibling {best['label']} {best['test_ce'] - centre[0]['test_ce']:+.4f}  alive {live[f'{alpha:g}']['alive_3sigma']}")
    eta_live = {f"{a['lr_mult']:g}": {"d_test": a["test_ce"] - centre[0]["test_ce"],
                                       "d_test_sigma": (a["test_ce"] - centre[0]["test_ce"]) / sigma_test if sigma_test > 0 else None}
                for a in lv["arms"] if a["lr_mult"] != 1.0}
    mus = torch.tensor([a["mu"] for a in lv["arms"] if a["lr_mult"] == 1.0], dtype=torch.float64)
    F = torch.tensor([[x if (x is not None and x == x) else 0.0 for x in a["strata"]] for a in lv["arms"] if a["lr_mult"] == 1.0], dtype=torch.float64)
    w = torch.tensor(noise["stratum_counts_test"], dtype=torch.float64); w = w / w.sum()
    mu_0 = torch.tensor(lv["mu_0"], dtype=torch.float64)
    sur = fit_surrogate(mus, F, w_q=w, select="loo")
    lin_s = fit_surrogate(mus, F, w_q=w, force_linear=True)
    mu_star, info = solve_trust_region(sur, w, mu_0, tau=TAU)
    log(f"  surrogate: {'linear' if sur.linear_only else 'quadratic'} (LOO on the weighted objective {sur.loo_rmse:.4f}; "
        f"linear {lin_s.loo_rmse:.4f}) vs protocol noise {sigma_test:.4f};  mu* {[round(x, 4) for x in mu_star.tolist()]}  "
        f"predicted {info['value']:.4f}  distance {info['distance']:.3f}  boundary {info['on_boundary']}")
    lv.update({"eval_version": EVAL_VERSION, "sigma_test": sigma_test, "sigma_strata": sigma_q.tolist(), "sigma_classes": sigma_c.tolist(),
               "live_mu": live, "live_eta": eta_live,
               "surrogate": {"linear_only": sur.linear_only, "n_samples": sur.n_samples, "loo_rmse": sur.loo_rmse,
                             "loo_rmse_linear": lin_s.loo_rmse, "selection": "loo on the w_q-weighted objective"},
               "mu_star": mu_star.tolist(), "mu_star_info": info, "stratum_weights": w.tolist(),
               "eval_test_digest": test.digest, "eval_fit_digest": fit.digest})
    write_json(lpath, manifest(lv))
    log(f"wrote noise.json, classes.json, live.json (v{EVAL_VERSION}); the previous version kept beside each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
