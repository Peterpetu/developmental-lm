"""Render `docs/tempo/RESULTS.md`, the register, from the run records and the analysis.

SPEC: 10 §10.4  (one row per number with its run directory, seed, budget and criterion;
                 a verdict line per criterion; negative results kept with their artifacts)

    python scripts/tempo_register.py [--out docs/tempo/RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tempo_common import REPO, RUNS
from tempo_analyse import analyse, load_runs, final_test, rewards, profiles


def row(name: str, seed: int, rec: dict) -> str:
    if "episodes" in rec:
        eps = rec["episodes"]
        n_ep = len([e for e in eps if "L_check" in e])
        last = eps[-1] if eps else {}
        test = final_test(rec)
        tps = last.get("tokens_per_s", float("nan"))
        mem = last.get("peak_mem_gb", float("nan"))
        lam = [p["lambda_block"] for p in profiles(rec) if p["lambda_block"] is not None]
        rw = rewards(rec)
        F = last.get("F_cum", float("nan"))
        return (f"| `{name}` | {seed} | {n_ep} ep × N={rec['life']['N']} B={rec['life']['B']}, F {F:.3e} | "
                f"test {test if test is not None else float('nan'):.4f} | "
                f"reward {np.mean(rw) if rw else float('nan'):+.4f} | "
                f"Λ/K [{min(lam) if lam else float('nan'):+.3f}, {max(lam) if lam else float('nan'):+.3f}] | "
                f"{tps:.0f} tok/s, {mem:.2f} GB | `runs/tempo/{name}/record.json` |")
    if "ceiling_nll_per_token" in rec:
        return (f"| `{name}` | {seed} | {rec['tokens']:,} tokens | ceiling {rec['ceiling_nll_per_token']:.5f} | "
                f"noise gap {rec['noise_gap_per_token']:.5f} (I1 {rec['I1']}, I2 {rec['I2']}) | — | "
                f"{rec['seconds']:.0f} s | `runs/tempo/{name}/result.json` |")
    if "test_nll_targets_K_to_TK" in rec:
        tr = rec["train_result"]
        return (f"| `{name}` | {seed} | {rec['tokens']:,} tokens (span agent {rec.get('tokens_span_agent', rec['tokens']):,}), "
                f"width {rec['width']}{', VE' if rec.get('use_ve') else ''} | "
                f"test {rec['test_nll_targets_K_to_TK']:.4f} | training-FLOP ratio {rec['flop_ratio']:.3f} | — | "
                f"{tr['tokens_per_s']:.0f} tok/s, {tr['peak_mem_gb']:.2f} GB | `runs/tempo/{name}/result.json` |")
    if "mlp_return" in rec:
        return (f"| `{name}` | {seed} | {rec['rollouts']} rollouts | MLP return {rec['mlp_return']:+.3f} | "
                f"agent {rec.get('agent_return', float('nan')):+.3f} | — | "
                f"{rec['seconds']:.0f} s | `runs/tempo/{name}/result.json` |")
    return f"| `{name}` | {seed} | — | — | — | — | — | `runs/tempo/{name}/` |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "tempo" / "RESULTS.md"))
    ap.add_argument("--config", default="ts")
    args = ap.parse_args()
    runs = load_runs(RUNS)
    rep = analyse(runs, args.config)
    lines = ["# TEMPO — the register", "",
             f"Rendered by `scripts/tempo_register.py` on {time.strftime('%Y-%m-%d %H:%M')} from `runs/tempo/`; "
             "every number carries its artifact. SPEC: 10 §10.4. Regenerate after every run; never edit by hand.", "",
             "## Verdicts (10 §10.2, §10.5)", ""]
    lines += ["```"] + rep["lines"] + ["```", ""]
    lines += ["## Runs", "",
              "| run | seed | budget | score | second | per-block Λ range | throughput, memory | artifact |",
              "|---|---|---|---|---|---|---|---|"]
    order = sorted(runs, key=lambda k: (k[0] != "codec", k[0], k[1]))
    lines.append("| *(every life row is a complete run; unfinished lives are not listed)* | | | | | | | | |")
    for key in order:
        for seed in sorted(runs[key]):
            name = (f"codec-{key[1]}-s{seed}" if key[0] == "codec"
                    else f"{key[0]}-{key[1]}-{key[2]}-s{seed}")
            lines.append(row(name, seed, runs[key][seed]))
    lines += ["", "## Numbers", "", "```", json.dumps(rep["numbers"], indent=1, default=float)[:20000], "```", ""]
    pathlib.Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}: {sum(len(v) for v in runs.values())} run rows")


if __name__ == "__main__":
    main()
