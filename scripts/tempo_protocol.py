"""The protocol as one serial chain: R0 -> R1 -> R2 -> R3 -> R4 -> R5 -> R6 -> R7 -> R8 -> R9 -> R10.

SPEC: 10 §10.3  (the runs, budgets, seeds), §10.1 (one chain per card; no missing run is a pass)

    python scripts/tempo_protocol.py --through R4 [--seeds 4] [--N 250] [--B 32] [--dry-run]

Idempotent: a run whose `record.json` says `"complete": true` (or whose `result.json` exists)
is skipped, so the chain can be resumed after an interruption. Every child is a separate
process (one GPU job at a time); `tempo_analyse.py` prints the verdicts at the end.

**Pausing.** Creating the file `runs/tempo/PAUSE` stops the chain at the next step of the
running life: that life writes its checkpoint and a record marked `"paused": true`, exits with
code 42, and the chain stops without touching anything else. Removing the file and re-running
the same command continues: the paused life reloads its checkpoint and resumes **inside** the
episode it was in, at the step it had reached, without re-running the boundary. The rest of
the chain then proceeds as if nothing happened. A run that was interrupted some other way (a
crash, a reboot) has no paused record and is simply re-run from its start.

    python scripts/tempo_protocol.py --pause      # ask the running chain to stop, then wait
    python scripts/tempo_protocol.py --resume ... # clear the request and continue the chain
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

from tempo_common import REPO, RUNS
from tempo.modalities import STREAMS

PY = sys.executable
S = REPO / "scripts"
ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
PAUSE_FILE = RUNS / "PAUSE"
PAUSE_EXIT = 42


class ChainPaused(Exception):
    pass


def done(name: str) -> bool:
    d = RUNS / name
    if (d / "result.json").exists():
        return True
    rec = d / "record.json"
    if rec.exists():
        try:
            return bool(json.loads(rec.read_text(encoding="utf-8")).get("complete"))
        except json.JSONDecodeError:
            return False
    return False


def run(cmd: list[str], name: str, dry: bool) -> None:
    if done(name):
        print(f"[skip] {name}")
        return
    if PAUSE_FILE.exists():
        raise ChainPaused(name)
    print(f"[run ] {name}: {' '.join(cmd)}", flush=True)
    if dry:
        return
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode == PAUSE_EXIT:
        raise ChainPaused(name)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    print(f"[done] {name} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", default="R10", choices=ORDER)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--N", type=int, default=250)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--r1-tokens", type=int, default=25_000_000)
    ap.add_argument("--codec-tokens", type=int, default=200_000_000,
                    help="R0's budget; 50M gave a ceiling of 0.0126 > 0.01 (I1), kept as codec-ts-s0-tokens50M")
    ap.add_argument("--morph-budget", type=float, default=40.0, help="R6: primordium-episodes per arm (K-MORPH at equal FLOPs)")
    ap.add_argument("--sota-episodes", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="ts", help="ts | tsk2 | tsk1: the K of every life stage")
    ap.add_argument("--kcont", default="tsk2,tsk1", help="the K-CONT fallback configs whose R0/R1/R2 run inside R2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pause", action="store_true",
                    help="ask the running chain to stop at the next step and exit (writes runs/tempo/PAUSE)")
    ap.add_argument("--resume", action="store_true", help="remove the pause request and run the chain")
    args = ap.parse_args()
    if args.pause:
        PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FILE.write_text(f"paused {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
        print(f"[pause] wrote {PAUSE_FILE}; the running life stops at its next step and checkpoints")
        return
    if args.resume and PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
        print(f"[resume] removed {PAUSE_FILE}")
    if PAUSE_FILE.exists():
        print(f"[pause] {PAUSE_FILE} exists: nothing runs until it is removed (--resume clears it)")
        return
    last = ORDER.index(args.through)
    dev = ["--device", args.device]
    cfgn = args.config
    C = ["--config", cfgn]
    # estimator 2's probes are a tenth of the episode (06 §6.6: n_probe = 100 at N = 1000)
    life = [PY, str(S / "tempo_life.py"), "--n-probe-steps", str(max(5, args.N // 10))]
    N1 = max(1, args.r1_tokens // (args.B * 512))
    # 11 §P-D.2 / work order D.5: the nominal product is wrong whenever G-STAT doubles N, which
    # it may do once per life -- the TEMPO arm would then consume up to twice the tokens the
    # FLOP-matched baseline is given. The baseline is matched to the arm's own record instead.
    sota_tokens = args.sota_episodes * args.N * args.B * 512

    def stages():
        # R0
        yield "R0", [PY, str(S / "tempo_codec.py"), "--tokens", str(args.codec_tokens), "--seed", "0"] + C + dev, f"codec-{cfgn}-s0"
        # R1: the span agent at the primordium, gamma = 0, one episode of r1 tokens, 4 seeds, the
        # registered text loss (nolat) and the latent-term ablation (lat01) on the default K
        for arm in (("nolat", "lat01") if cfgn == "ts" else ("nolat",)):
            for s in range(args.seeds):
                yield "R1", life + ["--run", "R1", "--arm", arm, "--seed", str(s), "--episodes", "1",
                                    "--N", str(N1), "--B", str(args.B), "--eval-every", str(max(1, N1 // 5))] + C + dev, f"R1-{arm}-{cfgn}-s{s}"
        # R2: the token baseline at matched training FLOPs; then the K-CONT fallbacks (K = 2, K = 1):
        # their codec, their R1 and their own FLOP-matched token baselines
        for s in range(args.seeds):
            yield "R2", [PY, str(S / "tempo_cont.py"), "--tokens", str(args.r1_tokens), "--seed", str(s),
                         "--span-record", str(RUNS / f"R1-nolat-{cfgn}-s{s}" / "record.json")] + C + dev, f"R2-token-{cfgn}-s{s}"
        for kc in [k for k in args.kcont.split(",") if k and k != cfgn]:
            KC = ["--config", kc]
            yield "R2", [PY, str(S / "tempo_codec.py"), "--tokens", str(args.codec_tokens), "--seed", "0"] + KC + dev, f"codec-{kc}-s0"
            for s in range(args.seeds):
                yield "R2", life + ["--run", "R1", "--arm", "nolat", "--seed", str(s), "--episodes", "1",
                                    "--N", str(N1), "--B", str(args.B), "--eval-every", str(max(1, N1 // 5))] + KC + dev, f"R1-nolat-{kc}-s{s}"
            for s in range(args.seeds):
                yield "R2", [PY, str(S / "tempo_cont.py"), "--tokens", str(args.r1_tokens), "--seed", str(s),
                             "--span-record", str(RUNS / f"R1-nolat-{kc}-s{s}" / "record.json")] + KC + dev, f"R2-token-{kc}-s{s}"
        # R3: a 20-episode life, gamma = 0, rule off: the profile
        for s in range(2):
            yield "R3", life + ["--run", "R3", "--arm", "a_gamma0", "--seed", str(s), "--episodes", str(args.episodes),
                                "--N", str(args.N), "--B", str(args.B)] + C + dev, f"R3-a_gamma0-{cfgn}-s{s}"
        # R4: K-GENE arms with estimator 2
        for arm, n_seeds in (("tempo", args.seeds), ("a_gamma0", args.seeds), ("b_homeo", args.seeds),
                             ("c_random", args.seeds), ("d_free", args.seeds), ("e_svf", 2), ("f_es", 2), ("g_grpo", 2)):
            for s in range(n_seeds):
                est = ["--estimator", "restem"] if arm in ("tempo", "e_svf") else []
                yield "R4", life + ["--run", "R4", "--arm", arm, "--seed", str(s), "--episodes", str(args.episodes),
                                    "--N", str(args.N), "--B", str(args.B)] + est + C + dev, f"R4-{arm}-{cfgn}-s{s}"
        # R5: K-REV and K-SAL against R4's full arm (same seeds, same windows: CRN)
        for arm in ("forward", "uniform"):
            for s in range(args.seeds):
                yield "R5", life + ["--run", "R5", "--arm", arm, "--seed", str(s), "--episodes", str(args.episodes),
                                    "--N", str(args.N), "--B", str(args.B), "--estimator", "restem"] + C + dev, f"R5-{arm}-{cfgn}-s{s}"
        # R6: K-MORPH at equal FLOPs: every arm stops at the same budget of primordium-episodes
        for arm in ("tempo", "schedule", "oneshot", "flat"):
            for s in range(args.seeds):
                yield "R6", life + ["--run", "R6", "--arm", arm, "--seed", str(s), "--episodes", "999",
                                    "--N", str(args.N), "--B", str(args.B), "--estimator", "restem",
                                    "--F-budget", str(args.morph_budget)] + C + dev, f"R6-{arm}-{cfgn}-s{s}"
        # R7: transfer (the verifier tasks) from R4's life, genome kept vs reset
        for arm, extra in (("tempo", []), ("reset", ["--reset-genome"])):
            for s in range(args.seeds):
                yield "R7", life + ["--run", "R7", "--arm", "tempo", "--tag", arm, "--seed", str(s), "--episodes", "1",
                                    "--N", str(args.N), "--B", str(args.B), "--streams", "tasks",
                                    "--load", str(RUNS / f"R4-tempo-{cfgn}-s{s}" / "life.pt"),
                                    "--eval-every", str(max(1, args.N // 10))] + extra + C + dev, f"R7-tempo-{arm}-{cfgn}-s{s}"
        # R8: the blend, the stream-specific agents at the uniform per-modality share, G-ACT
        for s in range(args.seeds):
            yield "R8", life + ["--run", "R8", "--arm", "tempo", "--seed", str(s), "--episodes", str(args.episodes),
                                "--N", str(args.N), "--B", str(args.B), "--streams", "all", "--estimator", "restem"] + C + dev, f"R8-tempo-{cfgn}-s{s}"
        for st in STREAMS:
            for s in range(args.seeds):
                yield "R8", life + ["--run", "R8", "--arm", "tempo", "--tag", f"only{st}", "--seed", str(s),
                                    "--episodes", str(max(1, round(args.episodes / len(STREAMS)))),
                                    "--N", str(args.N), "--B", str(args.B), "--streams", f"only-{st}",
                                    "--estimator", "restem"] + C + dev, f"R8-tempo-only{st}-{cfgn}-s{s}"
        for s in range(3):
            yield "R8", [PY, str(S / "tempo_gact.py"), "--episodes", str(args.episodes), "--seed", str(s),
                         "--N", str(args.N), "--B", str(args.B),
                         "--life", str(RUNS / f"R8-tempo-{cfgn}-s{min(s, args.seeds - 1)}" / "life.pt")] + C, f"GACT-mlp-{cfgn}-s{s}"
        # R9: cross-modal transfer from the text life to the blend, genome kept vs reset
        for arm, extra in (("tempo", []), ("reset", ["--reset-genome"])):
            for s in range(args.seeds):
                yield "R9", life + ["--run", "R9", "--arm", "tempo", "--tag", arm, "--seed", str(s), "--episodes", "1",
                                    "--N", str(args.N), "--B", str(args.B), "--streams", "all",
                                    "--load", str(RUNS / f"R4-tempo-{cfgn}-s{s}" / "life.pt"),
                                    "--eval-every", str(max(1, args.N // 10))] + extra + C + dev, f"R9-tempo-{arm}-{cfgn}-s{s}"
        # R10: K-SOTA at the adult budget against the FLOP-matched token baseline at the same tokens
        for s in range(2):
            yield "R10", life + ["--run", "R10", "--arm", "tempo", "--seed", str(s), "--episodes", str(args.sota_episodes),
                                 "--N", str(args.N), "--B", str(args.B), "--estimator", "reinforce"] + C + dev, f"R10-tempo-{cfgn}-s{s}"
        for s in range(2):
            yield "R10", [PY, str(S / "tempo_cont.py"), "--run", "R10", "--tokens", str(sota_tokens), "--seed", str(s)] + C + dev, f"R10-token-{cfgn}-s{s}"

    try:
        for stage, cmd, name in stages():
            if ORDER.index(stage) > last:
                break
            run(cmd, name, args.dry_run)
    except ChainPaused as p:
        print(f"[pause] the chain stopped at {p}; remove {PAUSE_FILE} and re-run the same command to continue",
              flush=True)
        return
    subprocess.run([PY, str(S / "tempo_analyse.py"), "--config", cfgn, "--json", str(RUNS / f"analysis-{cfgn}.json")],
                   cwd=REPO, check=False)


if __name__ == "__main__":
    main()
