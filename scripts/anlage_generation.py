#!/usr/bin/env python
"""R8 LOOP: one generation of ANLAGE-P -- siblings on a Dirichlet neighbourhood of the
programme's mixture, ε-lexicase over strata, the task-vector merge, the programme
regression, growth timed by `omega` and seeded by `Lambda`, the adult, and the next
programme.

SPEC: 12 §12.3
SPEC: 12 §12.4
SPEC: 12 §12.5
SPEC: 12 §12.7
SPEC: 12 §12.8 (K-LOOP)

    python scripts/anlage_generation.py --gen 0 --tokens-total 50000000 --K 4
    python scripts/anlage_generation.py --gen 1 --program runs/anlage/gen0-.../program.pt \\
        --prev-adult runs/anlage/gen0-.../adult.pt --history runs/anlage/gen0-.../generation.json
    python scripts/anlage_generation.py --gen 0 --no-population      # SINGLE-WARM control

Budgets are FLOPs: `C_total = 6 N_adult · tokens_total`, split by `omega` (§06.5) and
converted to tokens per stage; the `K` siblings share stage 0's budget (S9). The
SINGLE-WARM control (`--no-population`) spends stage 0's budget on one lifetime at the
programme's mixture and skips the programme regression: same warm start, same growth,
same compute, no population.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anlage_common import REPO, eval_sets, fit_windows, get_lineage, load_corpus, log_to, manifest, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from anlage.data import ClassifiedPool, EvalSet, classify_documents  # noqa: E402
from anlage.grow import Ramp, grow, shrink, unit_prior  # noqa: E402
from anlage.lifetime import evaluate, train  # noqa: E402
from anlage.model import Phenotype, load_phenotype, save_phenotype  # noqa: E402
from anlage.population import merge, select_parents  # noqa: E402
from anlage.program import (ALPHA_0, TAU, Program, dirichlet_neighbourhood, fit_surrogate,  # noqa: E402
                            solve_trust_region)
from anlage.recipe import TrainConfig, micro_batch_for  # noqa: E402
from gnomon.utils.rng import derive_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", default="tinystories")
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--program", default=None, help="the previous generation's program.pt")
    ap.add_argument("--prev-adult", default=None, help="the previous generation's adult.pt (warm start)")
    ap.add_argument("--history", nargs="*", default=[], help="previous generation.json files (surrogate memory)")
    ap.add_argument("--tokens-total", type=int, default=50_000_000, help="adult-token-equivalent budget")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=ALPHA_0)
    ap.add_argument("--n-par", type=int, default=2)
    ap.add_argument("--merge-steps", type=int, default=200)
    ap.add_argument("--gamma-memory", type=float, default=0.5)
    ap.add_argument("--growth", default="prior", choices=["prior", "copy", "random"])
    ap.add_argument("--no-population", action="store_true", help="SINGLE-WARM: K = 1, no programme step")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-emb", type=float, default=3e-3)
    ap.add_argument("--noise", default="runs/anlage/noise-tinystories-s0/noise.json")
    ap.add_argument("--pool-windows", type=int, default=200_000)
    ap.add_argument("--pool-seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke test beside a GPU job")
    args = ap.parse_args()

    device = args.device
    lin = get_lineage(args.lineage)
    stages = [lin.stage(s) for s in range(args.stages)]
    cfg0 = stages[0]
    single = args.no_population
    K = 1 if single else args.K
    tag = args.tag or ("single" if single else f"K{K}-{args.growth}")
    out = pathlib.Path(args.out or REPO / "runs" / "anlage" / f"gen{args.gen}-{lin.name}-{tag}-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    corpus = load_corpus(lin)
    cls_train, man = classify_documents(corpus.train, corpus.rank)
    n_C = man["n_classes"]
    pool = ClassifiedPool(corpus.train, cls_train, n_C, n_windows=args.pool_windows, T=lin.T,
                          pool_seed=args.pool_seed, device=device)
    test, fit = eval_sets(corpus, lin, device=device)
    fitw = fit_windows(corpus, lin, n=64, device=device)
    noise = json.loads(pathlib.Path(args.noise).read_text())
    sigma_e = float(noise["sigma_e"])
    eps_q = [float(x) if x is not None else float("nan") for x in noise["sigma_e_strata_fit"]]
    w_q = torch.tensor(noise["stratum_counts_test"], dtype=torch.float64)
    w_q = w_q / w_q.sum()

    # ---- the programme ------------------------------------------------------------
    fr = lin.budget_fractions[: args.stages]
    omega = tuple(f / sum(fr) for f in fr)
    if args.program:
        prog = Program.load(args.program)
        if prog.classes.get("names") != man["names"] or prog.classes.get("length_edges") != man["length_edges"]:
            raise RuntimeError("the programme's class partition differs from this corpus's")
    else:
        mu_c = pool.corpus_mixture
        prog = Program(mu=torch.stack([mu_c] * args.stages), omega=omega, eta={"lr_mult": 1.0},
                       s_state=None, Lambda={}, classes=man)
    mu_0 = prog.mu[0].clone()
    C_total = lin.flops(args.stages - 1, args.tokens_total)
    tokens_stage = list(lin.stage_tokens(C_total, n_stages=args.stages))
    tokens_sib = lin.sibling_tokens(C_total, K, n_stages=args.stages)
    lr_mult = float(prog.eta.get("lr_mult", 1.0))
    log(f"R8 gen {args.gen}  {lin.name}  {'SINGLE-WARM' if single else f'K={K}'}  C_total {C_total:.3e}  "
        f"stage tokens {tokens_stage}  tokens/sib {tokens_sib:,}  growth {args.growth}  alpha {args.alpha}  "
        f"mu_0 {[round(x, 4) for x in mu_0.tolist()]}  program {args.program}  prev adult {args.prev_adult}")

    def tc_for(tokens, n_params=13_000_000):
        return TrainConfig(tokens=tokens, batch_final=args.batch, T=lin.T, lr_muon=args.lr_muon * lr_mult,
                           lr_emb=args.lr_emb * lr_mult, lr_provenance="R1 (R2 not run)",
                           micro_batch=micro_batch_for(n_params, args.batch))

    # ---- birth: the primordium block of the previous adult, or the free init -----------
    if args.prev_adult:
        prev = load_phenotype(args.prev_adult, device=device)
        W0_model = shrink(prev, cfg0)
        del prev
    else:
        W0_model = Phenotype(cfg0, lin.vocab, seed=args.seed).to(device)
    W0 = {k: v.detach().clone() for k, v in W0_model.weights().items()}
    ce_birth, _ = evaluate(W0_model, test)
    cfg0_used = W0_model.cfg
    log(f"  birth (stage 0 block): test {ce_birth:.4f}")
    record = {"run": "R8-GEN", "gen": args.gen, "seed": args.seed, "single": single, "K": K, "growth": args.growth,
              "alpha": args.alpha, "tokens_total": args.tokens_total, "C_total": C_total, "tokens_stage": tokens_stage,
              "tokens_sib": tokens_sib, "birth_ce": ce_birth, "mu_0": mu_0.tolist(), "omega": list(omega),
              "classes": man, "pool_digest": pool.digest(), "class_digest": pool.class_digest(), "stages": []}

    # ---- stage 0: siblings ----------------------------------------------------------------
    nb = dirichlet_neighbourhood(mu_0, args.alpha, K, seed=derive_seed(args.seed, "mutation") ^ (args.gen * 7919))
    sibs, samples = [], []
    for k in range(K):
        m = Phenotype(cfg0_used, lin.vocab, weights=W0).to(device)
        r = train(m, pool.at(nb[k]), test, tc_for(tokens_sib), order_seed=derive_seed(args.seed, "member") ^ k,
                  compile_mode=None, log=None, device=device)
        ce_fit, strata = evaluate(m, fit)
        sibs.append(m)
        samples.append({"mu": nb[k].tolist(), "strata": strata, "test_ce": r.final_ce, "fitness_ce": ce_fit,
                        "gen": args.gen})
        log(f"  sibling {k}: mu {[round(x, 3) for x in nb[k].tolist()]}  test {r.final_ce:.4f}  fit {ce_fit:.4f}")
    if K > 1:
        rng = np.random.default_rng(derive_seed(args.seed, "lexicase") ^ args.gen)
        parents = sorted(set(select_parents([s["strata"] for s in samples], eps_q, args.n_par, rng)))
        elite, mrep = merge([sibs[i] for i in parents], W0, fitw, fit, sigma_e, steps=args.merge_steps, log=log)
        ce_elite, _ = evaluate(elite, test)
        best = int(np.argmin([s["fitness_ce"] for s in samples]))
        log(f"  parents {parents}  {mrep.chosen}  elite test {ce_elite:.4f}  best sibling {samples[best]['test_ce']:.4f}")
        record["population"] = {"parents": parents, "merge": mrep.as_dict(), "elite_test_ce": ce_elite,
                                "best_sibling": best, "best_sibling_test_ce": samples[best]["test_ce"]}
    else:
        elite = sibs[0]
        ce_elite = samples[0]["test_ce"]
    record["samples"] = samples
    record["stages"].append({"stage": 0, "test_ce": ce_elite})
    save_phenotype(elite, out / "stage0.pt")
    del sibs
    torch.cuda.empty_cache()

    # ---- the programme regression -----------------------------------------------------------
    if K > 1:
        hist_mu, hist_F, hist_w = [], [], []
        for h in args.history:
            hj = json.loads(pathlib.Path(h).read_text())
            age = args.gen - int(hj["gen"])
            for s_ in hj.get("samples", []):
                hist_mu.append(s_["mu"])
                hist_F.append([x if (x is not None and x == x) else 0.0 for x in s_["strata"]])
                hist_w.append(args.gamma_memory ** max(age, 0))
        mus = torch.tensor([s["mu"] for s in samples] + hist_mu, dtype=torch.float64)
        F = torch.tensor([[x if (x is not None and x == x) else 0.0 for x in s["strata"]] for s in samples] + hist_F,
                         dtype=torch.float64)
        wts = torch.tensor([1.0] * len(samples) + hist_w, dtype=torch.float64)
        sur = fit_surrogate(mus, F, weights=wts)
        mu_star, info = solve_trust_region(sur, w_q, mu_0, tau=TAU)
        log(f"  programme: {'linear' if sur.linear_only else 'quadratic'} surrogate on {sur.n_samples} samples "
            f"(LOO RMSE {sur.loo_rmse:.4f}); mu* {[round(x, 4) for x in mu_star.tolist()]}  "
            f"predicted {info['value']:.4f}  step {info['distance']:.3f}")
        record["programme"] = {"surrogate": {"linear_only": sur.linear_only, "n_samples": sur.n_samples,
                                             "loo_rmse": sur.loo_rmse}, "mu_star": mu_star.tolist(), "info": info}
    else:
        mu_star = mu_0.clone()

    # ---- growth stages ------------------------------------------------------------------------
    model = elite
    for s in range(1, args.stages):
        cfg_next = stages[s]
        Lam = unit_prior(model)
        pre, _ = evaluate(model, test)
        grown, plan = grow(model, cfg_next, mode=args.growth, Lambda=Lam,
                           generator=torch.Generator().manual_seed(derive_seed(args.seed, "growth") ^ s))
        post, _ = evaluate(grown, test)
        n_steps = max(1, tokens_stage[s] // (args.batch * lin.T))
        ramp = Ramp(grown, plan, n_ramp=max(1, int(0.02 * n_steps)))
        log(f"  grow -> {cfg_next.summary()} ({args.growth}): pre {pre:.4f}  at rho_0 {post:.4f}  "
            f"{tokens_stage[s]:,} tokens")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        def before_step(step, tokens_done, _r=ramp):
            if not _r.done:
                _r.step()

        r = train(grown, pool.at(mu_star), test, tc_for(tokens_stage[s], grown.n_params()),
                  order_seed=derive_seed(args.seed, "growth") ^ (s * 31), compile_mode=None,
                  hooks={"before_step": before_step}, log=log, eval_every_tokens=max(tokens_stage[s] // 4, 1),
                  device=device)
        log(f"  stage {s}: test {r.final_ce:.4f}  {r.tokens_per_s:,.0f} tok/s  peak {r.peak_mem_gb:.1f} GB")
        record["stages"].append({"stage": s, "test_ce": r.final_ce, "curve": r.curve, "pre_grow_ce": pre,
                                 "post_grow_ce_rho0": post, "ops": list(plan.cfg_new.ops()),
                                 "tokens_per_s": r.tokens_per_s})
        model = grown

    # ---- the adult and the next programme ---------------------------------------------------------
    ce_adult, strata_adult = evaluate(model, test)
    ce_adult_fit, _ = evaluate(model, fit)
    save_phenotype(model, out / "adult.pt")
    Lam_adult = unit_prior(model)
    prog_next = Program(mu=torch.stack([mu_star.clone() for _ in range(args.stages)]), omega=omega,
                        eta=dict(prog.eta), s_state=None, Lambda=Lam_adult, classes=man)
    prog_next.save(out / "program.pt")
    record.update({"adult_ce": ce_adult, "adult_fitness_ce": ce_adult_fit, "adult_strata": strata_adult,
                   "adult_config": model.cfg.summary(), "n_params_adult": model.n_params(),
                   "mu_star": mu_star.tolist(), "program_reals": prog_next.n_reals()})
    write_json(out / "generation.json", manifest(record))
    log(f"  adult {model.cfg.summary()}: test {ce_adult:.4f}  fit {ce_adult_fit:.4f}  "
        f"programme {prog_next.n_reals()} reals")
    log(f"wrote {out / 'generation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
