"""One lifetime: birth from the genome, one pass over the budget, the waves of development,
the ridge solve, the calibration, the refinement, the evaluation. SPEC: 01 §1.0-1.6, 02 §2.5.

The whole life runs under `torch.no_grad()`; nothing here calls autograd (G-BPF). A life is
a deterministic function of `(genome, cfg)` given the corpus (G-DET).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from anlage.data import EvalSet, N_STRATA
from gnomon.utils.shapes import check_shape
from spore.cortex import Cortex, ResidualScorer, activate, hash_targets, make_candidates
from spore.genome import Genome
from spore.hdc import ContextCoder, Reservoir, atoms, permutation
from spore.memory import Memory
from spore.readout import (AdamState, GramAccumulator, calibrate, calibrate_linear, cross_entropy_rows,
                           delta_rule_step, golden_section, linear_log_probs, softcap)

__all__ = ["LifeConfig", "Organism", "SequentialStream", "live"]


@dataclasses.dataclass
class LifeConfig:
    budget: int                       # target tokens read in the lifetime pass
    offset: int = 0                   # token offset into the training shard
    batch_windows: int = 32
    T: int = 512
    binding: bool = True              # K-HDC arm: False -> rho = I
    memory: bool = True               # K-HDC arm: False -> r = 0, m = 0
    growth: str = "select"            # "select" | "random" | "off" | "ridge"   (K-DEV arms; 11b)
    readout: str = "ridge"            # "ridge" | "sketch"            (N-SKETCH)
    refine: Optional[int] = None      # override of the genome's E
    refine_lr: float = 5e-3
    calib_windows: int = 64
    score_tokens: int = 2 ** 18       # B_c per wave
    tf32: bool = False
    eval_batch: int = 16
    device: str = "cuda"

    def __post_init__(self):
        if self.growth not in ("select", "random", "off", "ridge"):
            raise ValueError("growth in select|random|off|ridge")
        if self.readout not in ("ridge", "sketch"):
            raise ValueError("readout in ridge|sketch")
        if self.budget < self.batch_windows * self.T:
            raise ValueError("budget must cover at least one batch")


class SequentialStream:
    """Consecutive, non-overlapping windows of `T + 1` tokens from `offset` on; wraps at the end."""

    def __init__(self, shard, *, offset: int, T: int, batch_windows: int, device: str):
        self.tokens = shard.tokens
        self.n = int(self.tokens.shape[0])
        self.T, self.B, self.device = int(T), int(batch_windows), device
        self.pos = int(offset) % (self.n - self.T - 1)

    def next(self) -> torch.Tensor:
        L = self.T + 1
        need = self.B * L
        if self.pos + need > self.n:
            self.pos = 0
        arr = np.asarray(self.tokens[self.pos:self.pos + need]).astype(np.int64).reshape(self.B, L)
        self.pos += need
        return torch.from_numpy(arr).to(self.device)


class Organism:
    """The structure regenerated from a genome, plus what the lifetime writes.

    `shard` is needed only when `g.eta_sem > 0`: the first developmental stage counts bigrams
    over `sem_frac` of the budget and forms the semantic atoms (01 §1.1 variant 1b)."""

    def __init__(self, g: Genome, cfg: LifeConfig, V: int, shard=None):
        self.g, self.cfg, self.V = g, cfg, V
        dev = cfg.device
        self.A = atoms(V, g.D, g.s_A, dev)
        self.sem_seconds = 0.0
        if g.eta_sem > 0.0:
            if shard is None:
                raise ValueError("semantic atoms need the shard")
            from spore.semantic import bigram_counts, ppmi_svd, semantic_atoms
            t = time.time()
            N = bigram_counts(shard, offset=cfg.offset, tokens=int(g.sem_frac * cfg.budget), T=cfg.T,
                              batch_windows=cfg.batch_windows, V=V, device=dev)
            self.A = semantic_atoms(self.A, N, d=g.d_sem, eta=g.eta_sem, seed=g.s_A + 1_000_003)
            if g.lro:
                E, _ = ppmi_svd(N, g.d_sem, seed=g.s_A + 1_000_003)
                self.E_out = (E / E.norm(dim=1, keepdim=True).clamp_min(1e-6)).contiguous()   # (V, 2 d_sem)
            del N
            self.sem_seconds = time.time() - t
        self.perm = permutation(g.D, g.s_rho, dev)
        self.coder = ContextCoder(self.A, self.perm, orders=g.orders, gamma_b=g.gamma_b,
                                  gamma_e=g.gamma_e, J_e=g.J_e, binding=cfg.binding)
        self.memory = Memory(self.A, d_k=g.d_k, seed=g.s_M, beta=g.beta) if cfg.memory else None
        if self.memory is not None and g.n_k not in g.orders:
            raise ValueError(f"the memory's key order n_k = {g.n_k} must be one of the coded orders {g.orders}")
        self.cortex = Cortex(g.psi)
        self.reservoir = Reservoir(g.D, g.s_rho + 424_243, g.rho_res, dev) if g.reservoir else None
        self.F0 = (self.coder.n_blocks * g.D + (g.d_k if cfg.memory else 0) + (V if g.onehot else 0)
                   + (g.D if g.reservoir else 0))   # width of X0 (no bias)
        self.W: Optional[torch.Tensor] = None
        self.U: Optional[torch.Tensor] = None          # Definition 6b
        self.E_out: Optional[torch.Tensor] = None
        self.tau_s, self.tau_m = 1.0, 0.0
        self.head = "softmax"              # "softmax" | "linear" | "loglinear"
        self.mu, self.nu = 0.0, 0.0
        self.log_uni: Optional[torch.Tensor] = None

    @property
    def F(self) -> int:
        return 1 + self.F0 + self.cortex.M

    def features(self, tokens: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """`(Z (n, F), m, X_casc)`: the full feature rows `[1 ; X0 ; h]` written block by block into
        one buffer, and the cascade input `X_casc = Z[:, 1:]` as a *view* (no copies: the card's
        memory is the binding constraint, 03 §P-S.5)."""
        c = self.coder.codes(tokens)
        B, T, D = c["emb"].shape
        n = B * T
        Z = torch.empty(n, self.F, dtype=torch.float32, device=c["emb"].device)
        Z[:, 0] = 1.0
        col = 1
        if self.g.onehot:
            # the exact order-1 block: the identity code, D_1 = V (01 §1.2 variant 2a)
            Z[:, col:col + self.V] = 0.0
            Z[torch.arange(n, device=Z.device), col + tokens.reshape(-1)] = 1.0
            col += self.V
        for key in [f"c{o}" for o in self.g.orders] + ["b", "e"]:
            Z[:, col:col + D] = c[key].reshape(n, D)
            col += D
        if self.reservoir is not None:
            Z[:, col:col + D] = self.reservoir.run(c["emb"]).reshape(n, D)
            col += D
        m = None
        if self.memory is not None:
            r = self.memory.retrieve(c[f"c{self.g.n_k}"], c["emb"])
            Z[:, col:col + self.g.d_k] = r.reshape(n, self.g.d_k)
            col += self.g.d_k
            m = self.memory.cleanup(r).reshape(n, self.V)
            del r
        del c
        assert col == 1 + self.F0
        if self.cortex.M > 0:
            self.cortex.forward_into(Z, 1, col)
        return Z, m, Z[:, 1:]

    def logits(self, Z: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
        lg = self.tau_s * (Z @ self.W) if self.W is not None else Z.new_zeros(Z.shape[0], self.V)
        if self.U is not None:
            lg = lg + (Z @ self.U) @ self.E_out.T
        if m is not None and self.tau_m != 0.0:
            lg = lg + self.tau_m * m
        return softcap(lg)

    def log_probs(self, Z: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
        """`(n, V)` log-probabilities under the current head."""
        if self.head == "linear":
            S = Z @ self.W
            return linear_log_probs(S, m, self.log_uni, self.mu, self.nu, self.tau_m)
        return torch.log_softmax(self.logits(Z, m), dim=-1)

    def evaluate(self, ev: EvalSet) -> tuple[float, list[float]]:
        """Token-weighted mean CE and per-stratum CE on a fixed set, the metric every model here reports."""
        total = torch.zeros(N_STRATA, dtype=torch.float64, device=ev.windows.device)
        count = torch.zeros(N_STRATA, dtype=torch.float64, device=ev.windows.device)
        for i in range(0, ev.n_windows, self.cfg.eval_batch):
            w = ev.windows[i:i + self.cfg.eval_batch]
            s = ev.strata[i:i + self.cfg.eval_batch].reshape(-1)
            Z, m, _ = self.features(w[:, :-1])
            lp = self.log_probs(Z, m)
            l = -lp.gather(1, w[:, 1:].reshape(-1, 1)).squeeze(1).double()
            total.index_add_(0, s, l)
            count.index_add_(0, s, torch.ones_like(l))
        ce = float(total.sum() / count.sum())
        per = (total / count.clamp_min(1)).tolist()
        return ce, [p if c > 0 else float("nan") for p, c in zip(per, count.tolist())]

    def calibration_rows(self, ev: EvalSet, n_windows: int) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """`(S = Z W, M, targets)` on the first `n_windows` of `ev`, for Definition 5."""
        Ss, Ms, ts = [], [], []
        for i in range(0, min(n_windows, ev.n_windows), self.cfg.eval_batch):
            w = ev.windows[i:i + self.cfg.eval_batch]
            Z, m, _ = self.features(w[:, :-1])
            Ss.append((Z @ self.W).half() if self.W is not None else torch.zeros(Z.shape[0], self.V, device=Z.device, dtype=torch.float16))
            if m is not None:
                Ms.append(m.half())
            ts.append(w[:, 1:].reshape(-1))
        return torch.cat(Ss), (torch.cat(Ms) if Ms else None), torch.cat(ts)

    def calibrate(self, ev: EvalSet, head: str = "softmax") -> float:
        """Definition 5 (softmax head) or 5b (linear head) on the calibration windows of `ev`."""
        S, M, t = self.calibration_rows(ev, self.cfg.calib_windows)
        if head == "linear":
            mu, nu, tm, ce = calibrate_linear(S, M, t, self.log_uni)
            self.mu, self.nu, self.tau_m, self.tau_s = mu, nu, tm, 1.0
        else:
            ts, tm, ce = calibrate(S, M, t)
            self.tau_s, self.tau_m = ts, tm
        self.head = head
        del S, M
        return ce


def _wave_targets(budget: int, waves, batch_tokens: int) -> list[int]:
    return [int(round(w * budget / batch_tokens)) for w in waves]


def live(g: Genome, cfg: LifeConfig, *, shard, ev_fit: EvalSet, ev_test: EvalSet, V: int,
         log: Callable[[str], None] = print) -> dict:
    """The lifetime of Definition 12's organism. Returns the record (02 §2.5 rule 1)."""
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    # G-DET: every kernel with a deterministic implementation uses it (warn, not fail, on the rest)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()
    rec: dict = {"genome": g.as_dict(), "genome_digest": g.digest(), "cfg": dataclasses.asdict(cfg),
                 "stages": {}, "waves": [], "timings": {}}
    with torch.no_grad():
        org = Organism(g, cfg, V, shard=shard)
        rec["F_birth"] = org.F
        rec["timings"]["semantic"] = org.sem_seconds
        rec["timings"]["birth"] = time.time() - t0
        # ---- R-BIRTH: the innate predictor (Definition 9) --------------------------------
        if org.memory is not None:
            S, M, t = org.calibration_rows(ev_fit, cfg.calib_windows)

            def ce_tm(tm: float) -> float:
                return float(cross_entropy_rows(softcap(tm * M.float()), t).mean())

            tm, ce_fit = golden_section(ce_tm, 0.0, 60.0)
            org.tau_m, org.tau_s = tm, 1.0
            ce_test, strata = org.evaluate(ev_test)
            rec["stages"]["birth"] = {"tau_m": tm, "fitness_calib_ce": ce_fit, "test_ce": ce_test, "test_strata": strata}
            log(f"birth: innate memory-only predictor tau_m {tm:.2f}  test {ce_test:.4f}")
            del S, M, t
        # ---- the lifetime pass --------------------------------------------------------------
        stream = SequentialStream(shard, offset=cfg.offset, T=cfg.T, batch_windows=cfg.batch_windows, device=cfg.device)
        batch_tokens = cfg.batch_windows * cfg.T
        n_batches = int(math.ceil(cfg.budget / batch_tokens))
        gram = GramAccumulator(org.F, V, cfg.device)
        wave_at = _wave_targets(cfg.budget, g.waves, batch_tokens) if (cfg.growth != "off" and g.k_wave > 0) else []
        t1 = time.time()
        seen = 0
        pending_wave: Optional[dict] = None
        for bi in range(n_batches):
            w = stream.next()
            tokens, targets = w[:, :-1], w[:, 1:].reshape(-1)
            Z, m, Xc = org.features(tokens)
            gram.add(Z, targets)
            seen += batch_tokens
            if pending_wave is not None:
                # scoring pass of a wave in progress (01 Definition 11 step 2)
                lg = org.logits(Z, m)
                resid = -torch.softmax(lg, dim=-1)
                resid[torch.arange(resid.shape[0], device=resid.device), targets] += 1.0
                H = activate(Xc, pending_wave["Wc"], pending_wave["phase"], g.psi)
                pending_wave["scorer"].add(H, resid)
                pending_wave["scored"] += batch_tokens
                del H, resid, lg
                if pending_wave["scored"] >= cfg.score_tokens:
                    sc = pending_wave["scorer"].scores()
                    k = min(g.k_wave, sc.numel())
                    if cfg.growth == "select":
                        sel = torch.topk(sc, k).indices
                    else:
                        gsel = torch.Generator(device="cpu").manual_seed(g.s_U + 7919 * (len(rec["waves"]) + 1))
                        sel = torch.randperm(sc.numel(), generator=gsel)[:k].to(sc.device)
                    org.cortex.add_wave(pending_wave["Wc"][sel], pending_wave["phase"][sel])
                    F_in_wave = int(pending_wave["Wc"].shape[1])
                    del pending_wave["Wc"], pending_wave["scorer"]
                    gram.grow(org.F)
                    torch.cuda.empty_cache()
                    rec["waves"].append({"at_tokens": pending_wave["at"], "F_in": F_in_wave,
                                         "n_cand": int(sc.numel()), "kept": int(k), "mode": cfg.growth,
                                         "score_kept_mean": float(sc[sel].mean()), "score_all_mean": float(sc.mean()),
                                         "score_all_max": float(sc.max())})
                    log(f"wave {len(rec['waves'])}: kept {k} of {sc.numel()} (mode {cfg.growth}); "
                        f"score kept {float(sc[sel].mean()):.4f} vs all {float(sc.mean()):.4f}; F -> {org.F}")
                    pending_wave = None
            if wave_at and bi + 1 == wave_at[0] and pending_wave is None and cfg.growth == "ridge":
                wave_at.pop(0)
                # Definition 11b: units are the ridge predictors of seeded random token subsets,
                # solved from the Gram accumulated so far; born frozen, no scoring pass
                k = g.k_wave
                H = hash_targets(V, k, g.s_U + 104729 * (len(rec["waves"]) + 1), cfg.device)
                Wu = gram.solve_targets(g.lam, H)[1:]                     # drop the bias row: (F0 + M, k)
                Wu = (Wu / Wu.norm(dim=0, keepdim=True).clamp_min(1e-8)).T.contiguous()   # (k, F_in)
                phase = torch.zeros(k, device=cfg.device)
                org.cortex.add_wave(Wu, phase)
                gram.grow(org.F)
                torch.cuda.empty_cache()
                rec["waves"].append({"at_tokens": seen, "F_in": int(Wu.shape[1]), "kept": int(k), "mode": "ridge"})
                log(f"wave {len(rec['waves'])} (ridge units): {k} units over F_in {Wu.shape[1]}; F -> {org.F}")
                del H, Wu
            if wave_at and bi + 1 == wave_at[0] and pending_wave is None:
                wave_at.pop(0)
                # solve the current readout and calibrate, then open the scoring pass
                org.W = gram.solve(g.lam) if cfg.readout == "ridge" else gram.C / max(gram.n, 1)
                org.calibrate(ev_fit)
                M_cand = g.cand_mult * g.k_wave
                Wc, phase = make_candidates(M_cand, org.F0 + org.cortex.M, g.k_w, g.s_U + 104729 * (len(rec["waves"]) + 1), cfg.device)
                pending_wave = {"Wc": Wc, "phase": phase, "scorer": ResidualScorer(M_cand, V, cfg.device), "scored": 0, "at": seen}
                log(f"wave opened at {seen} tokens: {M_cand} candidates over F_in {Wc.shape[1]}")
            if (bi + 1) % 100 == 0:
                log(f"  batch {bi + 1}/{n_batches}  {seen / 1e6:.2f}M tokens  {(time.time() - t1) / (bi + 1) * 1000:.0f} ms/batch")
        if pending_wave is not None:
            log("a wave was still scoring at the end of the budget; it is dropped (recorded)")
            rec["waves"].append({"at_tokens": pending_wave["at"], "dropped": True})
        rec["tokens_seen"] = seen
        rec["timings"]["pass"] = time.time() - t1
        # ---- the readout ----------------------------------------------------------------------
        t2 = time.time()
        org.W = gram.solve(g.lam) if cfg.readout == "ridge" else gram.C / max(gram.n, 1)
        torch.cuda.empty_cache()
        rec["timings"]["solve"] = time.time() - t2
        rec["F_final"] = org.F
        rec["M_units"] = org.cortex.M
        # the unigram from the bias row of C: C[0, v] = sum_t 1_{v_{t+1} = v}
        org.log_uni = ((gram.C[0] + 1.0) / (gram.n + V)).log()
        best_stage, best_fit = None, float("inf")
        saved = {}
        for head in ("linear", "softmax"):
            ce_calib = org.calibrate(ev_fit, head)
            ce_fit, _ = org.evaluate(ev_fit)
            ce_test, strata = org.evaluate(ev_test)
            if not (math.isfinite(ce_fit) and math.isfinite(ce_test)):
                raise FloatingPointError(f"non-finite CE at the {head} head (fitness {ce_fit}, test {ce_test}); 03 §P-S.29")
            name = f"ridge-{head}"
            rec["stages"][name] = {"tau_s": org.tau_s, "tau_m": org.tau_m, "mu": org.mu, "nu": org.nu,
                                   "fitness_calib_ce": ce_calib, "fitness_ce": ce_fit, "test_ce": ce_test, "test_strata": strata}
            saved[name] = (org.head, org.tau_s, org.tau_m, org.mu, org.nu)
            log(f"{cfg.readout} {head} head: tau_s {org.tau_s:.2f} tau_m {org.tau_m:.2f} mu {org.mu:.3f} nu {org.nu:.3f}  "
                f"fitness {ce_fit:.4f}  test {ce_test:.4f}  (F {org.F})")
            if ce_fit < best_fit:
                best_stage, best_fit = name, ce_fit
        # the softmax head is the warm start of the log-linear refinement
        org.head, org.tau_s, org.tau_m, org.mu, org.nu = saved["ridge-softmax"]
        # ---- refinement (Definition 6) -------------------------------------------------------
        E = g.E if cfg.refine is None else int(cfg.refine)
        if E > 0:
            t3 = time.time()
            del gram                                   # the Gram is not needed again; its 1-3 GB would page (03 §P-S.10)
            torch.cuda.empty_cache()
            W_ridge = org.W
            org.W = (org.tau_s * org.W).contiguous()
            org.tau_s = 1.0
            org.head = "loglinear"
            adam = AdamState(org.W, lr=cfg.refine_lr)
            adam_U = None
            if g.lro and org.E_out is not None:
                org.U = torch.zeros(org.F, org.E_out.shape[1], device=cfg.device)
                adam_U = AdamState(org.U, lr=cfg.refine_lr)
            total = E * n_batches
            stream = SequentialStream(shard, offset=cfg.offset, T=cfg.T, batch_windows=cfg.batch_windows, device=cfg.device)
            k = 0
            for e in range(E):
                run = 0.0
                for bi in range(n_batches):
                    w = stream.next()
                    Z, m, _ = org.features(w[:, :-1])
                    off = org.tau_m * m if m is not None else None
                    ce = delta_rule_step(org.W, Z, w[:, 1:].reshape(-1), off, adam, lr_scale=1.0 - k / total,
                                         U=org.U, E_out=org.E_out, adam_U=adam_U)
                    run += ce
                    k += 1
                    if (bi + 1) % 100 == 0:
                        log(f"  refine pass {e + 1}/{E} batch {bi + 1}/{n_batches}  train CE {run / (bi + 1):.4f}")
                log(f"refine pass {e + 1}/{E}: mean train CE {run / n_batches:.4f}")
            del adam, adam_U
            rec["timings"]["refine"] = time.time() - t3
            ce_fit_r, _ = org.evaluate(ev_fit)
            ce_test_r, strata_r = org.evaluate(ev_test)
            rec["stages"]["refined"] = {"E": E, "lr": cfg.refine_lr, "tau_m": org.tau_m, "fitness_ce": ce_fit_r,
                                        "test_ce": ce_test_r, "test_strata": strata_r}
            log(f"refined: fitness {ce_fit_r:.4f}  test {ce_test_r:.4f}")
            if ce_fit_r < best_fit:
                best_stage, best_fit = "refined", ce_fit_r
            else:
                org.W = W_ridge
                org.U = None
                org.head, org.tau_s, org.tau_m, org.mu, org.nu = saved[best_stage]
        rec["final_stage"] = best_stage
        rec["fitness_ce"] = best_fit
        rec["test_ce"] = rec["stages"][best_stage]["test_ce"]
        rec["test_strata"] = rec["stages"][best_stage]["test_strata"]
        rec["timings"]["total"] = time.time() - t0
        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        rec["complete"] = True
    return rec
