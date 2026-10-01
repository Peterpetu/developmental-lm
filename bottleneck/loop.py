"""Intermittent training: the paper's Algorithm 1 with its Eq. 3, as a lifetime hook.

SPEC: docs/bottleneck/00-replication.md §3 (the loop), §3.1 (the schedule), §5 (controls)

The phenotype is trained freely by `anlage.lifetime.train`; this object is its `before_step`
hook. At every generation boundary -- every `T_gen` tokens -- it (b) refits the genome to the
current weights, (c) decodes, measures the decode at birth, and (d) regenerates
`W ← ε(n) W + (1 − ε(n)) Ŵ` with `ε(n) = exp(−n/λ)`. The decode runs here and nowhere else:
G-NOFREE-INVERSE holds by construction, because the training step never sees the genome
(I10). What TEMPO and ANLAGE did not do -- anneal `ε` toward zero so that the phenotype is
regenerated from the genome before every training burst -- is the whole of step (d).
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import time
from typing import Callable, Optional, Union

import torch

from anlage.config import MATRIX_ROLES
from anlage.model import Phenotype

from .genome import BottleneckGenome

__all__ = ["epsilon", "regenerate", "Intermittent", "shuffle_rows", "fisher_diag",
           "fisher_sampling_weights"]


def epsilon(n: int, lam: float) -> float:
    """Eq. 3 of the paper: `ε(n) = exp(−n/λ)`, `λ = 20` there. SPEC: (3.1)."""
    if n < 0 or lam <= 0:
        raise ValueError("n >= 0 and lambda > 0")
    return math.exp(-float(n) / float(lam))


@torch.no_grad()
def regenerate(model: Phenotype, D: dict, eps: float) -> None:
    """Step (d): `W ← ε W + (1 − ε) Ŵ` on every weight, in place. SPEC: (3.1), I6.

    `ε = 1` leaves every weight bit-identical (the update is `p ← p·1 + d·0`, which PyTorch's
    `mul_(1.0)` and `add_(alpha=0.0)` leave exact); `ε = 0` sets every weight to its decode."""
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps in [0, 1]")
    for k, p in model.weights().items():
        if eps == 1.0:
            continue
        d = D[k].to(p.dtype)
        if eps == 0.0:
            p.data.copy_(d)
        else:
            p.data.mul_(eps).add_(d, alpha=1.0 - eps)


@torch.no_grad()
def shuffle_rows(D: dict, generator: torch.Generator) -> dict:
    """The paper's shuffle control (Fig. 2D): rows of every matrix permuted. SPEC: §5, I8."""
    out = {}
    for (role, l), d in D.items():
        if role in MATRIX_ROLES and d.dim() == 2:
            perm = torch.randperm(d.shape[0], generator=generator).to(d.device)
            out[(role, l)] = d[perm]
        else:
            out[(role, l)] = d
    return out


def fisher_diag(model: Phenotype, windows: torch.Tensor, *, micro: int = 4) -> dict:
    """Diagonal empirical Fisher of the next-token CE at the model's current weights, per
    weight key: the mean over micro-batches of the squared gradient of the micro-batch's mean
    token CE (`Phenotype.loss`'s first output, the training objective without its z-loss
    regulariser). SPEC: (11.1).

    This is the estimator of Kirkpatrick et al. 2017 (PNAS 114:3521, Eq. 3) as practised: the
    cross terms between the tokens of one micro-batch are inside the square, which is the
    standard approximation and is recorded as such (§11.1). The probe is a fixed slice of the
    fitness set, never the test set. Gradients are left as `None`, the state the trainer keeps
    them in between steps, so the training step that follows the hook is unaffected."""
    if windows.dim() != 2 or windows.shape[0] < 1 or micro < 1:
        raise ValueError("windows (n, T+1) with n >= 1; micro >= 1")
    was = model.training
    model.eval()
    live = model.weights()
    acc = {k: torch.zeros(p.shape, dtype=torch.float32, device=p.device) for k, p in live.items()}
    n = 0
    with torch.enable_grad():
        for a in range(0, windows.shape[0], micro):
            w = windows[a:a + micro]
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
                ce, _ = model.loss(w)
            ce.backward()
            for k, p in live.items():
                if p.grad is not None:
                    acc[k].add_(p.grad.detach().float() ** 2)
            n += 1
    model.zero_grad(set_to_none=True)
    if was:
        model.train()
    return {k: v.div_(n) for k, v in acc.items()}


def fisher_sampling_weights(F: dict, keys, alpha: float) -> dict:
    """`F̃ = (1 − α) F / mean(F) + α`, flat per key, on the CPU where the sampler draws.
    SPEC: (11.1). `α = 1` is Algorithm 1's uniform sampler exactly; `α = 0` never fits an
    entry the probe's gradient did not reach. A key whose Fisher is identically zero (nothing
    on the probe reached it) gets the uniform weight, which is Algorithm 1 for that key."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha in [0, 1]")
    out = {}
    for k in keys:
        f = F[k].detach().float().reshape(-1)
        m = float(f.mean())
        if m <= 0.0 or not math.isfinite(m):
            out[k] = torch.ones(f.numel())
        else:
            out[k] = ((1.0 - alpha) * (f / m) + alpha).cpu()
    return out


class Intermittent:
    """SPEC: §3. One instance per run; `before_step` is handed to `anlage.lifetime.train`."""

    def __init__(self, genome: BottleneckGenome, model: Phenotype, *, T_gen: int, lam: float,
                 n_refit: int, fraction: float, cap: int, lr: float,
                 evaluate: Callable[[Phenotype], float], shuffle_every: int = 10,
                 seed: int = 0, log: Optional[Callable[[str], None]] = None,
                 max_generations: Optional[int] = None, out_dir=None,
                 fisher_windows: Optional[Union[torch.Tensor, Callable[[], torch.Tensor]]] = None,
                 fisher_alpha: float = 0.1, fisher_micro: int = 4,
                 evaluate_fit: Optional[Callable[[Phenotype], float]] = None, select_from: int = 30):
        if T_gen < 1 or n_refit < 0 or shuffle_every < 1:
            raise ValueError("T_gen >= 1, n_refit >= 0, shuffle_every >= 1")
        if fisher_windows is not None and not callable(fisher_windows) and fisher_windows.dim() != 2:
            raise ValueError("fisher_windows is a (n, T+1) tensor or a callable returning one")
        if not (0.0 <= fisher_alpha <= 1.0) or fisher_micro < 1 or select_from < 1:
            raise ValueError("fisher_alpha in [0, 1], fisher_micro >= 1, select_from >= 1")
        self.G, self.model = genome, model
        # §11.1: the Fisher metric of the refit (None = Algorithm 1's plain MSE)
        self.fisher_windows, self.fisher_alpha, self.fisher_micro = fisher_windows, float(fisher_alpha), int(fisher_micro)
        # §11.2: selection on the fitness set from generation `select_from` (None = no selection)
        self.evaluate_fit, self.select_from = evaluate_fit, int(select_from)
        self.accepted: Optional[dict] = None         # the decode in force: D, fit, birth, generation
        self.n_proposals, self.n_accepted = 0, 0
        # SPEC §3.2: the state at the generation of best at-birth CE is what transfer reads. The
        # end of a run is not that state -- the first GN30 run's last thirty-five generations
        # collapsed under the schedule's decay (11 §P-D.46) -- and the paper reports the best.
        self.out_dir = None if out_dir is None else pathlib.Path(out_dir)
        self.best_birth = float("inf")
        self.best_generation = 0
        self.T_gen, self.lam = int(T_gen), float(lam)
        self.n_refit, self.fraction, self.cap = int(n_refit), float(fraction), int(cap)
        self.evaluate = evaluate
        self.shuffle_every = int(shuffle_every)
        self.log = log
        self.max_generations = max_generations
        # Algorithm 1 holds one g_optimizer across generations: the refits are warm-started
        self.opt = torch.optim.Adam(self.G.parameters(), lr=float(lr))
        self.gen_rng = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + 1)
        self.shuffle_rng = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + 2)
        self.generation = 0                      # generations completed
        self.next_boundary = self.T_gen
        self.records: list[dict] = []
        self.seconds_in_hook = 0.0
        self.decode_calls = 0

    # ----------------------------------------------------------------- helpers
    @torch.no_grad()
    def _weights(self) -> dict:
        return {k: p.detach() for k, p in self.model.weights().items()}

    @torch.no_grad()
    def _evaluate_weights(self, W_new: dict, evaluate: Optional[Callable[[Phenotype], float]] = None) -> float:
        """The phenotype's CE under `evaluate` (the test set by default) with its weights
        temporarily replaced by `W_new`."""
        live = self.model.weights()
        saved = {k: p.data.clone() for k, p in live.items()}
        try:
            for k, p in live.items():
                p.data.copy_(W_new[k].to(p.dtype))
            return float((evaluate or self.evaluate)(self.model))
        finally:
            for k, p in live.items():
                p.data.copy_(saved[k])

    def _snapshot(self) -> dict:
        """The genome and its optimiser, copied: what a rejected proposal is restored to (§11.2)."""
        return {"genome": {k: v.detach().clone() for k, v in self.G.state_dict().items()},
                "opt": copy.deepcopy(self.opt.state_dict())}

    def _restore(self, snapshot: dict) -> None:
        self.G.load_state_dict(snapshot["genome"])
        self.opt.load_state_dict(snapshot["opt"])

    @torch.no_grad()
    def _save_best(self, n: int, ce_birth: float, ce_before: float) -> None:
        """`genome_best.pt` and `phenotype_best.pt` -- the genome and the trained weights it was
        fitted to (`W_n`, before regeneration) at the generation of best at-birth CE. SPEC: §3.2."""
        from anlage.model import save_phenotype
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.G.state_dict(), "summary": self.G.summary(), "keys": self.G.keys,
                    "shapes": self.G.shapes, "generation": n, "ce_birth": ce_birth},
                   self.out_dir / "genome_best.pt")
        save_phenotype(self.model, self.out_dir / "phenotype_best.pt")
        (self.out_dir / "best.json").write_text(json.dumps(
            {"generation": n, "ce_birth": ce_birth, "ce_before": ce_before}), encoding="utf-8")

    # ------------------------------------------------------------- generation
    def generation_step(self, step: int, tokens_done: int) -> dict:
        """(b) refit, (c) decode and measure, (d) regenerate. SPEC: §3; with the §11 variants
        when they are on: the Fisher metric is taken before (b), selection sits between (c)
        and (d), and neither touches the test set."""
        t0 = time.time()
        n = self.generation + 1
        W = self._weights()
        ce_before = float(self.evaluate(self.model))
        # §11.1: the regression's metric, the Fisher at W_n on the probe
        weights, fisher_seconds = None, 0.0
        if self.fisher_windows is not None:
            tf = time.time()
            probe = self.fisher_windows() if callable(self.fisher_windows) else self.fisher_windows
            F = fisher_diag(self.model, probe, micro=self.fisher_micro)
            weights = fisher_sampling_weights(F, self.G.compressed_keys(), self.fisher_alpha)
            fisher_seconds = time.time() - tf
        # §11.2: the state a rejected proposal is restored to
        selecting = self.evaluate_fit is not None and n >= self.select_from
        snapshot = self._snapshot() if selecting else None
        # (b) warm-started regression, n_refit Adam steps over every g-network at once
        last = {}
        for _ in range(self.n_refit):
            last = self.G.fit_step(W, fraction=self.fraction, cap=self.cap,
                                   generator=self.gen_rng, opt=self.opt, weights=weights)
        # (c) decode, the at-birth measurement, the residuals
        D = self.G.decode_all(W)
        self.decode_calls += 1
        res = self.G.residuals(W, D)
        res_F = self.G.weighted_residuals(W, D, weights) if weights is not None else None
        ce_birth = self._evaluate_weights(D)                      # the proposal, on the test set
        ce_shuffle = float("nan")
        if n % self.shuffle_every == 0 or n == 1:
            ce_shuffle = self._evaluate_weights(shuffle_rows(D, self.shuffle_rng))
        # §11.2: accept or reject on the fitness set; the test set decides nothing
        ce_birth_fit, accepted, ce_birth_used = float("nan"), None, ce_birth
        if self.evaluate_fit is not None:
            ce_birth_fit = self._evaluate_weights(D, self.evaluate_fit)
            if selecting and self.accepted is not None and ce_birth_fit > self.accepted["fit"]:
                accepted = False
                self._restore(snapshot)
                D, ce_birth_used = self.accepted["D"], self.accepted["birth"]
            else:
                accepted = True if selecting else None
                self.accepted = {"D": D, "fit": ce_birth_fit, "birth": ce_birth, "generation": n}
            if selecting:
                self.n_proposals += 1
                self.n_accepted += int(bool(accepted))
        if ce_birth_used < self.best_birth:
            self.best_birth, self.best_generation = ce_birth_used, n
            if self.out_dir is not None:
                self._save_best(n, ce_birth_used, ce_before)
        # (d) regeneration, Eq. 3, from the decode in force
        eps = epsilon(n, self.lam)
        regenerate(self.model, D, eps)
        ce_after = float(self.evaluate(self.model))
        dt = time.time() - t0
        self.seconds_in_hook += dt
        mats = [v for k, v in res.items() if k[0] in MATRIX_ROLES]
        mats_F = [v for k, v in res_F.items() if k[0] in MATRIX_ROLES] if res_F else []
        rec = {"generation": n, "step": step, "tokens": tokens_done, "eps": eps,
               "ce_before": ce_before, "ce_birth": ce_birth, "ce_after": ce_after,
               "ce_shuffle": ce_shuffle,
               "ce_birth_fit": ce_birth_fit, "accepted": accepted, "ce_birth_used": ce_birth_used,
               "residual_mean_matrix": float(sum(mats) / len(mats)) if mats else float("nan"),
               "residual_max_matrix": float(max(mats)) if mats else float("nan"),
               "residual": {self.G._name(k): v for k, v in res.items()},
               "residual_fisher_mean_matrix": float(sum(mats_F) / len(mats_F)) if mats_F else float("nan"),
               "residual_fisher": ({self.G._name(k): v for k, v in res_F.items()} if res_F else None),
               "fit_mse_last": {self.G._name(k): v for k, v in last.items()},
               "fisher_seconds": fisher_seconds, "seconds": dt}
        self.records.append(rec)
        self.generation = n
        self.next_boundary += self.T_gen
        if self.log:
            extra = ""
            if self.evaluate_fit is not None:
                extra += f"  fit {ce_birth_fit:.4f}" + ("" if accepted is None else ("  ACC" if accepted else "  rej"))
            if res_F:
                extra += f"  residF {rec['residual_fisher_mean_matrix']:.3f}"
            self.log(f"  gen {n:4d}  tokens {tokens_done:>12,}  eps {eps:.4f}  "
                     f"before {ce_before:.4f}  birth {ce_birth:.4f}  after {ce_after:.4f}  "
                     f"shuffle {ce_shuffle:.4f}  resid {rec['residual_mean_matrix']:.3f} "
                     f"(max {rec['residual_max_matrix']:.3f}){extra}  {dt:.1f}s")
        return rec

    # ------------------------------------------------------------------- hook
    def before_step(self, step: int, tokens_done: int) -> None:
        """Fires once per generation, at the first step whose `tokens_done` reaches the
        boundary (I7). Returns nothing: the phenotype's optimisers are kept (§3)."""
        if self.max_generations is not None and self.generation >= self.max_generations:
            return None
        if tokens_done >= self.next_boundary:
            self.generation_step(step, tokens_done)
        return None

    def as_dict(self) -> dict:
        return {"T_gen": self.T_gen, "lambda": self.lam, "n_refit": self.n_refit,
                "fraction": self.fraction, "cap": self.cap, "shuffle_every": self.shuffle_every,
                "generations": self.generation, "decode_calls": self.decode_calls,
                "best_birth": self.best_birth, "best_generation": self.best_generation,
                "fisher": self.fisher_windows is not None, "fisher_alpha": self.fisher_alpha,
                "fisher_micro": self.fisher_micro,
                "select": self.evaluate_fit is not None, "select_from": self.select_from,
                "n_proposals": self.n_proposals, "n_accepted": self.n_accepted,
                "accepted_generation": (self.accepted["generation"] if self.accepted else None),
                "seconds_in_hook": self.seconds_in_hook, "records": self.records}
