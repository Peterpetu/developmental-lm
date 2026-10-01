"""The life: birth, the episode loop, the boundary, evaluation, inference, the checkpoint.

SPEC: 08 §8.1  (birth)
SPEC: 08 §8.2  (the loop: B1-B5, S1-S7, O1-O10)
SPEC: 08 §8.3  (lanes, declared boundaries, entering states of group rows, window-local
                attention, fresh-state evaluation, the fitness sets)
SPEC: 08 §8.4  (inference is the same loop with eta_1 = 0: `run_episode(learn=False)`)
SPEC: 08 §8.5  (the checkpoint: everything, bit-exact on CPU)
SPEC: 08 §8.6  (the record; compute accounting including the probes)
SPEC: 01 §1.2  (the three timescales), §1.4 (G-STAT, two-sided)
SPEC: 02 §2.4  (losses per modality, insulation), §2.6 (the policy-gradient term)
SPEC: 04 §4.5  (the episode cycle of the gains, the history regression, A8 in gain space)
SPEC: 05 §5.3  (births; the one-shot doubling by hypercloning), §5.7 (controls by budget)
SPEC: 06       (the boundary: fitness reward, update-then-propose, the null probe, the kept winner)
SPEC: 07       (the probe on the episode's last windows with their live states; the zeta rule)
SPEC: 09 §9.5  (a non-finite loss skips the step and is counted; more than three abort)

One code path: `Life.run_episode(learn=True)` trains; `learn=False` is the same loop with
the optimiser step skipped (A1); `Life.step_inference` is one span of it.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import hashlib
import math
import pathlib
import random
import statistics
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from anlage.data import systematic_counts
from anlage.recipe import TrainConfig, lr_at
from gnomon.optim.muon import Muon
from .backbone import Backbone
from .codec import BernoulliHead, GaussianHead, SpanCodec
from anlage.ops import Z_LOSS
from .config import TempoConfig
from .consolidate import (
    ETA_2, G_PROPOSALS, LAMBDA_DEC, N_PROBE_SET, N_PROBE_STEPS, P_REHEARSAL, TAU_FACTOR, Consolidator,
    Rehearsal, Reinforce, RestEM, WindowSummaries, es_proposal, genome_step, L_check, reward as reward_fn,
    salience_fit, salience_targets,
)
from .criticality import KAPPA, N_DELTA, N_PROBE_WINDOWS, probe, trigger as crit_trigger, zeta_update
from .envs import grpo_loss, group_advantage
from .genome import (
    GENOME_NORM_BOUND, HISTORY, EpisodeGains, Expression, RoleRegistry, backbone_gains, genome_prior,
    lr_multipliers, mixture,
)
from .modalities import (
    MODALITIES, MODALITY_INDEX, STREAMS, ImageDecoder, ImageEncoder, Insulation, LinearEncoder, Perception,
    SpeechEncoder, TextIn, insulate, interface_flops,
)
from .morphology import (
    SwitchPattern, apply_births, born_flags_at_birth, control_pattern, flops_per_window, hypercloning_split,
    n_active_params,
)
from .streams import REWARD_STREAMS, Window, scored_tokens, window_batch_targets

__all__ = ["LifeConfig", "Agent", "Lane", "Life", "Paused", "G_STAT_SIGMA_FACTOR", "gstat_check", "code_digest",
           "P_REHEARSAL_BUFFER", "SCALE_FLOOR", "HEAD_OF_STREAM"]


class Paused(Exception):
    """Raised inside the step loop when the caller's `should_pause` says to stop (08 §8.5).

    The life is left in a resumable state: the in-flight episode's summaries, lanes, optimiser
    state and RNG are exactly what the next step needs, and `checkpoint_state` records that the
    episode is in progress and which step comes next. The caller writes the checkpoint and
    exits; the resumed life continues at that step without re-running the boundary.
    """

    def __init__(self, episode: int, next_step: int):
        super().__init__(f"paused in episode {episode} before step {next_step}")
        self.episode, self.next_step = episode, next_step

G_STAT_SIGMA_FACTOR = 2.0
G_STAT_RATE_FACTOR = 0.25
G_STAT_MAX_DOUBLINGS = 1
"""SPEC: 01 §1.4 -- G-STAT: the episode's transient is over when the loss's rate of decrease
over the last fifth of the steps is below a quarter of the rate over the first fifth after
the rehearsal phase, or when the last fifth is flat to twice the larger of `sigma_e` and its
own standard error; N is doubled at most once per life (11 §P-C.22)."""
P_REHEARSAL_BUFFER = 0.10
"""SPEC: 06 §6.9 -- the buffer holds `0.10 N B` windows, twice what the rehearsal phase
(`p_reh = 0.05`) consumes, so that the draw `∝ s_w` selects."""
SCALE_FLOOR = 0.1
"""SPEC: 06 (6.6) -- the per-stream scale `s_m` is floored at 0.1."""
HEAD_OF_STREAM = {"control": ["act"], "speech": ["speech"], "pattern": ["logistic", "lorenz", "eca"],
                  "vision": ["image"], "interleave": ["speech"]}
TRAIN_FACTOR = 3.0
"""SPEC: 09 §9.1 -- training FLOPs are three times the forward FLOPs. The same constant
`scripts/tempo_cont.py` uses, so that the two sides of K-CONT are counted alike."""
DEFAULT_LIFE_EPISODES = 20
"""SPEC: 05 §5.7 -- the K-MORPH controls follow a budget fraction; without an explicit
`F_budget` the fraction is `episode / 20` (the screens' 20 episodes)."""


def gstat_check(losses: list[float], sigma_e: float, *, rate_factor: float = G_STAT_RATE_FACTOR,
                sigma_factor: float = G_STAT_SIGMA_FACTOR, n_skip: int = 0) -> dict:
    """The G-STAT statistic on the per-step training losses of one episode (01 §1.4).

    With `l` the losses after the first `n_skip` (rehearsal) steps and `k = max(1, round(N/10))`:
    `delta_first = mean(l[0:k]) - mean(l[k:2k])`, `delta_last = mean(l[N-2k:N-k]) - mean(l[N-k:N])`,
    `se_last = std(l[N-2k:N]) sqrt(2/k)`; ok when `delta_last < rate_factor max(delta_first, 0)`
    or `|delta_last| < sigma_factor max(sigma_e, se_last)` (two-sided: a rising end is not a
    finished transient either)."""
    l_all = np.asarray(losses, dtype=np.float64)
    l = l_all[int(n_skip):] if 0 < n_skip < len(l_all) - 1 else l_all
    N = len(l)
    if N < 2:
        return {"ok": True, "delta_first": 0.0, "delta_last": 0.0, "k": 0, "se_last": 0.0}
    k = max(1, int(round(N / 10)))
    if 2 * k > N:
        k = max(1, N // 2)
    d_first = float(l[:k].mean() - l[k:2 * k].mean())
    d_last = float(l[N - 2 * k:N - k].mean() - l[N - k:].mean())
    tail = l[N - 2 * k:]
    se_last = float(tail.std(ddof=1) * math.sqrt(2.0 / k)) if tail.size >= 8 else 0.0     # no SE from a handful of steps
    ok = (d_last < rate_factor * max(d_first, 0.0) and d_last > -sigma_factor * max(sigma_e, se_last)) \
        or (abs(d_last) < sigma_factor * max(sigma_e, se_last))
    return {"ok": bool(ok), "delta_first": d_first, "delta_last": d_last, "k": k, "se_last": se_last}


@dataclasses.dataclass(frozen=True)
class LifeConfig:
    """The knobs of one life. SPEC: 01 §1.2, 06, 10 §10.3."""

    N: int = 250                       # steps per episode (the screens' value; 1000 in production)
    B: int = 32                        # lanes
    lr_muon: float = 0.02
    lr_emb: float = 3e-3
    lr_vec: float = 1e-3
    lr_head: float = 1e-3
    weight_decay: float = 0.1
    p_warm: float = 0.02
    p_dec: float = 0.8
    clip: float = 1.0
    micro_batch: int = 0               # 0 = the whole batch
    estimator: str = "reinforce"       # "reinforce" | "restem" | "grpo" | "es" | "none" | "random"
    svf: bool = False                  # arm (e): singular-direction gains instead of unit gains (04 §4.8)
    genome: bool = True                # False: gamma = 0 throughout (arm (a))
    free_gains: bool = True            # arm (d): gains free at tau_1 but no regression/regeneration
    regenerate: bool = True            # False: carry the gains over (arm (d))
    homeostasis: bool = True           # the zeta rule of 07
    kappa: float = KAPPA
    reverse: bool = True               # K-REV: the read order
    morphology: str = "genome"         # "genome" | "schedule" | "oneshot" | "flat" | "primordium"
    rehearsal: str = "salience"        # "salience" | "uniform" | "none"
    sigma_e: float = 0.003             # for G-STAT; measured in R1
    lambda_lat: float = 0.0            # the latent regression weight of (2.6): 0 after R1 (11 §P-C.23); 0.1 is the recorded ablation
    eval_every: int = 25               # steps
    n_eval: int = 64                   # evaluation windows per stream
    n_fitness: int = 32                # fitness windows per stream (the reward, 06 §6.3)
    n_eval_stream: int = 32            # evaluation windows for a non-text stream (one size, every evaluation)
    n_rollout_eval: int = 16           # greedy rollouts per reward-bearing stream (evaluation and fitness)
    batched_rollouts: bool = True      # step a stream's evaluation rollouts in lockstep (08 §8.3)
    n_probe_windows: int = N_PROBE_WINDOWS
    n_delta: int = N_DELTA
    G_proposals: int = G_PROPOSALS
    n_probe_steps: int = N_PROBE_STEPS
    null_probe: bool = True            # estimator 2: the Delta g = 0 probe as the control variate (06 §6.6)
    F_budget: float = 0.0              # the life's compute budget in primordium-episodes (0: none; K-MORPH)
    device: str = "cpu"
    autocast: bool = False
    seed: int = 0                      # the model seed (initialisation, proposals)
    order_seed: Optional[int] = None   # the window order (CRN across arms; defaults to `seed`)
    n_curriculum_eval: int = 8         # greedy instances per solve-rate query of the task curriculum


class Agent(nn.Module):
    """Every trainable part of the agent besides the genome machinery: the backbone, the
    interface (encoders, the perception stream, the heads). SPEC: 01 Definition 1."""

    def __init__(self, cfg: TempoConfig, codec: SpanCodec, *, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(seed)
        self.backbone = Backbone(cfg, seed=seed)
        self.codec = codec
        # the encoder is frozen (it defines x_bar); the decoder trains with the backbone (02 §2.2)
        for p in self.codec.encoder_parameters():
            p.requires_grad_(False)
        for p in self.codec.decoder_parameters():
            p.requires_grad_(True)
        self.text_in = TextIn(cfg.vocab, cfg.K, cfg.d_x)
        self.perception = Perception(cfg.d_x, len(MODALITIES))
        self.enc = nn.ModuleDict({
            "image": ImageEncoder(cfg.d_x), "speech": SpeechEncoder(cfg.d_x),
            "prop": LinearEncoder(8, cfg.d_x), "reward": LinearEncoder(1, cfg.d_x),
            "logistic": LinearEncoder(8, cfg.d_x), "lorenz": LinearEncoder(3, cfg.d_x),
            "eca": LinearEncoder(64, cfg.d_x),
        })
        self.heads = nn.ModuleDict({
            "act": GaussianHead(cfg.d_x, 2), "speech": GaussianHead(cfg.d_x, 64),
            "logistic": GaussianHead(cfg.d_x, 8), "lorenz": GaussianHead(cfg.d_x, 3),
            "eca": BernoulliHead(cfg.d_x, 64), "image": ImageDecoder(cfg.d_x),
        })
        self.insulation = Insulation()

    # ------------------------------------------------------------------ encode
    def encode(self, modality: str, raw: torch.Tensor) -> torch.Tensor:
        if modality in ("text", "task"):
            return self.text_in(raw)
        return self.enc[modality](raw)

    def features(self, windows: list[Window], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x (B, T_s, d_x)` of (2.1) for a batch of windows; also `mod (B, T_s)` and `t_w`."""
        cfg = self.cfg
        B, T_s = len(windows), cfg.T_s
        feat = torch.zeros(B, T_s, cfg.d_x, device=device)
        groups: dict[str, list] = {}
        for b, w in enumerate(windows):
            for m, (pos, raw) in w.inputs.items():
                if len(pos):
                    groups.setdefault(m, []).append((b, pos, raw))
        for m, items in groups.items():
            raw = torch.cat([r for _, _, r in items]).to(device)
            enc = self.encode(m, raw)
            bi = torch.tensor(np.concatenate([np.full(len(p), b) for b, p, _ in items]), device=device)
            pi = torch.tensor(np.concatenate([p for _, p, _ in items]), device=device)
            feat[bi, pi] = enc.to(feat.dtype)
        mod = torch.tensor(np.stack([w.mod for w in windows]), device=device)
        t_w = torch.tensor(np.stack([w.t_w for w in windows]), device=device, dtype=torch.float64)
        x = self.perception(feat.reshape(-1, cfg.d_x), mod.reshape(-1), t_w.reshape(-1)).view(B, T_s, cfg.d_x)
        return x, mod, t_w

    def head_loss(self, modality: str, y: torch.Tensor, target: torch.Tensor, kappa: float) -> torch.Tensor:
        h = self.heads[modality]
        return h.nll(insulate(y, kappa), target)

    @torch.no_grad()
    def project(self) -> None:
        """09 §9.4: the Gaussian heads' log-std bound, kept by projection after every step."""
        for h in self.heads.values():
            if hasattr(h, "project"):
                h.project()


@dataclasses.dataclass
class Lane:
    """One reader of the stream: its source, cursor, fast state and clock. SPEC: 08 §8.3."""

    index: int
    stream: str
    cursor: dict
    S: Optional[dict] = None
    cache: Optional[dict] = None
    boundary_pending: bool = True


class Life:
    """The loop of 08. `Life(cfg, life_cfg, codec, streams)`; `run_episode()` per episode."""

    def __init__(self, cfg: TempoConfig, lc: LifeConfig, codec: SpanCodec, streams: dict,
                 *, sigma_e: Optional[float] = None):
        self.cfg, self.lc = cfg, lc
        self.device = torch.device(lc.device)
        self.streams = streams
        self.seed = int(lc.seed)
        self.order_seed = int(lc.order_seed if lc.order_seed is not None else lc.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (1 << 31))
        random.seed(self.seed)
        # ------------------------------------------------------------ birth (08 §8.1)
        self.agent = Agent(cfg, codec, seed=self.seed).to(self.device)
        self.reg = RoleRegistry(self.agent.backbone, svf=lc.svf)
        self.g0 = genome_prior(cfg.d_g, self.seed).to(self.device)
        self.expr = Expression(self.reg, cfg.d_g, cfg.r, g0=self.g0, seed=self.seed).to(self.device)
        self.gains = EpisodeGains(self.reg).to(self.device)
        self.g = self.g0.clone()
        self.zeta = torch.zeros(2 * cfg.L, device=self.device)
        self.cons = Consolidator(cfg.d, cfg.d_x, cfg.d_g, d_c=cfg.d_c, chunk_size=cfg.chunk_size,
                                 seed=self.seed).to(self.device)
        # the frozen scale head (11 §P-D.8) would otherwise make Adam complain and, worse,
        # silently reappear as a trainable quantity under estimator 1
        self.cons_opt = torch.optim.Adam([p for p in self.cons.parameters() if p.requires_grad], lr=ETA_2)
        self.reinforce = Reinforce()
        self.restem = RestEM()
        self.pattern = SwitchPattern.primordium(cfg)
        self.pattern_frozen = False
        self.born = born_flags_at_birth(cfg)
        self.episode = 0
        self.N = int(lc.N)
        self.tau = TAU_FACTOR * math.sqrt(cfg.d_g)
        self.bound = GENOME_NORM_BOUND(cfg.d_g)
        self.rehearsal = Rehearsal(int(P_REHEARSAL_BUFFER * lc.N * lc.B))
        self.fit_pairs: collections.deque = collections.deque(maxlen=HISTORY)
        self.probe_set: list[tuple] = []
        self.probe_losses: dict = {}
        self.probe_t0: dict = {}                    # 11 §P-D.12: re-score at the window's own clock
        self.summaries: Optional[WindowSummaries] = None
        self.prev_summaries: Optional[WindowSummaries] = None
        self.c_prev: Optional[torch.Tensor] = None
        self.R_scale: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.record: dict = {"episodes": [], "steps": [], "evals": [], "config": cfg.as_dict(),
                             "life": dataclasses.asdict(lc), "seed": self.seed, "order_seed": self.order_seed,
                             "births": [], "probes": []}
        self.gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self.lanes: list[Lane] = []
        # 05 (5.2) counts the backbone. The interface is measured once, on the real modules,
        # with the same instrument `scripts/tempo_cont.py` uses for the token baseline, because
        # the two were 1.84x apart and the span decoder -- 44 % of the agent's forward cost --
        # was in neither `F_episode` nor `F_cum` (11 §P-C.34).
        self.iface = interface_flops(self.agent, cfg)
        self.F_window_ref = (flops_per_window(cfg, SwitchPattern.primordium(cfg))
                             + TRAIN_FACTOR * cfg.T_s * (self.iface["text"]["enc"] + self.iface["text"]["head"]))
        self.F_ref = self.F_window_ref * lc.B * lc.N
        self.F_cum = 0.0
        self.F_episode = 0.0
        self.F_probes = 0.0
        self.sigma_e = float(sigma_e if sigma_e is not None else lc.sigma_e)
        self.L_check_prev: Optional[float] = None
        self.fitness_prev: Optional[dict] = None
        self.g_prev = self.g.clone()
        self.dg_prev: Optional[torch.Tensor] = None
        self.r_in_prev: Optional[torch.Tensor] = None
        self.u_start: Optional[torch.Tensor] = None
        self._u_next: Optional[torch.Tensor] = None
        self.head_acc: dict[str, list] = {}
        self.return_acc: dict[str, list] = {}
        self.step_times: list[float] = []
        self.opt = None
        self.n_nonfinite = 0
        self.n_doublings = 0
        self.mu_episode: dict = {}
        self.budget_exhausted = False
        self._curriculum_off: Optional[str] = None
        self._last_probe: Optional[tuple] = None
        self._in_episode = False            # True between boundary_in and boundary_out (08 §8.5)
        self._resume_step: Optional[int] = None
        self._resume_info: dict = {}        # the boundary-in record of the in-flight episode
        self._learn = True
        self._expression_calls = 0
        self._regression_calls = 0
        self._consolidations = 0
        self._probe_calls = 0
        self._in_step = False

    # ================================================================== helpers
    def _optimizers(self, lr_mult: list[float]):
        """B3: Muon on matrices, AdamW on the rest, groups scaled by `exp(lambda_grp)`."""
        lc = self.lc
        muon_p, adam_p = self.agent.backbone.optimizer_groups()
        emb = list(self.agent.text_in.parameters()) + list(self.agent.perception.parameters())
        heads = list(self.agent.heads.parameters()) + list(self.agent.enc.parameters()) \
            + list(self.agent.codec.decoder_parameters())
        vec = adam_p + ([self.gains.u_diff] if lc.free_gains or lc.genome else [])
        muon = Muon(muon_p, lr=lc.lr_muon * lr_mult[0], momentum=0.95, weight_decay=lc.weight_decay)
        for g_ in muon.param_groups:
            g_["base_lr"] = lc.lr_muon * lr_mult[0]
        adam = torch.optim.AdamW([
            {"params": emb, "lr": lc.lr_emb * lr_mult[1], "base_lr": lc.lr_emb * lr_mult[1]},
            {"params": vec, "lr": lc.lr_vec * lr_mult[2], "base_lr": lc.lr_vec * lr_mult[2]},
            {"params": heads, "lr": lc.lr_head * lr_mult[3], "base_lr": lc.lr_head * lr_mult[3]},
        ], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        return muon, adam

    def _tc(self) -> TrainConfig:
        lc = self.lc
        return TrainConfig(tokens=max(8, lc.N), batch_final=max(8, (lc.B // 8) * 8), T=self.cfg.T,
                           p_warm=lc.p_warm, p_dec=lc.p_dec, clip=lc.clip)

    def _set_lr(self, n: int) -> float:
        # the *midpoint* of step `n`'s interval: `n / N` is exactly 0 at `n = 0`, where the
        # warm-up gives `lr = 0`, and `(n + 1) / N` is exactly 1 at the end, where D2Z gives
        # `lr = 0`. Neither endpoint is a learning step (11 §P-C.39).
        p = (n + 0.5) / max(1, self.N)
        tc = self._tc()
        for opt in self.opt:
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(p, g_["base_lr"], tc)
        return lr_at(p, 1.0, tc)

    def gains_dict(self) -> Optional[dict]:
        if not (self.lc.genome or self.lc.free_gains):
            return None
        return backbone_gains(self.reg, self.gains.gamma())

    def _order_seed(self, *parts: int) -> int:
        s = self.order_seed
        for p in parts:
            s = (s * 1_000_003 + int(p) * 7919 + 1) % (1 << 31)
        return s

    @property
    def n_rehearsal_steps(self) -> int:
        return int(P_REHEARSAL * self.N) if self.lc.rehearsal != "none" and len(self.rehearsal) else 0

    # ================================================================== lanes (08 §8.3)
    def _assign_lanes(self, mu: torch.Tensor) -> None:
        """B4: lanes per stream by systematic rounding of the episode's mixture; the window
        order is a function of the order seed alone (common random numbers across arms)."""
        lc = self.lc
        present = [s for s in STREAMS if s in self.streams]
        mu_np = np.array([float(mu[STREAMS.index(s)]) for s in present])
        mu_np = mu_np / mu_np.sum()
        u = float(np.random.default_rng(self._order_seed(self.episode)).random())
        counts = systematic_counts(lc.B, mu_np, u)
        assignment = []
        for s, n in zip(present, counts):
            assignment += [s] * int(n)
        new_lanes = []
        for b in range(lc.B):
            s = assignment[b]
            old = self.lanes[b] if b < len(self.lanes) else None
            if old is not None and old.stream == s:
                new_lanes.append(old)
            else:
                new_lanes.append(Lane(b, s, self.streams[s].cursor(b, self._order_seed(self.episode, b)), None, None, True))
        self.lanes = new_lanes

    def _lane_states(self, rows: list[Lane], windows: Optional[list[Window]] = None) -> dict:
        """Stack the entering fast state of each batch row `(B, h, dh, dh)` per delta layer:
        the row's own `state_in` when it carries one (a group row: the forked state under
        which its samples were drawn), zeros at a boundary, the lane's state otherwise."""
        B = len(rows)
        state = self.agent.backbone.init_state(B, self.device)
        for i, ln in enumerate(rows):
            w = windows[i] if windows is not None else None
            if w is not None and w.state_in is not None:
                for l in state:
                    state[l][i] = w.state_in[l][0].to(state[l].dtype)
            elif ln.S is not None and not ln.boundary_pending:
                for l in state:
                    state[l][i] = ln.S[l][0]
        return state

    def _store_states(self, rows: list[Lane], windows: list[Window], state: dict) -> None:
        """A lane continues from its rollout-0 row; the window-local cache is dropped."""
        seen = set()
        for i, ln in enumerate(rows):
            if ln.index in seen or windows[i].rollout != 0:
                continue
            seen.add(ln.index)
            ln.S = {l: s[i:i + 1].detach().clone() for l, s in state.items()}
            ln.cache = None
            ln.boundary_pending = False

    # ================================================================== the actor (08 §8.4)
    @torch.no_grad()
    def perceive(self, lane: Lane, raw: torch.Tensor, modality: str, t_w: float) -> torch.Tensor:
        """One span on the lane's live state: tau_0 in place, the KV cache appended."""
        raw = raw.to(self.device)
        feat = self.agent.encode(modality, raw[None])
        mod = torch.tensor([MODALITY_INDEX[modality]], device=self.device)
        x = self.agent.perception(feat, mod, torch.tensor([t_w], dtype=torch.float64, device=self.device))
        state = None if lane.S is None or lane.boundary_pending else lane.S
        if lane.boundary_pending:
            lane.cache = None
        y, S, C, _ = self.agent.backbone(x[None], state=state, cache=lane.cache, pattern=self.pattern,
                                         gains=self.gains_dict(), zeta=self.zeta)
        lane.S, lane.cache, lane.boundary_pending = S, C, False
        return y[0, 0]

    @torch.no_grad()
    def perceive_many(self, lanes: list[Lane], raws, modality: str, t_w) -> torch.Tensor:
        """One backbone step for several lanes at once (08 §8.3). `raws` is a sequence of
        per-lane raw inputs of one modality and `t_w` their wall-clock times; the returned
        `y_hat (n, d_x)` is row-aligned with `lanes`.

        **Every** row steps, always. There is no mask: the first form let a caller hold a row
        inactive, which made the lanes' KV caches diverge in length, and a batched call whose
        caches differ has no batched cache at all -- so the call silently ran with no cache and
        then wrote the resulting length-one cache back over every *active* lane, destroying the
        window's attention context permanently (11 §P-C.35). A caller whose branch has finished
        steps it with a pad and ignores the result; a finished branch's future is read by
        nobody, while an active branch's context is read by everything.

        This is the same computation as `perceive`, batched: the rollouts of a group and of an
        evaluation set are independent, and at batch size one the card is idle. "The same
        computation" is exact for the perception itself and for **greedy** decoding, which is
        what every recorded number goes through (10 §10.1). Under *sampling* the two paths draw
        from one generator in a different order -- one batched draw against `n` row draws -- so
        a sampled rollout's tokens differ between them while their distribution does not; the
        run stays deterministic given its seeds. `test_life.py` asserts the exact equality on
        the greedy path and the determinism of the sampled one.
        """
        n = len(lanes)
        raw = torch.stack([r.to(self.device) for r in raws]) if not torch.is_tensor(raws) else raws.to(self.device)
        feat = self.agent.encode(modality, raw)
        mod = torch.full((n,), MODALITY_INDEX[modality], device=self.device)
        t = torch.tensor(list(t_w), dtype=torch.float64, device=self.device)
        x = self.agent.perception(feat, mod, t)[:, None, :]                      # (n, 1, d_x)
        fresh = [ln.S is None or ln.boundary_pending for ln in lanes]
        state = self.agent.backbone.init_state(n, self.device)
        for i, ln in enumerate(lanes):
            if not fresh[i]:
                for l in state:
                    state[l][i] = ln.S[l][0]
        caches = [None if fresh[i] else lanes[i].cache for i in range(n)]
        cache = None
        if caches and all(c is not None for c in caches):
            lens = {tuple(k.shape) for c in caches for k, _ in c.values()}
            if len(lens) != 1:
                raise RuntimeError(
                    "perceive_many needs the lanes in one cache phase; got lengths "
                    f"{sorted({int(k.shape[2]) for c in caches for k, _ in c.values()})}. "
                    "Step the group in lockstep (every row every span) or start a new window.")
            cache = {l: (torch.cat([c[l][0] for c in caches]), torch.cat([c[l][1] for c in caches]))
                     for l in caches[0]}
        elif caches and any(c is not None for c in caches):
            raise RuntimeError("perceive_many needs every lane fresh or every lane cached, not a mixture")
        y, S, C, _ = self.agent.backbone(x, state=state, cache=cache, pattern=self.pattern,
                                         gains=self.gains_dict(), zeta=self.zeta)
        for i, ln in enumerate(lanes):
            ln.S = {l: s[i:i + 1].detach().clone() for l, s in S.items()}
            ln.cache = {l: (k[i:i + 1].clone(), v[i:i + 1].clone()) for l, (k, v) in C.items()} if C else None
            ln.boundary_pending = False
        return y[:, 0]

    @torch.no_grad()
    def decode_text_many(self, y_hat: torch.Tensor, generator: torch.Generator, *, greedy: bool = False
                         ) -> torch.Tensor:
        """`decode_text` on a batch of `y_hat (n, d_x)`: `(n, K)` token spans."""
        return self.agent.codec.sample(y_hat, temperature=0.0 if greedy else 1.0, generator=generator).cpu()

    @torch.no_grad()
    def decode_head_many(self, y_hat: torch.Tensor, head: str, generator: torch.Generator, *,
                         greedy: bool = False) -> torch.Tensor:
        """`decode_head` on a batch of `y_hat (n, d_x)`."""
        h = self.agent.heads[head]
        if greedy:
            if hasattr(h, "mean"):
                return h.mean(y_hat).float().cpu()
            return (torch.sigmoid(h.logits(y_hat).float()) > 0.5).to(torch.float32).cpu()
        return h.sample(y_hat, generator=generator).float().cpu()

    @torch.no_grad()
    def decode_text(self, y_hat: torch.Tensor, generator: torch.Generator, *, greedy: bool = False) -> torch.Tensor:
        return self.agent.codec.sample(y_hat[None], temperature=0.0 if greedy else 1.0, generator=generator)[0].cpu()

    @torch.no_grad()
    def decode_head(self, y_hat: torch.Tensor, head: str, generator: torch.Generator, *, greedy: bool = False
                    ) -> torch.Tensor:
        h = self.agent.heads[head]
        if greedy:
            if hasattr(h, "mean"):
                return h.mean(y_hat[None])[0].float().cpu()
            return (torch.sigmoid(h.logits(y_hat[None]).float()) > 0.5).to(torch.float32)[0].cpu()
        return h.sample(y_hat[None], generator=generator)[0].float().cpu()

    def fork(self, lane: Lane):
        return (None if lane.S is None else {l: s.clone() for l, s in lane.S.items()},
                None if lane.cache is None else {l: (k.clone(), v.clone()) for l, (k, v) in lane.cache.items()},
                lane.boundary_pending)

    def restore(self, lane: Lane, snapshot) -> None:
        S, C, bp = snapshot
        lane.S = None if S is None else {l: s.clone() for l, s in S.items()}
        lane.cache = None if C is None else {l: (k.clone(), v.clone()) for l, (k, v) in C.items()}
        lane.boundary_pending = bp

    def state_of(self, lane: Lane) -> Optional[dict]:
        if lane.S is None or lane.boundary_pending:
            return None
        return {l: s.clone() for l, s in lane.S.items()}

    # ================================================================== the loss (02 §2.4, §2.6)
    def window_loss(self, windows: list[Window], state: dict, *, train: bool = True, frac: float = 1.0
                    ) -> tuple[torch.Tensor, dict, dict]:
        """The summed loss of a batch of windows over their scored positions, the new states,
        and the per-stream bookkeeping (`loss_by_window`, `tokens_by_window`, `h_bar`, the
        heads' own losses)."""
        cfg = self.cfg
        B = len(windows)
        x, mod, _ = self.agent.features(windows, self.device)
        y_hat, state_out, _, hidden = self.agent.backbone(
            x, state=state, pattern=self.pattern, gains=self.gains_dict(), zeta=self.zeta)
        per_window = torch.zeros(B, device=self.device)         # the likelihood terms only (nats)
        latent = torch.zeros(B, device=self.device)             # lambda_lat ||y - x_bar||^2, reported apart
        tokens = torch.zeros(B, device=self.device)
        total = y_hat.new_zeros(())
        head_acc: dict[str, tuple[float, float]] = {}
        tg = window_batch_targets(windows)
        for m, (bi, pi, raw) in tg.items():
            bi_t, pi_t = torch.tensor(bi, device=self.device), torch.tensor(pi, device=self.device)
            y = y_hat[bi_t, pi_t]
            raw = raw.to(self.device)
            keep = torch.tensor(np.array([windows[b].score_mask[p] for b, p in zip(bi, pi)]), device=self.device)
            if not keep.any():
                continue
            if m in ("text", "task"):
                # (2.6): the NLL through the decoder is the score; the latent regression term
                # trains but is not a likelihood and is recorded separately; at lambda_lat = 0
                # the encoder does not run at all (it is not part of the training FLOPs)
                nll, zl = self.agent.codec.nll(y, raw, z_loss=Z_LOSS)
                lam = self.lc.lambda_lat
                if lam > 0:
                    with torch.no_grad():
                        x_bar, _, _ = self.agent.codec.encode(raw)
                    lat = (y.float() - x_bar.float()).square().sum(-1)
                else:
                    lat = torch.zeros_like(nll)
                total = total + ((nll + lam * lat) * keep).sum() + (zl * keep).sum()
                per_window.index_add_(0, bi_t, (nll * keep).detach())
                latent.index_add_(0, bi_t, (lam * lat * keep).detach())
                tokens.index_add_(0, bi_t, keep.float() * cfg.K)
            else:
                kappa = self.agent.insulation.value(m, frac, self.episode) if train else 1.0
                loss_span = self.agent.head_loss(m, y, raw, kappa) * keep
                total = total + loss_span.sum()
                per_window.index_add_(0, bi_t, loss_span.detach())
                tokens.index_add_(0, bi_t, keep.float())
                head_acc[m] = (float(loss_span.detach().sum()), float(keep.sum()))
        # the policy-gradient term over complete groups in the batch (02 (2.12))
        groups: dict[int, list[int]] = {}
        for b, w in enumerate(windows):
            if w.rl is not None:
                groups.setdefault((w.stream, w.rl.group), []).append(b)
        rl_total = y_hat.new_zeros(())
        for gid, members in groups.items():
            if len(members) < 2:
                continue
            rewards = torch.tensor([windows[b].rl.reward for b in members])
            adv, keep_g = group_advantage(rewards)
            if not keep_g:
                continue
            lps = []
            for b in members:
                rl = windows[b].rl
                pos = torch.tensor(rl.positions, device=self.device)
                y = y_hat[b, pos]
                samples = rl.samples.to(self.device)
                if rl.modality == "text":
                    nll, _ = self.agent.codec.nll(y, samples)
                    lp = -nll
                else:
                    kappa = self.agent.insulation.value(rl.modality, frac, self.episode) if train else 1.0
                    lp = self.agent.heads[rl.modality].log_prob(insulate(y, kappa), samples)
                lps.append(lp)
            T_max = max(int(l.numel()) for l in lps)
            LP = torch.zeros(len(members), T_max, device=self.device)
            MK = torch.zeros(len(members), T_max, device=self.device)
            for i, lp in enumerate(lps):
                LP[i, :lp.numel()] = lp
                MK[i, :lp.numel()] = 1.0
            rl_total = rl_total + grpo_loss(LP, LP.detach(), adv.to(self.device), MK) * MK.sum()
        total = total + rl_total
        h_bar = hidden.detach().float().mean(1)
        info = {"loss_by_window": per_window, "tokens_by_window": tokens, "h_bar": h_bar,
                "latent_by_window": latent, "rl": float(rl_total.detach()), "head_acc": head_acc}
        return total, state_out, info

    # ================================================================== evaluation (08 §8.3, 10 §10.1)
    @torch.no_grad()
    def _score_windows(self, windows: list[Window]) -> torch.Tensor:
        """Fresh-state per-token (text) or per-span mean loss of each window."""
        out = []
        bs = max(1, min(16, self.lc.B))
        for i in range(0, len(windows), bs):
            ws = windows[i:i + bs]
            _, _, info = self.window_loss(ws, self.agent.backbone.init_state(len(ws), self.device), train=False)
            out.append(info["loss_by_window"] / info["tokens_by_window"].clamp_min(1.0))
        return torch.cat(out).cpu() if out else torch.zeros(0)

    @torch.no_grad()
    def evaluate(self, stream: str, n: Optional[int] = None) -> dict:
        """The fixed evaluation set of a stream: fresh-state loss per token/span, the
        continuing-state score over the first eight windows on one lane, and, for a
        reward-bearing stream, the mean greedy reward on its evaluation rollouts."""
        n = n or self.lc.n_eval
        windows = self.streams[stream].eval_windows(n)
        was = self.agent.training
        self.agent.eval()
        per = self._score_windows(windows)
        state = self.agent.backbone.init_state(1, self.device)
        tot_cont, cnt_cont = 0.0, 0.0
        for w in windows[:min(len(windows), 8)]:
            _, state, info = self.window_loss([w], state, train=False)
            tot_cont += float(info["loss_by_window"].sum())
            cnt_cont += float(info["tokens_by_window"].sum())
        out = {"stream": stream, "fresh": float(per.mean()) if per.numel() else float("nan"),
               "continuing": tot_cont / max(cnt_cont, 1.0), "n": len(windows)}
        if stream in REWARD_STREAMS and hasattr(self.streams[stream], "evaluate"):
            rewards = self._rollout_rewards(stream, fitness=False)
            out["reward"] = float(np.mean(rewards)) if rewards else float("nan")
            out["n_rollouts"] = len(rewards)
        self.agent.train(was)
        return out

    def _rollout_rewards(self, stream: str, *, fitness: bool) -> list[float]:
        lane = Lane(2000 + STREAMS.index(stream), stream, {}, None, None, True)
        return [float(r) for r in self.streams[stream].evaluate(self, lane, self.lc.n_rollout_eval, fitness=fitness)]

    @torch.no_grad()
    def fitness(self, stream: str) -> tuple[float, torch.Tensor]:
        """The stream's fitness score on its reserved fitness set from a fresh state
        (06 §6.3): the mean per-token/per-span loss (text, patterns, interleaving) or minus
        the mean greedy reward (the reward-bearing streams); with its per-window values."""
        was = self.agent.training
        self.agent.eval()
        try:
            if stream in REWARD_STREAMS and hasattr(self.streams[stream], "evaluate"):
                vals = torch.tensor([-r for r in self._rollout_rewards(stream, fitness=True)], dtype=torch.float32)
            else:
                vals = self._score_windows(self.streams[stream].fitness_windows(self.lc.n_fitness))
            if vals.numel() == 0:
                raise ValueError(f"no fitness windows for {stream}")
            return float(vals.mean()), vals
        finally:
            self.agent.train(was)

    def _fitness_all(self) -> dict[str, float]:
        """Every stream present in the life; the scale `s_m` is set at a stream's first evaluation."""
        out = {}
        for s in [s for s in STREAMS if s in self.streams]:
            m, vals = self.fitness(s)
            if s not in self.scales:
                self.scales[s] = max(SCALE_FLOOR, float(vals.std(unbiased=False)) if vals.numel() > 1 else SCALE_FLOOR)
            out[s] = m
        return out

    # ================================================================== the episode (08 §8.2)
    def _n_eval(self, stream: str) -> int:
        """One fixed set size per stream, for every evaluation of a run. The mid-episode and
        end-of-episode series used 16 and 32 windows, so the curve stepped at `n = N` for a
        reason that had nothing to do with the agent, and the two series were not on the same
        set -- `EvalSet.sample_documents` draws `16 n` candidates, so a 16-window set is not a
        subset of a 32-window one (11 §P-C.36)."""
        return self.lc.n_eval if stream == "text" else min(self.lc.n_eval, self.lc.n_eval_stream)

    def _episode_frac(self) -> float:
        """The K-MORPH controls' schedule variable: the spent fraction of the budget."""
        if self.lc.F_budget > 0:
            return min(1.0, self.F_cum / (self.lc.F_budget * self.F_ref))
        return min(1.0, self.episode / DEFAULT_LIFE_EPISODES)

    def boundary_in(self) -> dict:
        """B1-B5."""
        cfg, lc = self.cfg, self.lc
        if lc.svf:                                                 # the SVF ablation caches the decomposition (04 §4.8)
            self.agent.backbone.refresh_svf()
        # B1: expression, once (reused when the boundary-out already expressed the new genome)
        if lc.genome and lc.regenerate:
            if self._u_next is None:
                self._u_next = self.expr.express(self.g)
                self._expression_calls += 1
            self.gains.set_from(self._u_next)
            self._u_next = None
        elif not lc.genome and not lc.free_gains:
            self.gains.set_from(torch.zeros(self.reg.n_gamma, device=self.device))
        self.u_start = self.gains.u().detach().clone()
        gamma = self.gains.gamma().detach()
        # B2: the switch pattern and births
        born: list[str] = []
        if self.pattern_frozen:
            new = self.pattern
        elif lc.morphology == "genome" and lc.genome:
            new = SwitchPattern.from_gains(cfg, gamma[self.reg.slice("MORPH_Q")].cpu(),
                                          gamma[self.reg.slice("MORPH_H")].cpu(), gamma[self.reg.slice("MORPH_F")].cpu())
        elif lc.morphology == "flat":
            new = SwitchPattern.adult(cfg)
        elif lc.morphology == "primordium":
            new = SwitchPattern.primordium(cfg)
        else:
            new = control_pattern(cfg, lc.morphology, self._episode_frac())
        if new != self.pattern or any(not v for v in self.born.values()):
            if lc.morphology == "oneshot" and new != self.pattern:
                born = hypercloning_split(self.agent.backbone, self.pattern, self.born, self.gen)
            else:
                born = apply_births(self.agent.backbone, self.pattern, new, self.born, self.gen)
        if born:
            self.record["births"].append({"episode": self.episode, "groups": born, "mode": lc.morphology})
        self.pattern = new
        # B3: optimisers
        lr_mult = lr_multipliers(self.reg, gamma) if lc.genome else [1.0] * 4
        self.opt = self._optimizers(lr_mult)
        # B4: the mixture and the lanes
        mu = mixture(self.reg, gamma) if lc.genome else torch.full((len(STREAMS),), 1.0 / len(STREAMS), dtype=torch.float64)
        self._assign_lanes(mu)
        self.mu_episode = {s: float(mu[STREAMS.index(s)]) for s in STREAMS}
        # B5: the rehearsal buffer of the previous episode is already attached; fresh summaries
        self.summaries = WindowSummaries(cfg.d, cfg.d_x, self.N)
        self.head_acc, self.return_acc = {}, {}
        self.F_episode = 0.0
        self.n_nonfinite = 0
        return {"lr_mult": lr_mult, "mu": [float(v) for v in mu], "pattern": self.pattern.to_dict(),
                "active_params": n_active_params(cfg, self.pattern), "born": born}

    def _draw_windows(self, n: int) -> list[tuple[Lane, list[Window]]]:
        """S1: the rehearsal buffer for the first `p_reh N` steps, the lanes otherwise; an
        interactive lane contributes its supervised row and its rollout group."""
        lc = self.lc
        out = []
        if n < self.n_rehearsal_steps:
            for b, (ident, t_w0) in enumerate(self.rehearsal.draw(lc.B, self.gen)):
                w = self.streams[ident[0]].fetch(ident, t0=t_w0)
                w.boundary = True
                out.append((Lane(1000 + b, ident[0], {}, None, None, True), [w]))
            return out
        for lane in self.lanes:
            ws = self.streams[lane.stream].next_windows(lane.cursor, self, lane, self.gen)
            if ws[0].boundary:
                lane.boundary_pending = True
            out.append((lane, ws))
        return out

    def _interface_flops(self, windows: list[Window], *, train: bool) -> float:
        """The interface's forward (and, when training, backward) cost of a batch: each input
        span pays its modality's encoder, each **scored** target span pays its modality's head.
        05 (5.2) counts the backbone alone, and leaving the interface out did not merely rescale
        the FLOP axis -- it changed the ratio between two morphologies, which is what an
        equal-compute comparison reads (11 §P-C.34)."""
        mult = TRAIN_FACTOR if train else 1.0
        total = 0.0
        for w in windows:
            for m, (pos, _) in w.inputs.items():
                f = self.iface.get(m)
                if f is not None:
                    total += len(pos) * f["enc"]
            for m, (pos, _) in w.targets.items():
                f = self.iface.get(m)
                if f is None or len(pos) == 0:
                    continue
                scored = int(np.asarray([w.score_mask[q] for q in pos]).sum())
                total += scored * f["head"]
        return mult * total

    def _denominator(self, windows: list[Window]) -> float:
        """08 §8.2 S3: the batch's summed loss is divided by **the batch's token count** -- the
        number of things the loss actually sums over.

        That is the scored supervised tokens plus the policy-gradient positions of the complete
        groups. The first form used `w.n_tokens or w.T_s`, and a rollout row has `n_tokens = 0`
        because it carries no supervised target, so `0 or T_s` charged it a whole window: at the
        blend, 70 % of the denominator came from rows with **zero** scored positions, and the
        inflation ran from 1.0x on a text-only batch to 16x on a control-only one. Since the
        stream mixture is a genome role (04 §4.4), the genome was moving the effective step size
        through a bookkeeping artefact rather than through the learning-rate role it is supposed
        to use -- and `clip_grad_norm_` is not scale-invariant, so the clip moved with it
        (11 §P-C.36).
        """
        total = sum(scored_tokens(w, self.cfg.K) for w in windows)
        groups: dict[int, list[Window]] = {}
        for w in windows:
            if w.rl is not None:
                groups.setdefault((w.stream, w.rl.group), []).append(w)
        for members in groups.values():
            if len(members) < 2:
                continue
            _, keep = group_advantage(torch.tensor([w.rl.reward for w in members]))
            if keep:
                total += sum(len(w.rl.positions) for w in members)
        return max(1.0, float(total))

    def train_step(self, n: int) -> dict:
        """S1-S7 for step `n`; with `learn=False` the optimiser step is skipped (08 §8.4)."""
        lc, cfg = self.lc, self.cfg
        self._in_step = True
        lr_frac = self._set_lr(n)
        frac = n / max(1, self.N)
        pairs = self._draw_windows(n)
        rows = [ln for ln, ws in pairs for _ in ws]                    # one lane per batch row
        windows = [w for _, ws in pairs for w in ws]
        rehearsed = n < self.n_rehearsal_steps
        t0 = time.perf_counter()
        micro = lc.B if lc.micro_batch <= 0 else min(lc.B, lc.micro_batch)
        cuts, count = [0], 0                                           # split by lanes, never inside a group
        for i, (ln, ws) in enumerate(pairs):
            count += len(ws)
            if count >= micro and i + 1 < len(pairs):
                cuts.append(sum(len(w) for _, w in pairs[:i + 1]))
                count = 0
        cuts.append(len(windows))
        for opt in self.opt:
            opt.zero_grad(set_to_none=True)
        loss_sum, tok_sum, lat_sum = 0.0, 0.0, 0.0
        infos = []
        denom = self._denominator(windows)
        probe_windows, probe_states = [], []
        for a, b_ in zip(cuts[:-1], cuts[1:]):
            if b_ <= a:
                continue
            ws, ls = windows[a:b_], rows[a:b_]
            state = self._lane_states(ls, ws)
            if len(probe_windows) < lc.n_probe_windows:
                for i, w in enumerate(ws):
                    if len(probe_windows) < lc.n_probe_windows and w.rollout == 0:
                        probe_windows.append(w)
                        probe_states.append({l: s[i:i + 1].detach().clone() for l, s in state.items()})
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=lc.autocast), \
                    torch.set_grad_enabled(self._learn):
                total, state_out, info = self.window_loss(ws, state, train=True, frac=frac)
            if self._learn:
                (total / denom).backward()
            self._store_states(ls, ws, self.agent.backbone.detach_state(state_out))
            loss_sum += float(info["loss_by_window"].sum())
            lat_sum += float(info["latent_by_window"].sum())
            tok_sum += float(info["tokens_by_window"].sum())
            for m, (s_, c_) in info["head_acc"].items():
                acc = self.head_acc.setdefault(m, [0.0, 0.0])
                acc[0] += s_
                acc[1] += c_
            infos.append((ws, info))
        self._last_probe = (probe_windows, probe_states)
        gnorm = float("nan")
        nonfinite = False
        if self._learn:
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in self.agent.parameters() if p.requires_grad] + [self.gains.u_diff], lc.clip))
            if not math.isfinite(gnorm):
                # 09 §9.5: the step is skipped and counted. One row per step, always: a second
                # row here pushed the episode's first steps out of `steps[-N:]`, which is the
                # window G-STAT reads (11 §P-C.36)
                nonfinite = True
                self.n_nonfinite += 1
                if self.n_nonfinite > 3:
                    raise FloatingPointError(f"more than three non-finite steps in episode {self.episode} (09 §9.5)")
            else:
                for opt in self.opt:
                    opt.step()
                self.agent.project()
            for opt in self.opt:
                opt.zero_grad(set_to_none=True)
        # S6: summaries; a rollout row without likelihood targets carries -return / R_scale (06 (6.1))
        for ws, info in infos:
            ell = info["loss_by_window"] / info["tokens_by_window"].clamp_min(1.0)
            is_return = []
            for i, w in enumerate(ws):
                ret = w.rl is not None and not w.targets
                is_return.append(ret)
                if ret:
                    ell[i] = -w.rl.reward / self.R_scale.get(w.stream, 1.0)
                if w.rl is not None and w.stream == "control":
                    acc = self.return_acc.setdefault(w.stream, [0.0, 0.0])
                    acc[0] += float(w.rl.reward)
                    acc[1] += 1.0
            mu_bar = torch.stack([self.agent.perception.mu(torch.tensor(w.mod, device=self.device)).mean(0)
                                  for w in ws]).detach()
            self.summaries.append(info["h_bar"], ell, mu_bar, n, [w.ident for w in ws], [w.stream for w in ws],
                                  [rehearsed] * len(ws), is_return,
                                  [float(w.t_w[0]) if len(w.t_w) else 0.0 for w in ws])
        # S7: compute accounting: the training rows and the rollouts' inference forwards (08 §8.6)
        F_train = (flops_per_window(cfg, self.pattern, train=self._learn) * len(windows)
                   + self._interface_flops(windows, train=self._learn))
        # a row is a rollout when it carries an emission, not when its index is non-zero: the
        # control stream has no teacher-forced row, so its rollout 0 was never charged and a
        # control-heavy mixture got 1/G of its acting compute for free (11 §P-C.36)
        F_roll = flops_per_window(cfg, self.pattern, train=False) * sum(
            (w.n_used or w.T_s) / w.T_s for w in windows if w.rl is not None)
        self.F_episode += F_train + F_roll
        self.F_cum += F_train + F_roll
        dt = time.perf_counter() - t0
        self.step_times.append(dt)
        rec = {"n": n, "episode": self.episode, "loss": loss_sum / max(tok_sum, 1.0), "latent": lat_sum / max(tok_sum, 1.0),
               "lr_frac": lr_frac, "gnorm": gnorm, "tokens": tok_sum, "seconds": dt, "denom": denom,
               "rl": sum(i["rl"] for _, i in infos), "F": F_train + F_roll, "rehearsed": rehearsed,
               "nonfinite": bool(nonfinite)}
        self.record["steps"].append(rec)
        self._in_step = False
        return rec

    def run_episode(self, *, log: Optional[Callable[[str], None]] = None, learn: bool = True,
                    should_pause: Optional[Callable[[int], bool]] = None) -> dict:
        """One episode: boundary-in, N steps, the evaluation at `n = N`, boundary-out.
        `learn=False` is the inference path (08 §8.4): the same loop with `eta_1 = 0`.

        `should_pause(n)` is consulted after every step; when it is true the method raises
        `Paused` and the life is left resumable at step `n + 1` (08 §8.5). A life resumed
        inside an episode continues the step loop and does **not** re-run the boundary: the
        gains, the switch pattern, the optimisers and the lane assignment of that episode are
        already in the checkpoint, and re-running B1-B5 would redraw them.
        """
        lc = self.lc
        self._learn = bool(learn)
        if self._in_episode and self._resume_step is not None:
            start, info_in = int(self._resume_step), dict(self._resume_info)
            if log:
                log(f"  resuming episode {self.episode} at step {start}/{self.N}")
        else:
            start, info_in = 0, self.boundary_in()
            self._in_episode = True
        self._resume_info = dict(info_in)
        self._resume_step = None
        evals = []
        for n in range(start, self.N):
            rec = self.train_step(n)
            if lc.eval_every and (n + 1) % lc.eval_every == 0 and n + 1 < self.N:
                ev = {s: self.evaluate(s, n=self._n_eval(s)) for s in self.streams}
                evals.append({"n": n + 1, "episode": self.episode, **ev})
                self.record["evals"].append(evals[-1])
            if log and (n + 1) % max(1, self.N // 10) == 0:
                log(f"  ep {self.episode} step {n + 1}/{self.N} nll {rec['loss']:.4f} latent {rec['latent']:.3f} "
                    f"gnorm {rec['gnorm']:.2f} {rec['seconds'] * 1000:.0f} ms")
            if should_pause is not None and n + 1 < self.N and should_pause(n + 1):
                self._resume_step = n + 1
                raise Paused(self.episode, n + 1)
        # the evaluation at n = N, before the boundary changes zeta or the gains (08 §8.3)
        eval_final = {s: self.evaluate(s, n=self._n_eval(s)) for s in self.streams}
        self.record["evals"].append({"n": self.N, "episode": self.episode, **eval_final})
        info_out = self.boundary_out(log=log)
        out = {"episode": self.episode, "learn": bool(learn), **info_in, **info_out, "eval_final": eval_final}
        self.record["episodes"].append(out)
        self.episode += 1
        self._in_episode, self._resume_step = False, None
        self._learn = True
        return out

    # ================================================================== the boundary (08 §8.2 O1-O10)
    def boundary_out(self, *, log=None) -> dict:
        cfg, lc = self.cfg, self.lc
        if self._in_step:
            raise RuntimeError("G-NOFREE-INVERSE: the boundary ran inside a step")
        ws = self.summaries
        out: dict = {}
        streams_here = ws.streams_present()
        # Everything below that describes *the episode* is captured before O6's estimator runs.
        # Estimator 2 lives `G + 1` probe episodes and may adopt one of them, which replaces the
        # weights, the gains, the switch pattern, the non-finite counter and the step times with
        # the probe's. Three independent reviews found the criticality profile being measured on
        # a probe's last step under a probe's gains (11 §P-C.36); the fix is to measure the
        # episode before the estimator is allowed to touch anything.
        ep_gains = self.gains_dict()
        ep_pattern = self.pattern
        ep_u_end = self.gains.u().detach().clone()
        ep_nonfinite = self.n_nonfinite
        ep_step_times = list(self.step_times[-self.N:])
        ep_mu = dict(self.mu_episode)
        # O1: the per-stream scale of the environment losses in the summaries (06 (6.1))
        for s in streams_here:
            # every stream whose rows carry a return rather than a likelihood, not only control
            # (06 §6.1: "once per environment stream"); a stream whose rows all have targets
            # never reaches the branch because `rets` stays empty (11 §P-C.36)
            if s in REWARD_STREAMS and s not in self.R_scale:
                rets = {}
                for ident, ell_w in zip(ws.ident, ws.losses().tolist()):
                    if ident[0] == s:
                        rets[ident[:3]] = -ell_w
                if not rets:
                    continue                                   # no return-bearing row: no scale
                vals = torch.tensor(list(rets.values()))
                self.R_scale[s] = max(1.0, float(vals.std())) if vals.numel() > 1 else 1.0
                if self.R_scale[s] != 1.0:
                    ws.rescale_rows(s, 1.0 / self.R_scale[s], only_returns=True)
        # O2: G-STAT (the two-sided transient test on the episode's per-step losses after rehearsal)
        step_losses = [r["loss"] for r in self.record["steps"][-self.N:] if "loss" in r]
        gs = gstat_check(step_losses, self.sigma_e, n_skip=self.n_rehearsal_steps)
        out["gstat_delta_first"], out["gstat_delta_last"], out["gstat_se"] = gs["delta_first"], gs["delta_last"], gs["se_last"]
        out["gstat_ok"] = gs["ok"]
        out["gstat_ratio"] = gs["delta_last"] / gs["delta_first"] if gs["delta_first"] > 0 else float("nan")
        F_ep = self.F_episode                                      # before any doubling (08 §8.6)
        out["F_episode"] = F_ep                                    # F_cum and F_probes at the end of the boundary
        if not out["gstat_ok"]:
            if self.n_doublings < G_STAT_MAX_DOUBLINGS:
                self.N *= 2
                self.n_doublings += 1
                out["N_doubled_to"] = self.N
            else:
                out["gstat_flagged"] = True
        out["L_by_stream"] = {s: ws.mean_loss(stream=s) for s in streams_here}
        out["L_bar"] = ws.mean_loss()
        out["L_fin"] = self._mean_last(ws, None)
        # O3: the fitness scores on the reserved sets and the reward of the previous proposal (06 §6.3)
        fit_now = self._fitness_all()
        L_now = L_check(fit_now, self.scales)
        out.update({"fitness": fit_now, "scales": dict(self.scales), "L_check": L_now})
        r_e = None
        if self.L_check_prev is not None:
            r_e = reward_fn(self.L_check_prev, L_now, F_ep, self.F_ref)
            out["reward"] = r_e
        # O4: the regression of theta over the history of pairs (04 (4.4))
        if lc.genome and lc.regenerate:
            self.fit_pairs.append((self.g.detach().clone(), self.gains.u_target().detach().clone()))
            R_res = self.expr.fit(list(self.fit_pairs))
            self._regression_calls += 1
            out["regression_residual"] = R_res
            out["G_REGRESS"] = R_res > 0.01
        # O5: the criticality probe, on the episode's last-step windows with their entering
        # states, under the episode's gains and pattern. It runs *before* the estimator (08 §8.2)
        # so that no probe episode can move what it measures.
        prof = None
        if self._last_probe is not None and self._last_probe[0]:
            pw, ps = self._last_probe
            x, _, _ = self.agent.features(pw, self.device)
            g_probe = torch.Generator(device="cpu").manual_seed(self._order_seed(self.episode, 4242))
            prof = probe(self.agent.backbone, x.detach(), states=ps, pattern=ep_pattern, gains=ep_gains,
                         zeta=self.zeta, n_delta=lc.n_delta, generator=g_probe)
            self._probe_calls += 1
            out["Lambda"], out["Lambda_abs"], out["Lambda_se"] = prof["Lambda"], prof["Lambda_abs"], prof["Lambda_se"]
            out["lambda_block"], out["lambda_block_se"] = prof["lambda_block"], prof["lambda_block_se"]
            out["n_blocks"] = prof["n_blocks"]
            out["lambda_mix"] = {int(k): v for k, v in prof["mix"].items()}
            out["lambda_ffn"] = {int(k): v for k, v in prof["ffn"].items()}
            out["lambda_mix_abs"] = {int(k): v for k, v in prof["mix_abs"].items()}
            out["lambda_ffn_abs"] = {int(k): v for k, v in prof["ffn_abs"].items()}
            out["lambda_mix_se"] = {int(k): v for k, v in prof["mix_se"].items()}
            out["lambda_ffn_se"] = {int(k): v for k, v in prof["ffn_se"].items()}
            out["kcrit_trigger"] = crit_trigger(prof)
            refs = [w.lambda_ref for w in pw if w.lambda_ref is not None]
            out["lambda_ref"] = float(np.mean(refs)) if refs else None
            out["probe_streams"] = sorted({w.stream for w in pw})
            out["probe_seed"] = self._order_seed(self.episode, 4242)

        # O6: the reverse read, the estimator's update, the proposal (update-then-propose)
        r_in = ws.inputs(self.device)
        dg = torch.zeros(cfg.d_g, device=self.device)
        c = None
        if lc.genome and lc.estimator != "none":
            if lc.estimator == "reinforce" and r_e is not None and self.dg_prev is not None and self.r_in_prev is not None:
                out["reinforce"] = self.reinforce.update(self.cons, self.cons_opt, self.r_in_prev, self.g_prev,
                                                         self.dg_prev, r_e, reverse=lc.reverse)
            with torch.no_grad():
                R, c = self.cons.read(r_in, self.g, reverse=lc.reverse)
                mu_p, ls_p = self.cons.propose(R)
            if lc.estimator == "reinforce":
                dg = self.cons.sample(mu_p, ls_p, self.gen)
            elif lc.estimator == "random":
                dg = torch.randn(cfg.d_g, generator=self.gen).to(self.device)
                dg = dg / dg.norm() * self.tau
            elif lc.estimator in ("restem", "es", "grpo"):
                dg, kept = self._best_of_G(r_in, mu_p, ls_p, L_now, log=log)
                out["probe_kept"] = kept
            self._consolidations += 1
            out["dg_norm"] = float(dg.norm())
            out["proposal_sigma_mean"] = float(torch.exp(ls_p).mean())
            with torch.no_grad():
                s_w = self.cons.salience(c).cpu()
        else:
            s_w = None
        # O7: the genome step
        if lc.genome:
            self.g_prev = self.g.clone()
            self.g = genome_step(self.g, dg, tau=self.tau, lambda_dec=LAMBDA_DEC, bound=self.bound, prior=self.g0)
            out["g_norm"] = float((self.g - self.g0).norm())
        self.dg_prev, self.r_in_prev = dg.detach().clone(), r_in.detach().clone()
        self.L_check_prev = L_now
        self.fitness_prev = fit_now
        # O8: the zeta rule on the episode's relative profile (measured at O5)
        if lc.homeostasis and prof is not None:
            self.zeta = zeta_update(self.zeta, prof, kappa=lc.kappa)
        out["zeta"] = self.zeta.tolist()
        # O9: insulation from each head's own loss (act: minus the mean return)
        for m, (s_, c_) in self.head_acc.items():
            if c_ > 0:
                self.agent.insulation.update(m, self.episode, s_ / c_)
        if "control" in self.return_acc and self.return_acc["control"][1] > 0:
            acc = self.return_acc["control"]
            self.agent.insulation.update("act", self.episode, -acc[0] / acc[1])
        out["kappa"] = {m: self.agent.insulation.value(m, 1.0, self.episode + 1) for m in
                        sorted(set(self.head_acc) | ({"act"} if "control" in self.return_acc else set()))}
        # O10: rehearsal, the salience regression, the probe set
        if lc.rehearsal != "none":
            self.rehearsal.build(ws.ident, s_w, generator=self.gen,
                                 uniform=(lc.rehearsal == "uniform" or s_w is None), t0=ws.t0)
            out["rehearsal_mode"] = self.rehearsal.mode
        if self.prev_summaries is not None and self.c_prev is not None and self.probe_set:
            out["salience_mse"] = self._salience_regression()
        self.probe_set = [ws.ident[int(i)] for i in torch.randperm(ws.n_windows, generator=self.gen)[:N_PROBE_SET]]
        keep = set(self.probe_set)
        self.probe_losses = {ws.ident[int(i)]: float(ws.losses()[int(i)]) for i in range(ws.n_windows) if ws.ident[int(i)] in keep}
        self.probe_t0 = {ws.ident[int(i)]: (ws.t0[int(i)] if int(i) < len(ws.t0) else 0.0)
                         for i in range(ws.n_windows) if ws.ident[int(i)] in keep}
        self.prev_summaries = ws
        self.c_prev = None if c is None else c.detach().cpu()
        # the task curriculum (02 §2.9.2): seed at the first boundary, then one archive step
        if "task" in self.streams and hasattr(self.streams["task"], "curriculum_step"):
            out["curriculum"] = self._curriculum()
        # A8 in gain space: the genome's move against the fast weights' move (01 §1.3)
        if lc.genome and lc.regenerate:
            with torch.no_grad():
                self._u_next = self.expr.express(self.g)
                self._expression_calls += 1
            u_end = ep_u_end                       # the episode's, not an adopted probe's (A8)
            num = float((self._u_next - u_end).norm())
            den = float((u_end - self.u_start).norm()) if self.u_start is not None else 0.0
            out["C2_ratio"] = num / max(den, 1e-12)
            out["C2_genome_move"], out["C2_fast_move"] = num, den
        else:
            out["C2_ratio"] = 0.0
        out["active_params"] = n_active_params(cfg, ep_pattern)
        out["F_cum"], out["F_probes"] = self.F_cum, self.F_probes   # after the probes are charged
        out["step_seconds_median"] = statistics.median(ep_step_times) if ep_step_times else float("nan")
        out["n_nonfinite"] = ep_nonfinite
        out["mu"] = [ep_mu.get(s, 0.0) for s in STREAMS]
        if lc.F_budget > 0 and self.F_cum >= lc.F_budget * self.F_ref:
            self.budget_exhausted = True
        out["budget_exhausted"] = self.budget_exhausted
        return out

    @staticmethod
    def _mean_last(ws: WindowSummaries, stream: Optional[str]) -> float:
        try:
            return ws.mean_loss(last_frac=0.1, stream=stream)
        except ValueError:
            return ws.mean_loss(stream=stream)

    def _salience_regression(self) -> float:
        """(6.11): re-score last episode's probe set now; fit the salience head on the
        cached outputs of last episode's read."""
        idents = [i for i in self.probe_set if i in self.probe_losses]
        if not idents:
            return float("nan")
        # 06 (6.11)'s target is the MODEL's improvement, which it is only if the window is the
        # same at both scorings; re-fetching at t = 0 made it a difference of inputs (11 §P-D.12)
        windows = [self.streams[i[0]].fetch(i, t0=self.probe_t0.get(i, 0.0)) for i in idents]
        now = self._score_windows(windows)
        before = torch.tensor([self.probe_losses[i] for i in idents])
        target = salience_targets(before - now).to(self.device)
        # `probe_losses` is a dict, so a repeated ident kept the *last* row's loss; `index`
        # returns the *first*. The two must agree or the regression pairs one row's `c_w` with
        # another row's improvement (11 §P-C.39).
        last = {ident: j for j, ident in enumerate(self.prev_summaries.ident)}
        pos = [last[i] for i in idents]
        return salience_fit(self.cons, self.c_prev[pos].to(self.device), target)

    def _curriculum(self) -> dict:
        """The learnability-gated archive of the verifier tasks, driven by the agent's greedy
        solve rate (02 §2.9.2): seeded at the first boundary, one step per boundary after."""
        ts = self.streams["task"]
        lane = Lane(3000, "task", {}, None, None, True)
        was = self.agent.training
        self.agent.eval()

        def solve_rate(params, n_eval: int) -> float:
            n = max(1, min(int(n_eval), self.lc.n_curriculum_eval))
            hits = 0
            for i in range(n):
                seed = self._order_seed(self.episode, 9000 + i)
                inst = ts.instance(seed, params)
                lane.S, lane.cache, lane.boundary_pending = None, None, True
                with torch.no_grad():
                    _, r = ts._act(self, lane, inst, torch.Generator().manual_seed(seed), 0.0, greedy=True)
                hits += int(r > 0)
            return hits / n

        try:
            if self._curriculum_off:
                return {"seeded": False, "reason": self._curriculum_off}
            if not ts.tasks._seeded:
                try:
                    ts.seed_curriculum(solve_rate)
                except ValueError as e:
                    # a context too short for a family's lower bound (the test shape): the
                    # stream keeps drawing from the family lower bounds, and the record says so
                    self._curriculum_off = str(e)
                    return {"seeded": False, "reason": str(e)}
                return {"seeded": True, "size": len(ts.tasks.archive)}
            info = ts.curriculum_step(solve_rate, 1)
            return {"size": len(ts.tasks.archive), "step": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}}
        finally:
            self.agent.train(was)

    def _best_of_G(self, r_in, mu_p, ls_p, L_now, *, log=None) -> tuple[torch.Tensor, str]:
        """Estimator 2 (06 §6.6) and the ES arm (06 §6.8): from this checkpoint probe the null
        proposal (`dg = 0`, the control variate) and `G` proposals for `n_probe` steps each under
        common random numbers; keep the winner's post-probe state when it beats the null (its
        probe steps are lived, its compute is charged); regress the policy toward the winner."""
        cfg, lc = self.cfg, self.lc
        snapshot = self.checkpoint_state()
        G = lc.G_proposals
        if lc.estimator == "es":
            half = max(1, G // 2)
            sigma_es = self.tau / math.sqrt(cfg.d_g)
            eps = torch.randn(half, cfg.d_g, generator=self.gen).to(self.device)
            cands = [sigma_es * eps[i] for i in range(half)] + [-sigma_es * eps[i] for i in range(half)]
        else:
            cands = [self.cons.sample(mu_p, ls_p, self.gen) for _ in range(G)]
        probes = ([torch.zeros(cfg.d_g, device=self.device)] if lc.null_probe else []) + cands
        rewards, fits = [], []
        null_state, best_state, best_i, best_r = None, None, None, -float("inf")
        saved_N, saved_episode = self.N, self.episode
        F_probes = 0.0
        for i, dg_i in enumerate(probes):
            self.load_state(snapshot)
            self.N = lc.n_probe_steps
            self.g = genome_step(self.g, dg_i, tau=self.tau, lambda_dec=LAMBDA_DEC, bound=self.bound, prior=self.g0)
            self._u_next = None
            self.boundary_in()                                     # the candidate's expression, pattern, optimisers
            for n in range(self.N):
                self.train_step(n)
            fit_i = self._fitness_all()
            L_probe = L_check(fit_i, self.scales)
            r_i = reward_fn(L_now, L_probe, self.F_episode, self.F_ref * lc.n_probe_steps / saved_N)
            F_probes += self.F_episode                             # every probe's steps are charged (09 §9.1)
            rewards.append(r_i)
            fits.append(L_probe)
            if log:
                log(f"  probe {i}{' (null)' if (lc.null_probe and i == 0) else ''}: L {L_probe:.4f} r {r_i:+.4f}")
            is_null = lc.null_probe and i == 0
            if lc.estimator != "es":
                if is_null:
                    null_state = self.checkpoint_state()
                elif r_i > best_r:
                    best_state, best_i, best_r = self.checkpoint_state(), i, r_i
        self.load_state(snapshot)
        self.N, self.episode = saved_N, saved_episode
        self.F_cum += F_probes                                     # the probes' compute is charged (08 §8.6)
        self.F_probes += F_probes
        r_null = rewards[0] if lc.null_probe else None
        cand_rewards = rewards[1:] if lc.null_probe else rewards
        rec = {"episode": self.episode, "probe_rewards": rewards, "probe_fitness": fits, "null_reward": r_null,
               "F_probes": F_probes, "adopted_probe": None}
        kept = "none"
        if lc.estimator == "es":
            half = len(cands) // 2
            dg = es_proposal(torch.stack(cands[:half]) / (self.tau / math.sqrt(cfg.d_g)),
                             torch.tensor(cand_rewards[:half]), torch.tensor(cand_rewards[half:]),
                             sigma_es=self.tau / math.sqrt(cfg.d_g), tau=self.tau)
        else:
            i_star = self.restem.select(cand_rewards)
            r_star = cand_rewards[i_star]
            if r_null is not None and r_star <= r_null:
                dg = torch.zeros(cfg.d_g, device=self.device)
                kept = "null"
                if null_state is not None:
                    self._adopt_probe_state(null_state, snapshot)
                    rec["adopted_probe"] = 0
            else:
                dg = cands[i_star]
                kept = f"candidate {i_star}"
                if lc.estimator == "grpo":
                    rec["grpo"] = self.restem.update_grpo(self.cons, r_in, self.g, cands, cand_rewards, reverse=lc.reverse)
                else:
                    rec["restem"] = self.restem.update(self.cons, r_in, self.g, dg, reverse=lc.reverse)
                # continue from the winner's post-probe state: its steps were lived (06 §6.6)
                if best_state is not None and best_i == i_star + (1 if lc.null_probe else 0):
                    self._adopt_probe_state(best_state, snapshot)
                    rec["adopted_probe"] = best_i
        rec["kept"] = kept
        self.record["probes"].append(rec)
        return dg, kept

    def _adopt_probe_state(self, probe_state: dict, snapshot: dict) -> None:
        """Take the kept probe's weights, fast states, optimiser state and cursors; keep the
        boundary-level state of the checkpoint (genome, summaries, record, the consolidator
        as just updated, the compute accounting). The probe's steps are not in the record's
        step list (G-STAT reads the last `N` steps of an episode) but their compute is charged."""
        keep = {"record": self.record, "cons": copy.deepcopy(self.cons.state_dict()),
                "cons_opt": copy.deepcopy(self.cons_opt.state_dict()), "restem": self.restem.state_dict(),
                "reinforce": self.reinforce.state_dict(),
                "F": (self.F_cum, self.F_episode, self.F_probes)}
        self.load_state(probe_state)
        # the probe's record is now live; its births are real (their weights are being adopted)
        # and must survive the restore of the boundary-level record (11 §P-D.10)
        adopted_births = list(self.record.get("births", []))[len(keep["record"].get("births", [])):]
        self.record = keep["record"]
        for b in adopted_births:
            self.record["births"].append({**b, "adopted_from_probe": True})
        self.cons.load_state_dict(keep["cons"])
        self.cons_opt.load_state_dict(keep["cons_opt"])
        self.restem.load_state_dict(keep["restem"], device=self.device)
        self.reinforce.load_state_dict(keep["reinforce"])
        self.N = snapshot["N"]
        self.episode = snapshot["episode"]
        self.g = snapshot["g"].clone()
        self.g_prev = snapshot["g_prev"].clone()
        self.summaries = None if snapshot["summaries"] is None else WindowSummaries.from_state_dict(snapshot["summaries"])
        self.prev_summaries = None if snapshot["prev_summaries"] is None else WindowSummaries.from_state_dict(snapshot["prev_summaries"])
        self.fit_pairs = collections.deque([(g.clone(), u.clone()) for g, u in snapshot["fit_pairs"]], maxlen=HISTORY)
        self.probe_set, self.probe_losses = list(snapshot["probe_set"]), dict(snapshot["probe_losses"])
        self.probe_t0 = dict(snapshot.get("probe_t0", {}))
        self.rehearsal.load_state_dict(snapshot["rehearsal"])
        self.head_acc = {k: list(v) for k, v in snapshot["head_acc"].items()}
        self.return_acc = {k: list(v) for k, v in snapshot["return_acc"].items()}
        self.u_start = None if snapshot["u_start"] is None else snapshot["u_start"].clone()
        self.F_cum, self.F_episode, self.F_probes = keep["F"]       # the probes are already charged
        self.gen.set_state(snapshot["rng"]["gen"].cpu().to(torch.uint8))

    # ================================================================== transfer (10 §10.3 R7/R9)
    def reset_genome(self) -> None:
        """`g <- g_0`, the gains regenerated from it, the consolidator's memory of pairs and
        the rehearsal buffer cleared; `psi`, `zeta` and the weights kept; the pattern frozen."""
        self.g = self.g0.clone()
        self.g_prev = self.g.clone()
        self._u_next = None
        self.gains.set_from(self.expr.express(self.g))
        self.fit_pairs.clear()
        self.restem.accepted = []
        self.reinforce.baseline = 0.0          # estimator 1's memory, cleared with estimator 2's:
        # `reset_genome` serves the domain shifts of K-TRANSFER and K-XFER, and an EMA of the old
        # domain's rewards mis-centres the new domain's first advantages (11 §P-C.39)
        self.rehearsal = Rehearsal(self.rehearsal.n_reh)
        self.summaries, self.prev_summaries, self.c_prev = None, None, None
        self.probe_set, self.probe_losses, self.probe_t0 = [], {}, {}
        self.L_check_prev, self.fitness_prev = None, None
        self.dg_prev, self.r_in_prev = None, None
        self.pattern_frozen = True

    # ================================================================== inference (08 §8.4)
    @torch.no_grad()
    def step_inference(self, lane: Lane, raw: torch.Tensor, modality: str, t_w: float, *, act: Optional[str] = None,
                       generator: Optional[torch.Generator] = None, greedy: bool = False):
        """One span with `eta_1 = 0`: perceive, optionally emit; the emission is what the
        caller feeds back (the closure of 01 (1.4))."""
        y = self.perceive(lane, raw, modality, t_w)
        if act is None:
            return y, None
        if act == "text":
            return y, self.decode_text(y, generator or self.gen, greedy=greedy)
        return y, self.decode_head(y, act, generator or self.gen, greedy=greedy)

    # ================================================================== checkpoint (08 §8.5)
    @staticmethod
    def _to_host(obj):
        """A deep copy of a (possibly nested) state dict with every tensor on the CPU.

        11 §P-D.22: the probe checkpoints used to be CUDA-resident deep copies, three of them
        live at once inside `_best_of_G`, ~2.1 GB beside the model and its activations. That is
        what stalled a probe for 3h22m at 12.87 GB of a 16.38 GB card. An fp32 host round trip
        is exact, so moving them is bit-neutral."""
        import torch as _t
        if _t.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: Life._to_host(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = [Life._to_host(v) for v in obj]
            return type(obj)(out) if not isinstance(obj, tuple) else tuple(out)
        return copy.deepcopy(obj)

    def checkpoint_state(self) -> dict:
        task = self.streams.get("task")
        return {
            # 11 §P-D.22: on the host, not the card
            "agent": self._to_host(self.agent.state_dict()),
            "expr": self._to_host(self.expr.state_dict()),
            "gains": self._to_host(self.gains.state_dict()),
            "g": self.g.clone(), "zeta": self.zeta.clone(), "g0": self.g0.clone(),
            "cons": self._to_host(self.cons.state_dict()),
            "cons_opt": self._to_host(self.cons_opt.state_dict()),
            "reinforce": self.reinforce.state_dict(), "restem": self.restem.state_dict(),
            "pattern": self.pattern.to_dict(), "pattern_frozen": self.pattern_frozen, "born": dict(self.born),
            "episode": self.episode, "N": self.N,
            "in_episode": self._in_episode, "resume_step": self._resume_step,
            "resume_info": copy.deepcopy(getattr(self, "_resume_info", {})),
            "lanes": [(ln.index, ln.stream, copy.deepcopy(ln.cursor),
                       None if ln.S is None else {l: s.detach().to("cpu", copy=True) for l, s in ln.S.items()},
                       ln.boundary_pending)
                      for ln in self.lanes],
            "opt": None if self.opt is None else [self._to_host(o.state_dict()) for o in self.opt],
            "rehearsal": self.rehearsal.state_dict(),
            "fit_pairs": [(g.clone(), u.clone()) for g, u in self.fit_pairs],
            "probe_set": list(self.probe_set), "probe_losses": dict(self.probe_losses),
            "probe_t0": dict(self.probe_t0),
            "summaries": None if self.summaries is None else self.summaries.state_dict(),
            "prev_summaries": None if self.prev_summaries is None else self.prev_summaries.state_dict(),
            "c_prev": None if self.c_prev is None else self.c_prev.clone(),
            "R_scale": dict(self.R_scale), "scales": dict(self.scales), "insulation": self.agent.insulation.state_dict(),
            "L_check_prev": self.L_check_prev, "fitness_prev": None if self.fitness_prev is None else dict(self.fitness_prev),
            "mu_episode": dict(self.mu_episode),
            "n_doublings": self.n_doublings, "n_nonfinite": self.n_nonfinite,
            "dg_prev": None if self.dg_prev is None else self.dg_prev.clone(),
            "r_in_prev": None if self.r_in_prev is None else self.r_in_prev.clone(),
            "g_prev": self.g_prev.clone(),
            "u_start": None if self.u_start is None else self.u_start.clone(),
            "u_next": None if self._u_next is None else self._u_next.clone(),
            "head_acc": {k: list(v) for k, v in self.head_acc.items()},
            "return_acc": {k: list(v) for k, v in self.return_acc.items()},
            "F_cum": self.F_cum, "F_episode": self.F_episode, "F_probes": self.F_probes,
            "budget_exhausted": self.budget_exhausted, "curriculum_off": self._curriculum_off,
            "rng": {"torch": torch.get_rng_state(), "gen": self.gen.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(), "python": random.getstate()},
            "record": copy.deepcopy(self.record), "step_times": list(self.step_times),
            "task_archive": None if task is None or not hasattr(task, "tasks") else copy.deepcopy(task.tasks),
            "provenance": {"config": self.cfg.as_dict(), "seed": self.seed, "order_seed": self.order_seed,
                           "torch": torch.__version__, "code_digest": code_digest()},
        }

    def load_state(self, s: dict) -> None:
        self.agent.load_state_dict(s["agent"])
        self.expr.load_state_dict(s["expr"])
        self.gains.load_state_dict(s["gains"])
        self.g, self.zeta = s["g"].clone(), s["zeta"].clone()
        if "g0" in s:
            self.g0 = s["g0"].clone()
        self.cons.load_state_dict(s["cons"])
        self.cons_opt.load_state_dict(s["cons_opt"])
        self.reinforce.load_state_dict(s["reinforce"])
        self.restem.load_state_dict(s["restem"], device=self.device)
        self.pattern = SwitchPattern.from_dict(self.cfg, s["pattern"])
        self.pattern_frozen = bool(s.get("pattern_frozen", False))
        self.born = dict(s["born"])
        self.episode, self.N = s["episode"], s["N"]
        self._in_episode = bool(s.get("in_episode", False))
        self._resume_step = s.get("resume_step")
        self._resume_info = dict(s.get("resume_info", {}))
        self.lanes = [Lane(i, st, copy.deepcopy(cur),
                           None if S is None else {l: t.to(self.device) for l, t in S.items()}, None, bp)
                      for i, st, cur, S, bp in s["lanes"]]
        if s["opt"] is not None:
            gamma = self.gains.gamma().detach()
            self.opt = self._optimizers(lr_multipliers(self.reg, gamma) if self.lc.genome else [1.0] * 4)
            for o, sd in zip(self.opt, s["opt"]):
                o.load_state_dict(sd)
        self.rehearsal.load_state_dict(s["rehearsal"])
        self.fit_pairs = collections.deque([(g.clone(), u.clone()) for g, u in s.get("fit_pairs", [])], maxlen=HISTORY)
        self.probe_set, self.probe_losses = list(s["probe_set"]), dict(s["probe_losses"])
        self.probe_t0 = dict(s.get("probe_t0", {}))
        self.summaries = None if s["summaries"] is None else WindowSummaries.from_state_dict(s["summaries"])
        self.prev_summaries = None if s["prev_summaries"] is None else WindowSummaries.from_state_dict(s["prev_summaries"])
        self.c_prev = None if s.get("c_prev") is None else s["c_prev"].cpu().clone()   # cached; moved when used
        self.R_scale = dict(s["R_scale"])
        self.scales = dict(s.get("scales", {}))
        self.agent.insulation.load_state_dict(s["insulation"])
        self.L_check_prev = s["L_check_prev"]
        self.fitness_prev = None if s.get("fitness_prev") is None else dict(s["fitness_prev"])
        self.mu_episode = dict(s.get("mu_episode", {}))
        self.n_doublings = int(s.get("n_doublings", 0))
        self.n_nonfinite = int(s.get("n_nonfinite", 0))
        self.dg_prev = None if s["dg_prev"] is None else s["dg_prev"].clone()
        self.r_in_prev = None if s["r_in_prev"] is None else s["r_in_prev"].clone()
        self.g_prev = s["g_prev"].clone()
        self.u_start = None if s.get("u_start") is None else s["u_start"].clone()
        self._u_next = None if s.get("u_next") is None else s["u_next"].clone()
        self.head_acc = {k: list(v) for k, v in s.get("head_acc", {}).items()}
        self.return_acc = {k: list(v) for k, v in s.get("return_acc", {}).items()}
        self.F_cum, self.F_episode, self.F_probes = s.get("F_cum", 0.0), s.get("F_episode", 0.0), s.get("F_probes", 0.0)
        self.budget_exhausted = bool(s.get("budget_exhausted", False))
        self._curriculum_off = s.get("curriculum_off")
        # the generator states are CPU byte tensors; `torch.load(map_location=cuda)` moves every
        # tensor in the blob to the device, so they are brought back explicitly (08 §8.5)
        torch.set_rng_state(s["rng"]["torch"].cpu().to(torch.uint8))
        self.gen.set_state(s["rng"]["gen"].cpu().to(torch.uint8))
        if s["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in s["rng"]["cuda"]])
        np.random.set_state(s["rng"]["numpy"])
        random.setstate(s["rng"]["python"])
        self.record = copy.deepcopy(s["record"])
        self.step_times = list(s["step_times"])
        if s.get("task_archive") is not None and "task" in self.streams and hasattr(self.streams["task"], "tasks"):
            self.streams["task"].tasks = copy.deepcopy(s["task_archive"])
        if self.lc.svf:
            # the SVD cache is a derived quantity, not state: it is rebuilt here because a life
            # resumed *inside* an episode skips the boundary that would otherwise fill it, and
            # the first forward would then raise (11 §P-C.36)
            self.agent.backbone.refresh_svf()

    def save(self, path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), p)

    def load(self, path, *, allow_digest_mismatch: bool = False) -> None:
        s = torch.load(path, map_location=self.device, weights_only=False)
        if s["provenance"]["code_digest"] != code_digest() and not allow_digest_mismatch:
            raise RuntimeError("checkpoint code digest differs from the running code (L9)")
        self.load_state(s)


def code_digest() -> str:
    """A hash of `tempo/*.py` (08 §8.5 provenance)."""
    h = hashlib.blake2b(digest_size=8)
    root = pathlib.Path(__file__).resolve().parent
    for f in sorted(root.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()
