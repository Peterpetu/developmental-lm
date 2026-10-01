"""The phenotype: free weights per role, growth-aware.

SPEC: 01 §1.1
SPEC: 04 §4.1
SPEC: 04 §4.3
SPEC: 04 §4.4
SPEC: 04 §4.7
SPEC: 06 §6.3

`Phenotype` owns one `nn.Parameter` per `(role, layer)` and nothing else (P13).
Its forward pass is the block of §04.3 with the two mixers of §04.4, plus the
growth gates of §06.3 (residual channel gate, per-layer inner and head gates,
per-layer branch gate), which are buffers that are identically one outside a
ramp and removed when the ramp completes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnomon.utils.shapes import check_shape
from . import ops
from .config import (
    GAIN_ROLES, MATRIX_ROLES, N_TAPS, ROLE_AB, ROLE_AGATE, ROLE_CONV_K, ROLE_CONV_Q,
    ROLE_CONV_V, ROLE_DOWN, ROLE_EMB, ROLE_GATE, ROLE_HEAD_OUT, ROLE_K,
    ROLE_LOGIT_SCALE, ROLE_NAMES, ROLE_NORM_FFN, ROLE_NORM_FINAL, ROLE_NORM_MIX,
    ROLE_O, ROLE_OGATE, ROLE_Q, ROLE_UP, ROLE_V, ROLE_VE, CONV_ROLES, StageConfig,
)

__all__ = ["Phenotype", "key_name", "free_init", "save_phenotype", "load_phenotype"]


def key_name(role: int, l: int) -> str:
    """Parameter name of `(role, layer)`; `g` for the global pseudo-layer."""
    return f"{ROLE_NAMES[role]}_{'g' if l == -1 else l}"


def free_init(role: int, shape: tuple[int, ...], cfg: StageConfig,
              generator: torch.Generator) -> torch.Tensor:
    """SPEC: 04 §4.7 -- the free-weight initialisation of one role."""
    if role in CONV_ROLES:
        w = torch.zeros(*shape)
        w[:, 0] = 1.0
        return w
    if role == ROLE_EMB or role == ROLE_HEAD_OUT:
        return torch.randn(*shape, generator=generator) * cfg.d_model ** -0.5
    if role == ROLE_VE:
        return torch.randn(*shape, generator=generator) * shape[1] ** -0.5
    if role in MATRIX_ROLES:
        return torch.randn(*shape, generator=generator) * shape[1] ** -0.5
    if role in (ROLE_NORM_MIX, ROLE_NORM_FFN):
        return torch.full(shape, (2.0 * cfg.L) ** -0.5)
    if role == ROLE_NORM_FINAL:
        return torch.ones(*shape)
    if role == ROLE_LOGIT_SCALE:
        return torch.zeros(*shape)
    raise KeyError(role)


class Phenotype(nn.Module):
    """`W` at one stage. SPEC: 04."""

    def __init__(self, cfg: StageConfig, vocab: int, *, tied: bool = True,
                 seed: int = 0, weights: Optional[dict] = None):
        super().__init__()
        self.cfg, self.vocab, self.tied = cfg, vocab, tied
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.W = nn.ParameterDict()
        for role, l in cfg.keys(vocab, tied=tied):
            shp = cfg.role_shape(role, l, vocab, tied=tied)
            if weights is not None:
                t = weights[(role, l)].detach().clone().float()
                check_shape(t, list(shp), name=key_name(role, l))
            else:
                t = free_init(role, shp, cfg, g)
            self.W[key_name(role, l)] = nn.Parameter(t)
        # SPEC: 06 §6.3 -- growth gates; None means "identically one, removed".
        self.res_gate: Optional[torch.Tensor] = None
        self.inner_gate: dict[int, torch.Tensor] = {}
        self.head_gate: dict[int, torch.Tensor] = {}
        self.layer_gate: dict[int, float] = {}
        self.use_reference = False

    # ---------------------------------------------------------------- access
    def get(self, role: int, l: int) -> torch.Tensor:
        return self.W[key_name(role, l)]

    def has(self, role: int, l: int) -> bool:
        return key_name(role, l) in self.W

    def weights(self) -> dict[tuple[int, int], torch.Tensor]:
        return {(r, l): self.get(r, l) for r, l in self.cfg.keys(self.vocab, tied=self.tied)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_params_nonemb(self) -> int:
        return self.n_params() - self.get(ROLE_EMB, -1).numel()

    def head(self) -> torch.Tensor:
        return self.get(ROLE_EMB, -1) if self.tied else self.get(ROLE_HEAD_OUT, -1)

    # --------------------------------------------------------------- forward
    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
        """`tokens (B, T)` -> logits `(B, T, V)`, or the final normalised hidden `(B, T, d)`."""
        cfg = self.cfg
        check_shape(tokens, [None, None], name="tokens")
        E = self.get(ROLE_EMB, -1)
        x = F.embedding(tokens, E) * cfg.emb_scale                              # (4.1)
        rg = self.res_gate
        for l in range(cfg.L):
            op = cfg.op(l)
            h = ops.rms_norm_gf(x, rg)
            hm = self.head_gate.get(l)
            if op == "attn":
                ve = None
                if cfg.use_ve and self.has(ROLE_VE, l):
                    ve = F.embedding(tokens, self.get(ROLE_VE, l))
                y = ops.attention_gated(
                    h, self.get(ROLE_Q, l), self.get(ROLE_K, l), self.get(ROLE_V, l),
                    self.get(ROLE_O, l), self.get(ROLE_AGATE, l),
                    n_h=cfg.n_h, n_kv=cfg.n_kv(l), d_head=cfg.d_head, ve=ve, head_mask=hm)
            else:
                y, _ = ops.delta_mixer(
                    h, self.get(ROLE_Q, l), self.get(ROLE_K, l), self.get(ROLE_V, l),
                    self.get(ROLE_O, l), self.get(ROLE_AB, l), self.get(ROLE_OGATE, l),
                    self.get(ROLE_CONV_Q, l), self.get(ROLE_CONV_K, l), self.get(ROLE_CONV_V, l),
                    n_h=cfg.n_h, d_head=cfg.d_head, chunk_size=cfg.chunk_size,
                    head_mask=hm, use_reference=self.use_reference)
            lg = self.layer_gate.get(l, 1.0)
            x = x + lg * self.get(ROLE_NORM_MIX, l) * ops.rms_norm_gf(y, rg)
            h = ops.rms_norm_gf(x, rg)
            f = ops.swiglu(h, self.get(ROLE_GATE, l), self.get(ROLE_UP, l),
                           self.get(ROLE_DOWN, l), inner_mask=self.inner_gate.get(l))
            x = x + lg * self.get(ROLE_NORM_FFN, l) * ops.rms_norm_gf(f, rg)
        xn = ops.rms_norm_gf(x, rg) * self.get(ROLE_NORM_FINAL, -1)
        if return_hidden:
            return xn
        return ops.softcap_logits(F.linear(xn, self.head()), self.get(ROLE_LOGIT_SCALE, -1))

    def loss(self, window: torch.Tensor, *, chunk: int = 4096
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(ce, zloss)` over a `(B, T+1)` window. SPEC: 04 §4.3 (4.5)."""
        xn = self.forward(window[:, :-1], return_hidden=True)
        d = xn.shape[-1]
        return ops.chunked_cross_entropy(
            xn.reshape(-1, d), self.head(), self.get(ROLE_LOGIT_SCALE, -1),
            window[:, 1:].reshape(-1), chunk=chunk)

    # ------------------------------------------------------------- untying
    @torch.no_grad()
    def untie(self) -> None:
        """SPEC: 03 §3.5, 04 §4.3 -- copy the embedding into a free head."""
        if not self.tied:
            return
        self.W[key_name(ROLE_HEAD_OUT, -1)] = nn.Parameter(self.get(ROLE_EMB, -1).detach().clone())
        self.tied = False

    # ------------------------------------------------------------- optimiser
    def optimizer_groups(self) -> tuple[list, list]:
        """SPEC: 04 §4.6 -- group M (Muon) and group A (Adam), by role name."""
        muon, adam = [], []
        muon_roles = {ROLE_Q, ROLE_K, ROLE_V, ROLE_O, ROLE_AB, ROLE_OGATE, ROLE_AGATE,
                      ROLE_GATE, ROLE_UP, ROLE_DOWN}
        for role, l in self.cfg.keys(self.vocab, tied=self.tied):
            p = self.get(role, l)
            (muon if role in muon_roles else adam).append(p)
        return muon, adam


AB_LAYOUT = "interleaved"
"""SPEC: 04 §4.4 -- the gate-row layout a checkpoint was trained under. Checkpoints
written before the interleaving (decay block, then write block) carry no marker
and are converted on load."""


def save_phenotype(model: Phenotype, path) -> None:
    """Weights + stage config + vocab + gate layout, so a phenotype reloads without its script."""
    import dataclasses
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": dataclasses.asdict(model.cfg), "vocab": model.vocab,
                "tied": model.tied, "ab_layout": AB_LAYOUT,
                "state": {k: v.detach().cpu() for k, v in model.W.items()}},
               p)


def _interleave_ab(t: torch.Tensor) -> torch.Tensor:
    """Block layout `[a_0..a_{h-1}, b_0..b_{h-1}]` -> interleaved `[a_0, b_0, a_1, b_1, ...]`."""
    h = t.shape[0] // 2
    idx = torch.empty(2 * h, dtype=torch.long)
    idx[0::2] = torch.arange(h)
    idx[1::2] = torch.arange(h) + h
    return t[idx]


def load_phenotype(path, device="cpu") -> Phenotype:
    blob = torch.load(path, map_location="cpu")
    cfg_d = dict(blob["cfg"])
    if "ops_override" in cfg_d and cfg_d["ops_override"] is not None:
        cfg_d["ops_override"] = tuple(cfg_d["ops_override"])
    cfg = StageConfig(**cfg_d)
    m = Phenotype(cfg, blob["vocab"], tied=blob["tied"])
    convert = blob.get("ab_layout") != AB_LAYOUT
    with torch.no_grad():
        for k, v in blob["state"].items():
            if convert and k.startswith("AB_"):
                v = _interleave_ab(v)
            m.W[k].copy_(v)
    return m.to(device)
