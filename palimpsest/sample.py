"""Incremental (cached) sampling from a phenotype: one token per step, the delta layers by
their exact recurrence with a carried state and a three-token convolution history, the
attention layers by a key/value cache. SPEC: docs/palimpsest/01-model.md §1.5 (Definition 6),
§1.10. The identity with the full forward is test T4b.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from anlage import ops
from anlage.config import (ROLE_AB, ROLE_AGATE, ROLE_CONV_K, ROLE_CONV_Q, ROLE_CONV_V, ROLE_DOWN, ROLE_EMB,
                           ROLE_GATE, ROLE_K, ROLE_LOGIT_SCALE, ROLE_NORM_FFN, ROLE_NORM_FINAL, ROLE_NORM_MIX,
                           ROLE_O, ROLE_OGATE, ROLE_Q, ROLE_UP, ROLE_V, ROLE_VE)
from anlage.model import Phenotype
from gnomon.pheno.mixer_chunk import BETA_MAX
from gnomon.pheno.mixer_ref import gated_delta_rule_reference
from gnomon.pheno.rmsnorm import rms_norm_headwise

__all__ = ["Sampler", "rope_at"]


def rope_at(x: torch.Tensor, pos: int, theta: float = 1e4) -> torch.Tensor:
    """`anlage/ops.py::rope` for a single position `pos`; `x` is `(B, H, 1, d)`."""
    d = x.shape[-1]
    half = d // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = float(pos) * freqs
    cos, sin = ang.cos()[None, None, None], ang.sin()[None, None, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class Sampler:
    """Step a phenotype one token at a time with the same arithmetic as its forward."""

    def __init__(self, model: Phenotype, batch: int, device: str):
        self.m, self.cfg, self.B, self.device = model, model.cfg, batch, device
        cfg = self.cfg
        self.S: dict[int, torch.Tensor] = {}
        self.hist: dict[int, list[torch.Tensor]] = {}          # last three raw (q, k, v) per delta layer
        self.kc: dict[int, torch.Tensor] = {}
        self.vc: dict[int, torch.Tensor] = {}
        self.t = 0
        for l in range(cfg.L):
            if cfg.op(l) == "delta":
                self.S[l] = torch.zeros(batch, cfg.n_h, cfg.d_head, cfg.d_head, device=device)
                self.hist[l] = []

    def _conv_step(self, x_raw: torch.Tensor, W: torch.Tensor, hist: list[torch.Tensor]) -> torch.Tensor:
        """`sum_tau W[c, tau] x[t - tau, c]` with `hist = [x_{t-1}, x_{t-2}, x_{t-3}]` (missing = 0)."""
        y = x_raw * W[:, 0].to(x_raw.dtype)
        for tau in (1, 2, 3):
            if len(hist) >= tau:
                y = y + hist[tau - 1] * W[:, tau].to(x_raw.dtype)
        return y

    @torch.no_grad()
    def step(self, tok: torch.Tensor) -> torch.Tensor:
        """`tok (B,)` at position `self.t` -> logits `(B, V)` for the next token."""
        m, cfg = self.m, self.cfg
        B = tok.shape[0]
        x = F.embedding(tok, m.get(ROLE_EMB, -1))[:, None, :] * cfg.emb_scale          # (B, 1, d)
        for l in range(cfg.L):
            h = ops.rms_norm_gf(x)
            if cfg.op(l) == "attn":
                n_h, n_kv, dk = cfg.n_h, cfg.n_kv(l), cfg.d_head
                q = F.linear(h, m.get(ROLE_Q, l)).view(B, 1, n_h, dk).transpose(1, 2)
                k = F.linear(h, m.get(ROLE_K, l)).view(B, 1, n_kv, dk).transpose(1, 2)
                v = F.linear(h, m.get(ROLE_V, l))
                if cfg.use_ve and m.has(ROLE_VE, l):
                    v = v + F.embedding(tok, m.get(ROLE_VE, l))[:, None, :]
                v = v.view(B, 1, n_kv, dk).transpose(1, 2)
                q, k = rms_norm_headwise(q), rms_norm_headwise(k)
                q, k = rope_at(q, self.t), rope_at(k, self.t)
                self.kc[l] = k if l not in self.kc else torch.cat([self.kc[l], k], 2)
                self.vc[l] = v if l not in self.vc else torch.cat([self.vc[l], v], 2)
                kk, vv = self.kc[l], self.vc[l]
                if n_kv != n_h:
                    rep = n_h // n_kv
                    kk, vv = kk.repeat(1, rep, 1, 1), vv.repeat(1, rep, 1, 1)
                a = F.scaled_dot_product_attention(q, kk, vv, is_causal=False, scale=1.0 / math.sqrt(dk))
                a = a.transpose(1, 2).reshape(B, 1, n_h * dk)
                a = a * torch.sigmoid(F.linear(h, m.get(ROLE_AGATE, l)) + ops.G0_GATE)
                y = F.linear(a, m.get(ROLE_O, l))
            else:
                n_h, dk = cfg.n_h, cfg.d_head
                raw = [F.linear(h, m.get(r, l)) for r in (ROLE_Q, ROLE_K, ROLE_V)]
                hist = self.hist[l]
                convs = (m.get(ROLE_CONV_Q, l), m.get(ROLE_CONV_K, l), m.get(ROLE_CONV_V, l))
                outs = []
                for i in range(3):
                    hh = [hs[i] for hs in hist]
                    outs.append(F.silu(self._conv_step(raw[i], convs[i], hh)))
                q, k, v = [o.view(B, 1, n_h, dk).transpose(1, 2) for o in outs]
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                ab = F.linear(h, m.get(ROLE_AB, l)).transpose(1, 2)                      # (B, 2h, 1)
                alpha = torch.sigmoid(ab[:, 0::2] + ops.A0_DECAY)
                beta = (2.0 * torch.sigmoid(ab[:, 1::2])).clamp(max=BETA_MAX)
                o, self.S[l] = gated_delta_rule_reference(q.float(), k.float(), v.float(), alpha.float(),
                                                          beta.float(), initial_state=self.S[l])
                o = rms_norm_headwise(o.to(h.dtype)).transpose(1, 2).reshape(B, 1, n_h * dk)
                o = o * F.silu(F.linear(h, m.get(ROLE_OGATE, l)))
                y = F.linear(o, m.get(ROLE_O, l))
                hist.insert(0, raw)
                del hist[3:]
            x = x + m.get(ROLE_NORM_MIX, l) * ops.rms_norm_gf(y)
            h = ops.rms_norm_gf(x)
            f = ops.swiglu(h, m.get(ROLE_GATE, l), m.get(ROLE_UP, l), m.get(ROLE_DOWN, l))
            x = x + m.get(ROLE_NORM_FFN, l) * ops.rms_norm_gf(f)
        xn = ops.rms_norm_gf(x) * m.get(ROLE_NORM_FINAL, -1)
        self.t += 1
        return ops.softcap_logits(F.linear(xn, m.head()), m.get(ROLE_LOGIT_SCALE, -1))[:, 0, :]
