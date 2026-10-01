"""The backbone `F`: one span step over the active blocks, with gain folds and switches.

SPEC: 03 §3.2  (the residual block, Peri-LN, (3.1)-(3.3))
SPEC: 03 §3.3-3.4  (the delta operator, (3.4)-(3.8); the fast state persists)
SPEC: 03 §3.5  (the attention operator with GQA, QK-norm, RoPE, the KV cache, (3.9))
SPEC: 03 §3.6  (the FFN, (3.10))
SPEC: 03 §3.7  (the output, (3.11))
SPEC: 03 §3.8  (the gain fold (3.12) in-graph; active views as prefixes)
SPEC: 08 §8.1  (initialisation)

`Backbone` owns one parameter per `(role, layer)` at the provisioned shape and
nothing else. Its forward takes the perception stream `x (B, T, d_x)`, the
lanes' fast states, an optional KV cache, the switch pattern (05), the gains
(04) and `zeta` (07), and returns `y_hat`, the new states, the new cache and
the final normalised hidden. At `gains = None`, `zeta = None` and the adult
pattern it is exactly ANLAGE's `Phenotype` block (P6).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anlage.ops import rms_norm_gf, causal_conv4
from gnomon.pheno.mixer_chunk import BETA_MAX, gated_delta_rule_chunk
from gnomon.pheno.mixer_ref import gated_delta_rule_reference
from gnomon.pheno.rmsnorm import rms_norm_headwise
from gnomon.utils.shapes import check_shape
from .config import TempoConfig
from .morphology import SwitchPattern

__all__ = ["Backbone", "A0_DECAY", "G0_WRITE", "GOG_ATTN", "GOG_DELTA", "MATRIX_ROLES",
           "DELTA_ROLES", "EVERY_ROLES", "ATTN_ROLES", "VECTOR_ROLES"]

A0_DECAY = 4.0
"""SPEC: 03 (3.5) -- the decay offset's weight value: `alpha = sigmoid(a + 4) ~ 0.98`."""
G0_WRITE = 0.0
"""SPEC: 03 (3.5) -- the write offset's weight value: `beta = 2 sigmoid(b + 0) = 1`."""
GOG_ATTN = 2.0
"""SPEC: 03 (3.9) -- the attention output gate's offset."""
GOG_DELTA = 0.0
"""SPEC: 03 (3.8) -- the delta output gate (SiLU) has no offset."""

EVERY_ROLES = ("Q", "K", "V", "O", "OG", "GATE", "UP", "DOWN", "NMIX", "NFFN", "RMIX", "RFFN", "GOG")
DELTA_ROLES = ("AB", "CQ", "CK", "CV", "A0", "G0")
ATTN_ROLES = ()
MATRIX_ROLES = ("Q", "K", "V", "O", "OG", "AB", "GATE", "UP", "DOWN")
"""The 2-D roles that carry matrix gains (04 §4.2); taps, norms and offsets do not."""
VECTOR_ROLES = ("NMIX", "NFFN", "RMIX", "RFFN")


def _rope_at(x: torch.Tensor, offset: int, theta: float = 1e4) -> torch.Tensor:
    """RoPE over `(B, H, T, d)` with positions `offset .. offset+T-1`; equals
    `anlage/ops.rope` at `offset = 0` in fp32 and keeps float64 inputs in float64
    (the ANLAGE form downcasts, which a float64 finite-difference check detects)."""
    B, H, T, d = x.shape
    if d % 2:
        raise ValueError("RoPE needs an even head dimension")
    acc = torch.promote_types(x.dtype, torch.float32)
    half = d // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device, dtype=acc) / half)
    pos = torch.arange(offset, offset + T, device=x.device, dtype=acc)
    ang = pos[:, None] * freqs[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half].to(acc), x[..., half:].to(acc)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class Backbone(nn.Module):
    """`F` of 01 (1.2) at the provisioned shape. SPEC: 03."""

    def __init__(self, cfg: TempoConfig, *, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        if cfg.primordium_heads % cfg.h_kv:
            raise ValueError("the primordium's heads must be a multiple of h_kv")
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.W = nn.ParameterDict()
        d, dh, h, hkv, Fw = cfg.d, cfg.d_head, cfg.h, cfg.h_kv, cfg.F

        def mat(shape):
            return torch.randn(*shape, generator=g) * shape[1] ** -0.5      # N(0, 1/fan_in)

        for l in range(cfg.L):
            delta = cfg.op(l) == "delta"
            n_kv = h if delta else hkv
            self.W[f"L{l}_Q"] = nn.Parameter(mat((h * dh, d)))
            self.W[f"L{l}_K"] = nn.Parameter(mat((n_kv * dh, d)))
            self.W[f"L{l}_V"] = nn.Parameter(mat((n_kv * dh, d)))
            self.W[f"L{l}_O"] = nn.Parameter(mat((d, h * dh)))
            self.W[f"L{l}_OG"] = nn.Parameter(mat((h * dh, d)))
            self.W[f"L{l}_GATE"] = nn.Parameter(mat((Fw, d)))
            self.W[f"L{l}_UP"] = nn.Parameter(mat((Fw, d)))
            self.W[f"L{l}_DOWN"] = nn.Parameter(mat((d, Fw)))
            self.W[f"L{l}_NMIX"] = nn.Parameter(torch.full((d,), (2.0 * cfg.L) ** -0.5))
            self.W[f"L{l}_NFFN"] = nn.Parameter(torch.full((d,), (2.0 * cfg.L) ** -0.5))
            self.W[f"L{l}_RMIX"] = nn.Parameter(torch.ones(d))
            self.W[f"L{l}_RFFN"] = nn.Parameter(torch.ones(d))
            self.W[f"L{l}_GOG"] = nn.Parameter(torch.full((1,), GOG_DELTA if delta else GOG_ATTN))
            if delta:
                self.W[f"L{l}_AB"] = nn.Parameter(mat((2 * h, d)))
                for role in ("CQ", "CK", "CV"):
                    taps = torch.zeros(h * dh, 4)
                    taps[:, 0] = 1.0
                    self.W[f"L{l}_{role}"] = nn.Parameter(taps)
                self.W[f"L{l}_A0"] = nn.Parameter(torch.full((h,), A0_DECAY))
                self.W[f"L{l}_G0"] = nn.Parameter(torch.full((h,), G0_WRITE))
        self.W["IN_W"] = nn.Parameter(mat((d, cfg.d_x)))
        self.W["IN_b"] = nn.Parameter(torch.zeros(d))
        self.W["OUT_W"] = nn.Parameter(mat((cfg.d_x, d)))
        self.W["OUT_rho"] = nn.Parameter(torch.ones(d))
        self.use_reference = False
        self._views_key = None
        self.svf: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        """04 §4.8: when non-empty, `W_eff = W + U diag(s a) V^T` with the SVD cached at the
        episode boundary by `refresh_svf`; the gain `a` has `min(n_in, n_out)` entries."""

    # ------------------------------------------------------------------ names
    def layer_param_names(self, l: int) -> tuple[str, ...]:
        roles = EVERY_ROLES + (DELTA_ROLES if self.cfg.op(l) == "delta" else ATTN_ROLES)
        return tuple(f"L{l}_{r}" for r in roles)

    def matrix_names(self) -> tuple[str, ...]:
        """The 2-D parameters that carry matrix gains, in registry order."""
        names = []
        for l in range(self.cfg.L):
            for r in MATRIX_ROLES:
                if r == "AB" and self.cfg.op(l) != "delta":
                    continue
                names.append(f"L{l}_{r}")
        names += ["IN_W", "OUT_W"]
        return tuple(names)

    def optimizer_groups(self) -> tuple[list, list]:
        """Muon on the 2-D matrices (taps excluded, as ANLAGE), AdamW on the rest."""
        muon, adam = [], []
        mats = set(self.matrix_names())
        for name, p in self.W.items():
            (muon if name in mats else adam).append(p)
        return muon, adam

    def n_params(self) -> int:
        return sum(p.numel() for p in self.W.values())

    # ------------------------------------------------------------------ state
    def init_state(self, B: int, device=None, dtype=torch.float32) -> dict[int, torch.Tensor]:
        """Zero fast states `(B, h, d_head, d_head)` for every delta layer (A3)."""
        cfg = self.cfg
        device = device or self.W["IN_W"].device
        return {l: torch.zeros(B, cfg.h, cfg.d_head, cfg.d_head, device=device, dtype=dtype)
                for l in range(cfg.L) if cfg.op(l) == "delta"}

    def reset_state_layer(self, l: int) -> None:
        """Hook for births: states are owned by the life; nothing to do at the parameter level."""
        return None

    def reset_state_heads(self, l: int, from_head: int) -> None:
        return None

    # ------------------------------------------------------------------ gains
    @torch.no_grad()
    def refresh_svf(self, names: Optional[tuple[str, ...]] = None) -> None:
        """Cache `U, s, V^T` of every matrix role at the episode boundary (04 §4.8, E12)."""
        for name in (names or self.matrix_names()):
            U, s, Vh = torch.linalg.svd(self.W[name].detach().float(), full_matrices=False)
            self.svf[name] = (U, s, Vh)

    def _eff(self, name: str, gains: Optional[dict]) -> torch.Tensor:
        """(3.12): `diag(1 + a_out) W diag(1 + a_in)` in-graph, or `W` itself; under the SVF
        ablation `W + U diag(s a) V^T` with the cached decomposition."""
        W = self.W[name]
        if gains is None:
            return W
        a_svf = gains.get(name + ":svf")
        if a_svf is not None:
            if name not in self.svf:
                raise RuntimeError(f"SVF gains for {name} but no cached decomposition: call refresh_svf at the boundary")
            U, s, Vh = self.svf[name]
            return W + ((U * (s * a_svf.float())[None, :]) @ Vh).to(W.dtype)
        if self.svf and (name + ":in") in gains:
            raise RuntimeError("diagonal gains supplied to a backbone in SVF mode")
        a_in, a_out = gains.get(name + ":in"), gains.get(name + ":out")
        if a_in is None and a_out is None:
            return W
        if a_out is not None:
            W = (1.0 + a_out.to(W.dtype))[:, None] * W
        if a_in is not None:
            W = W * (1.0 + a_in.to(W.dtype))[None, :]
        return W

    def _vec(self, name: str, gains: Optional[dict]) -> torch.Tensor:
        v = self.W[name]
        if gains is None or name not in gains:
            return v
        return v * (1.0 + gains[name].to(v.dtype))

    def _off(self, name: str, gains: Optional[dict]) -> torch.Tensor:
        v = self.W[name]
        if gains is None or name not in gains:
            return v
        return v + gains[name].to(v.dtype)

    # ------------------------------------------------------------------ operators
    def _delta(self, l: int, hn: torch.Tensor, S: Optional[torch.Tensor], pattern: SwitchPattern,
               gains: Optional[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B, T, _ = hn.shape
        h_act, dh = pattern.h_act(l), cfg.d_head
        n = h_act * dh
        Wq, Wk, Wv = (self._eff(f"L{l}_{r}", gains)[:n] for r in ("Q", "K", "V"))
        Wo = self._eff(f"L{l}_O", gains)[:, :n]
        Wog = self._eff(f"L{l}_OG", gains)[:n]
        Wab = self._eff(f"L{l}_AB", gains)[:2 * h_act]
        Cq, Ck, Cv = (self.W[f"L{l}_{r}"][:n] for r in ("CQ", "CK", "CV"))
        a0 = self._off(f"L{l}_A0", gains)[:h_act]
        g0 = self._off(f"L{l}_G0", gains)[:h_act]
        gog = self._off(f"L{l}_GOG", gains)
        q = F.silu(causal_conv4(F.linear(hn, Wq), Cq))
        k = F.silu(causal_conv4(F.linear(hn, Wk), Ck))
        v = F.silu(causal_conv4(F.linear(hn, Wv), Cv))
        q = q.view(B, T, h_act, dh).transpose(1, 2)
        k = k.view(B, T, h_act, dh).transpose(1, 2)
        v = v.view(B, T, h_act, dh).transpose(1, 2)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ab = F.linear(hn, Wab).transpose(1, 2)                                   # (B, 2h, T)
        alpha = torch.sigmoid(ab[:, 0::2] + a0.to(ab.dtype)[None, :, None])
        beta = (2.0 * torch.sigmoid(ab[:, 1::2] + g0.to(ab.dtype)[None, :, None])).clamp(max=BETA_MAX)
        acc = torch.promote_types(q.dtype, torch.float32)          # the state solves at >= fp32
        S_in = None if S is None else S[:, :h_act].to(acc)
        qf, kf, vf, af, bf = q.to(acc), k.to(acc), v.to(acc), alpha.to(acc), beta.to(acc)
        if self.use_reference:
            o, S_new = gated_delta_rule_reference(qf, kf, vf, af, bf, initial_state=S_in)
        else:
            o, S_new = gated_delta_rule_chunk(qf, kf, vf, af, bf, chunk_size=cfg.chunk_size,
                                              initial_state=S_in)
        o = rms_norm_headwise(o).to(hn.dtype).transpose(1, 2).reshape(B, T, n)
        o = o * F.silu(F.linear(hn, Wog) + gog.to(hn.dtype))
        o = o * pattern.head_mask(l, o.device, o.dtype)
        if S is not None:
            S_full = S.clone()
            S_full[:, :h_act] = S_new.to(S.dtype)
        else:
            S_full = torch.zeros(B, cfg.h, dh, dh, device=S_new.device, dtype=S_new.dtype)
            S_full[:, :h_act] = S_new
        return F.linear(o, Wo), S_full

    def _attn(self, l: int, hn: torch.Tensor, cache: Optional[tuple[torch.Tensor, torch.Tensor]],
              pattern: SwitchPattern, gains: Optional[dict], *, explicit: bool = False
              ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """`explicit=True` computes the softmax attention with plain tensor ops (the fused
        SDPA kernels support neither forward-mode AD nor double backward, which the
        criticality probe of 07 needs); P-explicit asserts the two paths agree."""
        cfg = self.cfg
        B, T, _ = hn.shape
        h_act, dh, hkv = pattern.h_act(l), cfg.d_head, cfg.h_kv
        n = h_act * dh
        Wq = self._eff(f"L{l}_Q", gains)[:n]
        Wk, Wv = self._eff(f"L{l}_K", gains), self._eff(f"L{l}_V", gains)
        Wo = self._eff(f"L{l}_O", gains)[:, :n]
        Wog = self._eff(f"L{l}_OG", gains)[:n]
        gog = self._off(f"L{l}_GOG", gains)
        q = F.linear(hn, Wq).view(B, T, h_act, dh).transpose(1, 2)
        k = F.linear(hn, Wk).view(B, T, hkv, dh).transpose(1, 2)
        v = F.linear(hn, Wv).view(B, T, hkv, dh).transpose(1, 2)
        q, k = rms_norm_headwise(q), rms_norm_headwise(k)
        if cache is not None and int(cache[0].shape[2]) >= cfg.T_s:
            cache = None                          # window-local attention: a full cache ends the window (08 §8.4)
        offset = 0 if cache is None else int(cache[0].shape[2])
        if offset + T > cfg.T_s:
            raise ValueError(f"a window holds at most T_s = {cfg.T_s} spans of attention context; "
                             f"got {offset} cached + {T} new")
        q, k = _rope_at(q, offset), _rope_at(k, offset)
        if cache is not None:
            k = torch.cat([cache[0].to(k.dtype), k], dim=2)
            v = torch.cat([cache[1].to(v.dtype), v], dim=2)
        new_cache = (k.detach(), v.detach())
        rep = h_act // hkv
        kr, vr = k.repeat(1, rep, 1, 1), v.repeat(1, rep, 1, 1)
        Tk = k.shape[2]
        qpos = torch.arange(offset, offset + T, device=hn.device)[:, None]
        kpos = torch.arange(Tk, device=hn.device)[None, :]
        mask = kpos <= qpos
        if explicit:
            scores = (q @ kr.transpose(-1, -2)) / math.sqrt(dh)
            scores = scores.masked_fill(~mask, float("-inf"))
            prob = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            a = prob @ vr
        elif offset == 0:
            a = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=1.0 / math.sqrt(dh))
        else:
            a = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask, scale=1.0 / math.sqrt(dh))
        a = a.transpose(1, 2).reshape(B, T, n)
        a = a * torch.sigmoid(F.linear(hn, Wog) + gog.to(hn.dtype))
        a = a * pattern.head_mask(l, a.device, a.dtype)
        return F.linear(a, Wo), new_cache

    def _ffn(self, l: int, hn: torch.Tensor, pattern: SwitchPattern, gains: Optional[dict]) -> torch.Tensor:
        Fa = pattern.F_act(l)
        Wg = self._eff(f"L{l}_GATE", gains)[:Fa]
        Wu = self._eff(f"L{l}_UP", gains)[:Fa]
        Wd = self._eff(f"L{l}_DOWN", gains)[:, :Fa]
        f = F.silu(F.linear(hn, Wg)) * F.linear(hn, Wu)
        f = f * pattern.slice_mask(l, f.device, f.dtype)
        return F.linear(f, Wd)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, *, state: Optional[dict] = None, cache: Optional[dict] = None,
                pattern: Optional[SwitchPattern] = None, gains: Optional[dict] = None,
                zeta: Optional[torch.Tensor] = None, trace: Optional[list] = None
                ) -> tuple[torch.Tensor, dict, dict, torch.Tensor]:
        """`x (B, T, d_x)` -> `(y_hat (B, T, d_x), state', cache', hidden (B, T, d))`.

        `state` maps delta layers to `(B, h, d_head, d_head)` (zeros from `init_state`
        when absent: a declared boundary); `cache` maps attention layers to `(k, v)`
        of the preceding spans of the window (absent at a window start); `pattern`
        defaults to the adult; `gains` is the backbone-facing gain dictionary of 04;
        `zeta (2L,)` the homeostatic log-gains of 07. `trace`, if a list, receives
        `("mix", l, h_in, S_in)` and `("ffn", l, h_in)` for every executed layer (the
        criticality probe of 07 reads the residual stream at each sub-block's input).
        """
        cfg = self.cfg
        check_shape(x, [None, None, cfg.d_x], name="x")
        B, T, _ = x.shape
        pattern = pattern or SwitchPattern.adult(cfg)
        state = dict(state) if state is not None else self.init_state(B, x.device)
        cache = dict(cache) if cache is not None else {}
        h = F.linear(x, self._eff("IN_W", gains), self.W["IN_b"])
        for l in pattern.active_layers():
            m_l = pattern.layer_m[l]
            if trace is not None:
                trace.append(("mix", l, h.detach(), None if state.get(l) is None else state[l].detach()))
            rho_mix = self._vec(f"L{l}_RMIX", gains)
            hn = rms_norm_gf(h) * rho_mix.to(h.dtype)
            if cfg.op(l) == "delta":
                y, state[l] = self._delta(l, hn, state.get(l), pattern, gains)
            else:
                y, cache[l] = self._attn(l, hn, cache.get(l), pattern, gains)
            n_mix = self._vec(f"L{l}_NMIX", gains)
            if zeta is not None:
                n_mix = n_mix * torch.exp(zeta[2 * l].to(n_mix.dtype))
            h = h + m_l * n_mix.to(h.dtype) * rms_norm_gf(y)
            if trace is not None:
                trace.append(("ffn", l, h.detach()))
            rho_ffn = self._vec(f"L{l}_RFFN", gains)
            hn = rms_norm_gf(h) * rho_ffn.to(h.dtype)
            f = self._ffn(l, hn, pattern, gains)
            n_ffn = self._vec(f"L{l}_NFFN", gains)
            if zeta is not None:
                n_ffn = n_ffn * torch.exp(zeta[2 * l + 1].to(n_ffn.dtype))
            h = h + m_l * n_ffn.to(h.dtype) * rms_norm_gf(f)
        hidden = rms_norm_gf(h) * self._vec("OUT_rho", gains).to(h.dtype)
        y_hat = F.linear(hidden, self._eff("OUT_W", gains))
        return y_hat, state, cache, hidden

    def sub_block(self, kind: str, l: int, H: torch.Tensor, *, S: Optional[torch.Tensor],
                  pattern: SwitchPattern, gains: Optional[dict], zeta: Optional[torch.Tensor]) -> torch.Tensor:
        """The map `f_l^{mix}` or `f_l^{ffn}` of 07 (7.1) on one window `H (1, T, d)` with the
        entering state held fixed: what the criticality probe differentiates."""
        m_l = pattern.layer_m[l]
        if kind == "mix":
            hn = rms_norm_gf(H) * self._vec(f"L{l}_RMIX", gains).to(H.dtype)
            if self.cfg.op(l) == "delta":
                y, _ = self._delta(l, hn, S, pattern, gains)
            else:
                y, _ = self._attn(l, hn, None, pattern, gains, explicit=True)
            n = self._vec(f"L{l}_NMIX", gains)
            if zeta is not None:
                n = n * torch.exp(zeta[2 * l].to(n.dtype))
            return H + m_l * n.to(H.dtype) * rms_norm_gf(y)
        if kind == "ffn":
            hn = rms_norm_gf(H) * self._vec(f"L{l}_RFFN", gains).to(H.dtype)
            f = self._ffn(l, hn, pattern, gains)
            n = self._vec(f"L{l}_NFFN", gains)
            if zeta is not None:
                n = n * torch.exp(zeta[2 * l + 1].to(n.dtype))
            return H + m_l * n.to(H.dtype) * rms_norm_gf(f)
        raise ValueError(kind)

    def detach_state(self, state: dict) -> dict:
        """Truncated BPTT across windows (08 §8.2 S5)."""
        return {l: s.detach() for l, s in state.items()}
