"""The latent bank `Z` and its prior.

SPEC: 02 §2.6

`Z` is the acquired half of identity. It is fit by regression (§05) under a
Gaussian prior, never learned by gradient on the language loss -- the runaway
norm of GNOMON (§00.2) is a property of learning identities behind a saturating
map with Adam's fixed-size steps, and the regression's prior is what removes it.
New units' latents are *sampled* from the fitted moments of their axis (§02.6),
which is the stochastic decoding the PNAS paper proposed and never did.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import (
    AX_GATE, AX_HEAD, AX_INNER, AX_KV, AX_RES, AX_TOK, GenomeShape, StageConfig,
)

__all__ = ["LatentBank", "SIGMA_Z"]

SIGMA_Z = 1.0
"""SPEC: 02 §2.6 -- the prior scale of every latent."""

PER_LAYER_AXES = (AX_INNER, AX_HEAD, AX_GATE)


class LatentBank(nn.Module):
    """`Z` at one stage. SPEC: 02 §2.6.

    Tensors: `z_res (d, d_z)`, `z_inner (L, F, d_z)`, `z_head (L, h d_k, d_z)`,
    `z_gate (L, 2h, d_z)`, `z_tok (V, d_z)`, `y (L, d_y)`. `KV(l)` is the prefix
    of `HEAD(l)` and shares its latents.
    """

    def __init__(self, shape: GenomeShape, cfg: StageConfig, vocab: int, *,
                 seed: int = 0, init: str = "prior"):
        super().__init__()
        self.shape, self.cfg, self.vocab = shape, cfg, vocab
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        d_z, d_y = shape.d_z, shape.d_y

        def draw(*size):
            if init == "zeros":
                return torch.zeros(*size)
            return torch.randn(*size, generator=g) * SIGMA_Z

        self.z_res = nn.Parameter(draw(cfg.d_model, d_z))
        self.z_inner = nn.Parameter(draw(cfg.L, cfg.d_ff, d_z))
        self.z_head = nn.Parameter(draw(cfg.L, cfg.n_h * cfg.d_head, d_z))
        self.z_gate = nn.Parameter(draw(cfg.L, 2 * cfg.n_h, d_z))
        self.z_tok = nn.Parameter(draw(vocab, d_z))
        self.y = nn.Parameter(draw(cfg.L, d_y))
        self.requires_grad_(False)

    # ------------------------------------------------------------------ access
    def unit_latents(self, axis: str, l: int, n: int) -> torch.Tensor:
        """`(n, d_z)` latents for the first `n` units of `axis` in layer `l`."""
        if axis == AX_RES:
            t = self.z_res
        elif axis == AX_TOK:
            t = self.z_tok
        elif axis == AX_INNER:
            t = self.z_inner[l]
        elif axis in (AX_HEAD, AX_KV):
            t = self.z_head[l]
        elif axis == AX_GATE:
            t = self.z_gate[l]
        else:
            # TAP and ONE carry no latent (§02.1): zeros of the right width.
            return torch.zeros(n, self.shape.d_z, device=self.z_res.device,
                               dtype=self.z_res.dtype)
        if n > t.shape[0]:
            raise ValueError(f"{axis} layer {l}: asked {n} latents, bank holds {t.shape[0]}")
        return t[:n]

    def layer_latent(self, l: int) -> torch.Tensor:
        """`(1, d_y)`; zeros for the global pseudo-layer."""
        if l == -1:
            return torch.zeros(1, self.shape.d_y, device=self.y.device, dtype=self.y.dtype)
        return self.y[l:l + 1]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_table(self) -> dict[str, int]:
        """The rows of §02.6, for `G-SPECNUM`."""
        return {name: p.numel() for name, p in self.named_parameters()}

    # ---------------------------------------------------------------- moments
    @torch.no_grad()
    def moments(self, axis: str, l: Optional[int] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mu_X, s_X)` per coordinate. SPEC: 02 §2.6.

        `RES`, `TOK`: over all units. Per-layer axes: over layer `l`'s units.
        A single-unit axis returns `s = SIGMA_Z` so sampling stays defined.
        """
        if axis == AX_RES:
            t = self.z_res
        elif axis == AX_TOK:
            t = self.z_tok
        elif axis == AX_INNER:
            t = self.z_inner[l]
        elif axis in (AX_HEAD, AX_KV):
            t = self.z_head[l]
        elif axis == AX_GATE:
            t = self.z_gate[l]
        else:
            raise KeyError(axis)
        mu = t.mean(0)
        s = t.std(0, unbiased=False) if t.shape[0] > 1 else torch.full_like(mu, SIGMA_Z)
        return mu, s

    @torch.no_grad()
    def layer_moments(self, neighbours: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Moments of `y` over the given existing layers (a new layer's prior)."""
        t = self.y[neighbours]
        mu = t.mean(0)
        s = t.std(0, unbiased=False) if t.shape[0] > 1 else torch.full_like(mu, SIGMA_Z)
        return mu, s

    @torch.no_grad()
    def sample_new(self, axis: str, n: int, l: Optional[int], generator: torch.Generator,
                   *, mode: str = "sample") -> torch.Tensor:
        """`(n, d_z)` latents for `n` new units. SPEC: 02 §2.6.

        `mode`: `"sample"` (Definition 2), `"mean"` (the MEAN ablation),
        `"zero"` (the ZERO control). At `d_z = 0` every mode returns `(n, 0)`.
        """
        if self.shape.d_z == 0:
            return torch.zeros(n, 0, device=self.z_res.device)
        if mode == "zero":
            return torch.zeros(n, self.shape.d_z, device=self.z_res.device)
        mu, s = self.moments(axis, l)
        if mode == "mean":
            return mu.unsqueeze(0).expand(n, -1).clone()
        if mode != "sample":
            raise ValueError(f"unknown mode {mode!r}")
        eps = torch.randn(n, self.shape.d_z, generator=generator).to(mu.device)
        return mu + s * eps

    # ------------------------------------------------------------------ shrink
    @torch.no_grad()
    def shrunk(self, cfg_small: StageConfig) -> "LatentBank":
        """The stage-`s` bank a next generation is born from. SPEC: 05 §5.4, 07 §7.2.

        The inverse of `grown`'s spreading: unit prefixes on every channel axis,
        and the `L_s` layers at slots `round(i (L - 1)/(L_s - 1))` of the adult,
        which are the positions the stage-`s` layers were spread to (§06.2 step
        3); their latents `y` and per-layer unit latents travel with them.
        """
        big = self.cfg
        if (cfg_small.d_model > big.d_model or cfg_small.d_ff > big.d_ff
                or cfg_small.n_h > big.n_h or cfg_small.L > big.L):
            raise ValueError("shrunk() needs a stage no larger than the bank's on every axis")
        new = LatentBank(self.shape, cfg_small, self.vocab, init="zeros").to(self.z_res.device)
        new.z_res.copy_(self.z_res[: cfg_small.d_model])
        new.z_tok.copy_(self.z_tok)
        Ls, L = cfg_small.L, big.L
        for i in range(Ls):
            j = 0 if Ls == 1 else int(round(i * (L - 1) / (Ls - 1)))
            new.z_inner[i] = self.z_inner[j, : cfg_small.d_ff]
            new.z_head[i] = self.z_head[j, : cfg_small.n_h * cfg_small.d_head]
            new.z_gate[i] = self.z_gate[j, : 2 * cfg_small.n_h]
            new.y[i] = self.y[j]
        return new

    # ------------------------------------------------------------------ growth
    @torch.no_grad()
    def grown(self, cfg_new: StageConfig, layer_map: list[int], generator: torch.Generator,
              *, mode: str = "sample") -> "LatentBank":
        """A bank for `cfg_new` keeping every existing latent. SPEC: 02 §2.6, 06 §6.2.

        `layer_map[l_new]` is the index of the existing layer placed at `l_new`,
        or `-1` for a new layer. New units on existing layers and all units of
        new layers are sampled from the moments of §02.6.
        """
        new = LatentBank(self.shape, cfg_new, self.vocab, init="zeros")
        dev = self.z_res.device
        new.to(dev)
        d_z = self.shape.d_z
        # RES
        new.z_res[: self.cfg.d_model] = self.z_res
        if cfg_new.d_model > self.cfg.d_model:
            new.z_res[self.cfg.d_model:] = self.sample_new(
                AX_RES, cfg_new.d_model - self.cfg.d_model, None, generator, mode=mode)
        # TOK unchanged
        new.z_tok.copy_(self.z_tok)
        # per-layer axes
        existing = [l for l in range(cfg_new.L) if layer_map[l] >= 0]
        for l_new in range(cfg_new.L):
            src = layer_map[l_new]
            if src >= 0:
                # keep old units, sample the new ones
                for axis, t_old, t_new, n_old, n_new in (
                    (AX_INNER, self.z_inner[src], new.z_inner[l_new], self.cfg.d_ff, cfg_new.d_ff),
                    (AX_HEAD, self.z_head[src], new.z_head[l_new],
                     self.cfg.n_h * self.cfg.d_head, cfg_new.n_h * cfg_new.d_head),
                    (AX_GATE, self.z_gate[src], new.z_gate[l_new], 2 * self.cfg.n_h, 2 * cfg_new.n_h),
                ):
                    t_new[:n_old] = t_old
                    if n_new > n_old:
                        t_new[n_old:] = self.sample_new(axis, n_new - n_old, src, generator, mode=mode)
                new.y[l_new] = self.y[src]
            else:
                # a new layer: its units are sampled from the nearest existing
                # layers' moments (two nearest by position in the new layout).
                nb = sorted(existing, key=lambda e: abs(e - l_new))[:2]
                src_nb = [layer_map[e] for e in nb]
                for axis, t_new, n_new, src_tensor in (
                    (AX_INNER, new.z_inner[l_new], cfg_new.d_ff, self.z_inner),
                    (AX_HEAD, new.z_head[l_new], cfg_new.n_h * cfg_new.d_head, self.z_head),
                    (AX_GATE, new.z_gate[l_new], 2 * cfg_new.n_h, self.z_gate),
                ):
                    if d_z == 0:
                        continue
                    pool = torch.cat([src_tensor[s] for s in src_nb], 0)
                    mu = pool.mean(0)
                    s_ = pool.std(0, unbiased=False) if pool.shape[0] > 1 else torch.full_like(mu, SIGMA_Z)
                    if mode == "zero":
                        t_new.zero_()
                    elif mode == "mean":
                        t_new[:] = mu
                    else:
                        t_new[:] = mu + s_ * torch.randn(n_new, d_z, generator=generator).to(dev)
                mu_y, s_y = self.layer_moments(src_nb)
                if mode == "zero":
                    new.y[l_new].zero_()
                elif mode == "mean":
                    new.y[l_new] = mu_y
                else:
                    new.y[l_new] = mu_y + s_y * torch.randn(self.shape.d_y, generator=generator).to(dev)
        return new
