"""The phenotype: a transformer whose every weight is decoded from the genome.

SPEC: 01 §1.1
SPEC: 01 §1.2
SPEC: 04 §4.1
SPEC: 04 §4.3
SPEC: 04 §4.4
SPEC: 04 §4.6
SPEC: 05 §5.5
SPEC: 07 §7.1

There is no `nn.Parameter` here holding a weight. The only parameters a
`Gnomon` owns are `Gamma_w` (the decoder) and `Z` (the soma); every `W` used in
the forward pass is an output of `Decoder.decode_many`, recomputed each step
because `W` is a function of `Z` and `Z` moves (§05.1's line 5).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    ROLE_DELTA_BETA, ROLE_EMB, ROLE_FFN_DOWN, ROLE_FFN_GATE, ROLE_FFN_UP,
    ROLE_K, ROLE_LOGIT_SCALE, ROLE_NORM_FFN, ROLE_NORM_FINAL, ROLE_NORM_MIX,
    ROLE_O, ROLE_PLAST_CHAN, ROLE_PLAST_TOK, ROLE_Q, ROLE_SINK, ROLE_V,
    GenomeShape, PhenoConfig,
)
from ..genome.decoder import Decoder, Soma, s_target, vector_offset
from . import ops
from .plan import GLOBAL_ROLES, Arch

__all__ = ["Arch", "Gnomon", "Weights", "GLOBAL_ROLES"]

Weights = dict           # {(role, layer) -> Tensor}; layer = -1 for global roles


class Gnomon(nn.Module):
    """`Gamma_w` + `Z`, and the forward pass they jointly determine."""

    def __init__(self, cfg: PhenoConfig, shape: GenomeShape, *,
                 genome_seed: int = 0, soma_seed: int = 1,
                 arch: Optional[Arch] = None,
                 decoder: Optional[Decoder] = None):
        super().__init__()
        self.cfg, self.shape = cfg, shape
        self.arch = arch or Arch.uniform(cfg)
        if self.arch.L > cfg.L:
            raise ValueError(f"arch has {self.arch.L} layers, cfg allows {cfg.L}")
        self.decoder = (decoder if decoder is not None
                        else Decoder(shape, seed=genome_seed))
        self.soma = Soma(cfg, shape, seed=soma_seed)
        # SPEC: 04 §4.1 -- see `ops.embed`. `False` keeps the deterministic
        # embedding backward, which §05.7's L7 requires; only the runner's
        # CUDA-graph backend turns it on, and it says why.
        self.capturable = False
        self._prepare()

    def _prepare(self) -> None:
        """Build the innate code tables eagerly; never under `torch.compile`."""
        self.decoder.prepare(self.cfg, self.arch.widest_axis(self.cfg),
                             self.arch.L)

    def _apply(self, *a, **kw):
        out = super()._apply(*a, **kw)
        out._prepare()                    # `.to(device)` changes the cache key
        return out

    def set_arch(self, arch: Arch) -> "Gnomon":
        """Change the decoded architecture. Shapes move; `Gamma` does not."""
        if arch.L > self.cfg.L:
            raise ValueError(f"arch has {arch.L} layers, cfg allows {self.cfg.L}")
        self.arch = arch
        self._prepare()
        return self

    # ------------------------------------------------------------- parameters

    def genome_parameters(self):
        return list(self.decoder.parameters())

    def soma_parameters(self):
        return list(self.soma.parameters())

    # ----------------------------------------------------------------- decode

    def decode_keys(self) -> list[tuple[int, int]]:
        return self.arch.decode_keys()

    def shapes(self) -> dict:
        return {(r, l): self.arch.role_shape(self.cfg, r, l)
                for r, l in self.decode_keys()}

    def decode(self) -> Weights:
        """Every weight the forward pass needs. SPEC: 04 §4.1.

        Only the roles the decoded architecture actually uses are decoded: an
        attention layer never materialises `DELTA_BETA` and a delta layer never
        materialises `SINK`. A role no layer uses receives no gradient, which is
        the honest behaviour and is what §07's operator mutations exploit.
        """
        return self.decoder.decode_many(self.decode_keys(), self.soma, self.cfg,
                                        shapes=self.shapes())

    @torch.no_grad()
    def inference_weights(self) -> Weights:
        """Decode once, for repeated use. SPEC: 03 §3.7, SPEC: 04 §4.7 P7.

        The production path. `Z` does not move at inference, so `W` is constant
        and the decode is a **one-off cost**: throughput afterwards is exactly
        an ordinary transformer's of the same shape, and the artifact that has
        to be shipped is `|Gamma| + |Z|` -- 20.7 KB + 544.4 KB at M1 -- which
        expands to 49.9M weights.

        This is the opposite of §05.1's line 5 and does not contradict it. A
        *training* decode must be inside the loop because `Z` moves and a cached
        `W` would be stale, severing `dL/dZ`; an *inference* decode must be
        outside it because `Z` does not move and re-decoding is pure waste.
        `G-INFER` asserts both halves: the cached logits are bit-identical to
        the per-step ones, and the cache is measurably faster.

        Detached, because nothing downstream of it may reach `Gamma` or `Z`: an
        inference path that accumulated a graph over a long generation would
        grow without bound.
        """
        return {k: v.detach() for k, v in self.decode().items()}

    def random_weights(self, *, seed: int) -> Weights:
        """`W_rand` for §05.5's annealing: the innate scale, no genome content."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        dev, dt = self.decoder.W_row.device, self.decoder.W_row.dtype
        out: Weights = {}
        for (role, l), shp in self.shapes().items():
            sc = s_target(role, shp, self.cfg, self.arch.L)
            t = torch.randn(*shp, generator=g) * sc + vector_offset(role)
            out[(role, l)] = t.to(device=dev, dtype=dt)
        return out

    # ---------------------------------------------------------------- forward

    def forward(self, tokens: torch.Tensor, *, weights: Optional[Weights] = None,
                eps=0.0, W_rand: Optional[Weights] = None,
                doc_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`tokens` is `(B, T)` int64; returns logits `(B, T, V)`."""
        cfg, arch = self.cfg, self.arch
        W = self.decode() if weights is None else weights
        if isinstance(eps, torch.Tensor) or eps > 0.0:
            if W_rand is None:
                raise ValueError("eps > 0 requires W_rand (SPEC: 05 §5.5)")
            W = {k: eps * W_rand[k] + (1.0 - eps) * v for k, v in W.items()}

        E = W[(ROLE_EMB, -1)]
        x = ops.embed(tokens, E, capturable=self.capturable)
        branch_scale = 1.0 / (2.0 * arch.L) ** 0.5

        for l in range(arch.L):
            # SPEC: 04 §4.3 -- Peri-LN: gain-free normalisation *into* the
            # sublayer, decoded-gain normalisation *out* of it. The decoded
            # matrices are structured, not iid, so a `sqrt(fan_in)` variance
            # argument does not bound the branch output (measured RMS 278
            # against a predicted 0.2); normalising the output does, for every
            # genome, which is what makes an arbitrary mutant decode to a
            # trainable network (§03.2's first reason).
            h = ops.rms_norm(x, 1.0)
            if arch.ops[l] == "attn":
                y = ops.attention(
                    h, W[(ROLE_Q, l)], W[(ROLE_K, l)], W[(ROLE_V, l)],
                    W[(ROLE_O, l)], W[(ROLE_SINK, l)],
                    n_h=arch.n_h[l], n_kv=cfg.n_kv, d_head=cfg.d_head,
                    window=arch.window[l], doc_id=doc_id,
                )
            else:
                y, _ = ops.delta_mixer(
                    h, W[(ROLE_Q, l)], W[(ROLE_K, l)], W[(ROLE_V, l)],
                    W[(ROLE_DELTA_BETA, l)], W[(ROLE_O, l)],
                    n_h=arch.n_h[l], n_kv=cfg.n_kv, d_head=cfg.d_head,
                    chunk_size=cfg.chunk_size,
                )
            x = x + ops.rms_norm(y, W[(ROLE_NORM_MIX, l)]) * branch_scale
            f = ops.swiglu(ops.rms_norm(x, 1.0), W[(ROLE_FFN_GATE, l)],
                           W[(ROLE_FFN_UP, l)], W[(ROLE_FFN_DOWN, l)])
            x = x + ops.rms_norm(f, W[(ROLE_NORM_FFN, l)]) * branch_scale

        # SPEC: 04 §4.3 -- cosine logits with a decoded temperature. The tied
        # head shares the coherence problem: `x` is built out of `E`'s own rows,
        # so `x . E_v` is far larger than the `sqrt(d)` an independence argument
        # predicts (measured std 76 against a predicted 16). Normalising both
        # sides removes the dependence on that structure and leaves the genome
        # one clean control, the temperature.
        x = ops.rms_norm(x, W[(ROLE_NORM_FINAL, -1)])
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Eh = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tau = torch.exp(torch.clamp(W[(ROLE_LOGIT_SCALE, -1)][0], -2.0, 5.0))
        return F.linear(x, Eh) * tau

    def loss(self, tokens: torch.Tensor, **kw) -> torch.Tensor:
        """Next-token cross-entropy in nats. SPEC: 05 §5.2 -- and nothing else."""
        logits = self(tokens[:, :-1], **kw)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tokens[:, 1:].reshape(-1))

    # -------------------------------------------------------------- plasticity

    def plasticity(self, W: Optional[Weights] = None) -> dict[str, torch.Tensor]:
        """The decoded plasticity-rule coefficients. SPEC: 10 §10.2, §10.3.

        Returns `(n, N_RULE)` per acquired-identity tensor: columns are
        `(A, B, C, D)` of §10.2's rule, already mapped through §10.3's ranges,
        so `A=0, B=1, C=0, D=0` -- and therefore AdamW -- is what an
        uninformative genome produces.

        Detached. Differentiating through the update would make it a
        second-order object and let a lifetime minimise its loss by switching
        its own plasticity off, which is a degenerate optimum rather than a
        learning mechanism.
        """
        from ..lifetime.plasticity import coefficients_from

        with torch.no_grad():
            W = self.decode() if W is None else W
            chan = torch.stack([W[(ROLE_PLAST_CHAN, l)].detach()
                                for l in range(self.arch.L)]).mean(0)
            tok = W[(ROLE_PLAST_TOK, -1)].detach()
            return {"z": coefficients_from(chan),
                    "zeta": coefficients_from(tok)}

    @torch.no_grad()
    def health(self, role: int = ROLE_Q, layer: int = 0) -> dict:
        """Is the decoder still decoding? SPEC: 03 §3.8 D5, SPEC: 07 §7.4.

        **The instrument that was missing.** D5a and D5b measure decoded
        `erank` at *initialisation*; `KG-RANK` measures it across *generations*,
        also at birth. Nothing measured it **during a lifetime** -- which is the
        only place the 40,000-step collapse happened, and why a run could slide
        from 5.22 back to the unigram entropy over 34,000 steps with every gate
        green.

        Four numbers, chosen because together they identify the failure rather
        than merely detecting it:

        ``z_rms``, ``zeta_rms``   the identity norms. Unbounded growth here is
                                  the cause; everything below is consequence.
        ``responsive``            fraction of the decoder's first hidden layer
                                  with `|h1| <= 0.99`. Saturation is the
                                  mechanism by which a large `z` destroys the
                                  decode: 0.321 at initialisation, 0.001 after
                                  the collapse.
        ``erank``                 decoded `erank / min(d)`. The effect: 0.2507
                                  at initialisation, 0.0065 after. §03.8's floor
                                  is 0.20, so this crosses it long before the
                                  loss shows anything.

        Cheap enough to run at every evaluation: one decode of one matrix and
        one SVD of it, against a step that decodes the whole network.
        """
        from ..genome.decoder import AXIS_KIND
        from ..utils.stats import effective_rank

        cache = self.decoder.codes(self.cfg)
        shp = self.arch.role_shape(self.cfg, role, layer)
        u_ctx = self.decoder._u_ctx(layer, role, self.soma, cache)
        row_kind, col_kind = AXIS_KIND[role]
        rf = self.decoder._axis(row_kind, shp[0], self.soma, cache, column=False)
        cf = self.decoder._axis(col_kind, shp[1], self.soma, cache, column=True)
        ur = F.linear(rf, self.decoder.W_row)
        uc = F.linear(cf, self.decoder.W_col)
        h1 = torch.tanh(torch.exp(self.decoder.log_gain)
                        * (ur.unsqueeze(1) + uc.unsqueeze(0) + u_ctx)
                        + self.decoder.c1)
        W = self.decoder.decode(role, layer, self.soma, self.cfg,
                                shape_out=shp).float()
        finite = torch.isfinite(W).all()
        er = (effective_rank(torch.linalg.svdvals(W).tolist()) / min(W.shape)
              if finite else float("nan"))
        return {
            "z_rms": float(self.soma.z.pow(2).mean().sqrt()),
            "zeta_rms": float(self.soma.zeta.pow(2).mean().sqrt()),
            "responsive": float((h1.abs() <= 0.99).float().mean()),
            "erank": er,
            "finite": bool(finite),
        }

    # ------------------------------------------------------------------ counts

    def n_genome(self) -> int:
        return self.decoder.n_params()

    def n_soma(self) -> int:
        return self.soma.n_params()

    def n_decoded(self) -> int:
        return self.arch.n_decoded(self.cfg)
