"""The SOMA phenotype: block layout, embeddings, head, and the forward pass.

SPEC: 05 §5.1
SPEC: 05 §5.6

```
x <- x + Norm_post( Mix ( Norm_pre(x) ) )
x <- x + Norm_post( Site( Norm_pre(x) ) )
```

**Peri-LN**: both the input and the output of every sublayer are normalised.
Pre-norm residual streams exhibit variance growth exponential in depth;
post-norm has stable variance but degraded gradient flow.

Blocks alternate mixer and attention on a fixed period (§05.1): at M1
(`L = 12`, period 4) attention sits at `i mod 4 == 3`, giving 3 attention and 9
mixer sublayers — the 3:1 ratio a controlled 72-model study recommends. That
study's own finding is that language-modelling **loss** is nearly flat across
the ratio while **recall** improves sharply below 3:1, so §05.1 requires any
ablation of the ratio to be evaluated on recall-intensive probes, never on loss.

The mixer is **not** cross-layer weight-tied. What is shared is the *parts
library* `(A, B, kappa)`; every mixer, every `W_shared`, and every unit
population is per-block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from soma.config import SomaConfig
from soma.development.population import UnitPopulation
from soma.germline.field import Field
from soma.germline.genome import Genome
from soma.model.attention import Attention
from soma.model.metabolic import MetabolicTerms, metabolic_loss
from soma.model.mixer import Mixer
from soma.model.rmsnorm import RMSNorm
from soma.model.site import PartsLibrary, Site, SiteOutput
from soma.utils.shapes import check_shape

__all__ = ["SomaModel", "ForwardOutput", "DenseSite"]


@dataclass
class ForwardOutput:
    """Logits plus everything selection, the metabolic term and logging need."""

    logits: torch.Tensor
    site_outputs: List[SiteOutput] = field(default_factory=list)
    metabolic: Optional[MetabolicTerms] = None
    log_Z: Optional[torch.Tensor] = None
    residuals: List[torch.Tensor] = field(default_factory=list)
    """Per-block residual-stream inputs, retained only when
    `capture_residuals=True`. §02.5 step 2's gradient-flow test compares
    `||dL/dx||` at the first and last block, which is parameter-count
    independent -- unlike a comparison of raw parameter-gradient norms, where a
    mixer block and an attention block differ in size by 2x at M1."""


class DenseSite(nn.Module):
    """The M0 / B1 Site: a dense `ReLU^2` FFN of inner width `N_G` (§07.4.2).

    SPEC: 08 §8.4

    The baseline is *the same stack* with the Site replaced, so every other
    difference between M0 and SOMA is eliminated by construction. It reuses the
    `PartsLibrary`'s `A` and `B` as its two projections, which is what makes
    "inner width `N_G`" literal rather than merely numerically equal.
    """

    def __init__(self, cfg: SomaConfig, parts: PartsLibrary):
        super().__init__()
        self.cfg = cfg
        self.parts = parts
        self.W_shared = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.W_shared.weight, std=cfg.d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.parts.A.transpose(0, 1)
        h = torch.relu(z).square()
        return h @ self.parts.B.transpose(0, 1) + self.W_shared(x)


class SomaModel(nn.Module):
    """The full phenotype.

    `dense_sites=True` builds the M0 / B1 baseline (§08.4 B1): identical width,
    depth, tokenizer, data, optimizer and schedule, with the Site replaced by a
    dense FFN and no germline, units, selection or stream.
    """

    def __init__(self, cfg: SomaConfig, *, dense_sites: bool = False):
        super().__init__()
        self.cfg = cfg
        self.dense_sites = dense_sites

        self.E_emb = nn.Embedding(cfg.V, cfg.d_model)
        with torch.no_grad():
            nn.init.normal_(self.E_emb.weight, std=cfg.d_model ** -0.5)
        self.emb_scale = math.sqrt(cfg.d_model)
        """§05.6: the embedding is read scaled by `sqrt(d_model)` and the head is
        **not**.

        Under weight tying the same matrix sets two scales that want different
        values. The residual stream should start at RMS ~1, because Peri-LN
        normalises each sublayer's output to RMS 1 before adding it; the head
        should produce logits of order 1. With `Var(E) = 1/d_model` and no
        scaling, the residual starts at RMS `d_model^-1/2` = 0.072 at M-nano, so
        `Norm_pre` amplifies it 14x at block 0 and the residual jumps ~20x in a
        single block. Measured, that is a **43x** (M-nano) to **61x** (M1)
        gradient discontinuity at block 0 against 1.1-2.4x everywhere else --
        the whole of §02.5's gradient-flow failure, concentrated in one place.

        Scaling on read fixes the residual scale and leaves the head's: `x` has
        RMS 1, and `norm_out(x) @ E^T` sums `d_model` terms of size
        `1 * d_model^-1/2`, giving logits of order 1. See AUDIT A-57."""

        self.parts = PartsLibrary(cfg)

        attn_set = set(cfg.attention_layers)
        self.is_attention = [i in attn_set for i in range(cfg.L)]

        self.norm_pre_mix = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.L)])
        self.norm_post_mix = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.L)])
        self.norm_pre_site = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.L)])
        self.norm_post_site = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.L)])

        self.sublayers = nn.ModuleList(
            [Attention(cfg) if self.is_attention[i] else Mixer(cfg) for i in range(cfg.L)]
        )

        if dense_sites:
            self.sites = nn.ModuleList([DenseSite(cfg, self.parts) for _ in range(cfg.L)])
            self.populations = nn.ModuleList()
        else:
            self.sites = nn.ModuleList([Site(cfg, self.parts) for _ in range(cfg.L)])
            self.populations = nn.ModuleList(
                [UnitPopulation(cfg, i) for i in range(cfg.L)]
            )

        self.norm_out = RMSNorm(cfg.d_model)
        # §05.6: the token embedding is tied to the output head at M-nano and M1,
        # untied at M4 and above.
        self.tied_head = True

    # ------------------------------------------------------------------

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.E_emb.weight.transpose(0, 1)

    def seed_populations(
        self, genome: Genome, rng, generator, *, run: str = "run", mutate_fn=None
    ) -> None:
        """`SEED_POPULATION` at every Site (§02.2.4)."""
        for pop in self.populations:
            pop.seed(
                genome.inventory,
                rng,
                generator,
                run=run,
                G_gen=genome.G_gen,
                mutate_fn=mutate_fn,
            )

    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        genome: Optional[Genome] = None,
        doc_id: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        pre_gate: bool = False,
        want_metabolic: bool = True,
        use_reference_mixer: bool = False,
        capture_residuals: bool = False,
        keep_z: bool = False,
        grad_checkpoint: bool = False,
    ) -> ForwardOutput:
        """`tokens` is `(B, T)` int64; returns logits `(B, T, V)`.

        `grad_checkpoint` recomputes each block's interior during the backward
        pass instead of storing it. Measured at M1, one block retains ~1 GB of
        activations for 8192 tokens — 586 MB in the mixer's intermediates and
        399 MB in the Site's — so the stack needs ~14 GB for a batch of 8192,
        and `B_tok = 65536` (§01.9) does not fit in 16 GB at any setting. With
        checkpointing only the `L` block boundaries are retained, and the peak
        is one block's interior plus the boundaries.

        The cost is one extra forward, and §07.8 requires the realised
        wall-clock to be reported rather than the analytic estimate.
        """
        cfg = self.cfg
        check_shape(tokens, [None, None], name="tokens")
        x = self.E_emb(tokens) * self.emb_scale

        site_outputs: List[SiteOutput] = []
        residuals: List[torch.Tensor] = []
        wire = disp = size = None

        for i in range(cfg.L):
            if capture_residuals:
                x = x.clone()
                x.retain_grad()
                residuals.append(x)
            use_ckpt = grad_checkpoint and self.training and torch.is_grad_enabled()
            # --- mixer / attention sublayer -------------------------------
            def _mix(inp, _i=i):
                hh = self.norm_pre_mix[_i](inp)
                if self.is_attention[_i]:
                    hh = self.sublayers[_i](
                        hh, doc_id=doc_id, key_padding_valid=valid
                    )
                else:
                    hh, _ = self.sublayers[_i](
                        hh, valid=valid, use_reference=use_reference_mixer
                    )
                return self.norm_post_mix[_i](hh)

            if use_ckpt:
                x = x + _ckpt.checkpoint(_mix, x, use_reentrant=False)
            else:
                x = x + _mix(x)

            # --- Site sublayer --------------------------------------------
            h = self.norm_pre_site[i](x)
            if self.dense_sites:
                s = self.sites[i](h)
                x = x + self.norm_post_site[i](s)
            else:
                pop = self.populations[i]
                assert genome is not None, "SOMA Sites require a genome"

                def _site(inp, _i=i, _pop=pop):
                    o = self.sites[_i](
                        inp,
                        field=genome.field_,
                        p_u=_pop.p_u,
                        g_u=_pop.g_u,
                        m_u=_pop.m_u,
                        active=_pop.active,
                        pre_gate=pre_gate,
                        keep_z=keep_z,
                        n_active=_pop.n_active_cached,
                    )
                    # `checkpoint` requires a flat tuple of tensors, so the
                    # dataclass is unpacked here and rebuilt below. The aux
                    # tensors are what §04's credit and §03.4's utility read;
                    # they are recomputed with the block, which is exactly the
                    # trade checkpointing makes.
                    return (
                        self.norm_post_site[_i](o.y),
                        o.delta_norm, o.slots, o.r_ut, o.mask_u, o.z,
                        o.n_candidates,
                    )

                if use_ckpt:
                    y, dn, sl, rr, mu, zz, nc = _ckpt.checkpoint(
                        _site, h, use_reentrant=False
                    )
                else:
                    y, dn, sl, rr, mu, zz, nc = _site(h)
                out = SiteOutput(y, dn, sl, rr, mu, zz, nc)
                site_outputs.append(out)
                x = x + y

                if want_metabolic:
                    q_t = self.sites[i].positions_query(h)
                    terms = metabolic_loss(
                        q_t,
                        pop.p_u,
                        out.slots,
                        # (1.8) is the **only** term written on `r_ut * m_u`;
                        # `r_ut` is now the routing weight alone (audit A-76).
                        out.r_ut * out.mask_u,
                        pop.m_u,
                        pop.live,
                        genome.constants(),
                    )
                    wire = terms.L_wire if wire is None else wire + terms.L_wire
                    disp = terms.L_disp if disp is None else disp + terms.L_disp
                    size = terms.L_size if size is None else size + terms.L_size

        x = self.norm_out(x)
        logits = self.head(x)

        met = None
        if wire is not None:
            met = MetabolicTerms(wire, disp, size)  # type: ignore[arg-type]

        return ForwardOutput(logits, site_outputs, met, None, residuals)

    # ------------------------------------------------------------------
    # Accounting (§05.10)
    # ------------------------------------------------------------------

    def parameter_report(self) -> Dict[str, int]:
        # SPEC: 05 §5.10
        """Parameter counts by component, for §05.10 and `G-SPECNUM`.

        Computed by enumerating the live modules rather than by re-deriving the
        arithmetic, because §05.10's own numbers were quoted for a configuration
        that had not been recomputed (audit A-44's neighbour).
        """
        cfg = self.cfg
        rep: Dict[str, int] = {}
        rep["embedding"] = self.E_emb.weight.numel()
        rep["parts_library"] = sum(p.numel() for p in self.parts.parameters())
        # The mixer's internal output norm (§05.2.4's `RMSNorm(o)`) belongs to
        # the mixer block, not to the block-level Peri-LN gains, and is counted
        # exactly once. Spec v3.0's §05.10 table counted it in neither row and
        # understated the total by 4,608 at M1 (audit A-49).
        mix = attn = 0
        for i, sub in enumerate(self.sublayers):
            n = sum(p.numel() for p in sub.parameters())
            if self.is_attention[i]:
                attn += n
            else:
                mix += n
        rep["mixer_blocks"] = mix
        rep["attention_blocks"] = attn
        rep["sites"] = sum(
            p.numel()
            for s in self.sites
            for name, p in s.named_parameters()
            if not name.startswith("parts.")
        )
        rep["populations"] = sum(
            p.numel() for pop in self.populations for p in pop.parameters()
        )
        rep["norm_gains"] = sum(
            m.gain_rms.numel()
            for m in list(self.norm_pre_mix)
            + list(self.norm_post_mix)
            + list(self.norm_pre_site)
            + list(self.norm_post_site)
            + [self.norm_out]
            if m.gain_rms is not None
        )
        rep["total"] = sum(p.numel() for p in self.parameters())
        rep["non_embedding"] = rep["total"] - rep["embedding"]
        return rep

    def optimizer_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """§05.7's groups, **enumerated by name, not by tensor rank**.

        SPEC: 05 §5.7

        | Group | Members | Optimizer |
        |---|---|---|
        | **M** | `W_qkv`, `W_gate`, `OutProj`, `W_q`, `W_k`, `W_v`, `W_o`, `A`, `B`, `W_shared`, `W_key`, `W_ctx`, `kappa`, `K_unit1`, `K_unit2` | Muon |
        | **A** | `E`, all `gain_rms`, `z_sink`, `w_alpha`, `w_beta`, `W_pos`, **`p_u`**, conv filters | AdamW |

        Spec v1's "all 2-D soma matrices" rule wrongly captured both `p_u` and
        the embeddings. `p_u` is a `U_max x m_dim` matrix with `m_dim = 8`:
        Newton-Schulz would drive all 8 singular values toward 1, destroying the
        *relative* geometry on `S^7` that `phi_g` reads and `L_disp` shapes, and
        Muon's `0.2 sqrt(max(rows,cols))` scaling with `rows = U_max` gives a
        `6.4x` step multiplier on a sphere-constrained parameter.
        """
        muon_names = (
            "W_qkv", "W_gate", "OutProj", "W_q", "W_k", "W_v", "W_o",
            "parts.A", "parts.B", "parts.kappa", "W_shared", "W_key", "W_ctx",
            "K_unit1", "K_unit2",
        )
        adamw_names = (
            "E_emb", "gain_rms", "z_sink", "w_alpha", "w_beta", "W_pos",
            "p_u", "conv",
        )
        muon: List[nn.Parameter] = []
        adamw: List[nn.Parameter] = []
        unassigned: List[str] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(f".{m}." in f".{name}." or name.endswith(m) or f".{m}" in name
                   for m in muon_names):
                muon.append(p)
            elif any(a in name for a in adamw_names):
                adamw.append(p)
            else:
                unassigned.append(name)
        if unassigned:
            raise ValueError(
                "§05.7 enumerates optimizer groups by name and the table is "
                f"exhaustive; these parameters match neither group: {unassigned}"
            )
        return muon, adamw
