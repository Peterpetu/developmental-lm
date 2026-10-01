"""The free-weight twin: the same network, with its weights as free parameters.

SPEC: 04 §4.7 (P1)
SPEC: 09 -- the baselines

Two jobs, and they are the same object because they must be:

1. **P1, twin equivalence.** Decoding `Gamma` and loading the result here must
   give bit-comparable logits. That is what makes the compression claim a claim
   about *this* model rather than about a different one that happens to score
   similarly.
2. **The baseline of §09.** Comparing GNOMON against a transformer is only a
   comparison of the bottleneck if the transformer is architecturally
   identical. Re-implementing the forward pass here would confound the
   measurement with implementation differences -- a different masking
   convention, a different normalisation placement -- so the twin **shares**
   `Gnomon.forward` and differs from it in exactly one respect: where the
   weights come from.

The twin is trained with Muon on its matrix parameters (SOMA's vendored
implementation), because using anything weaker would understate the baseline
and make the compression look better than it is.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import GenomeShape, PhenoConfig
from ..genome.decoder import s_target, vector_offset
from .model import Gnomon, Weights
from .plan import Arch

__all__ = ["FreeTwin"]


def _key(role: int, layer: int) -> str:
    return f"r{role}_l{layer}"


class FreeTwin(nn.Module):
    """Every decoded role held as an `nn.Parameter`."""

    def __init__(self, cfg: PhenoConfig, *, arch: Optional[Arch] = None,
                 seed: int = 0, shape: Optional[GenomeShape] = None):
        super().__init__()
        self.cfg = cfg
        self.arch = arch or Arch.uniform(cfg)
        # A `Gnomon` supplies the forward pass and the key set. Its own decoder
        # and soma are never used or optimised here; `parameters()` below is
        # restricted to `self.W` so they cannot leak into a baseline's training.
        self._host = Gnomon(cfg, shape or GenomeShape(), arch=self.arch)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = nn.ParameterDict()
        for (role, layer), shp in self._host.shapes().items():
            t = torch.randn(*shp, generator=g) * s_target(role, shp, cfg,
                                                          self.arch.L)
            self.W[_key(role, layer)] = nn.Parameter(t + vector_offset(role))

    # `self._host` is a submodule, so its decoder and soma would otherwise
    # appear in `parameters()` and be optimised alongside the twin's weights --
    # exactly the kind of silent extra capacity §04.6 exists to forbid.
    def parameters(self, recurse: bool = True):
        return iter(list(self.W.values()))

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True):
        return iter([(f"W.{k}", v) for k, v in self.W.items()])

    def weights(self) -> Weights:
        return {(role, layer): self.W[_key(role, layer)]
                for role, layer in self._host.decode_keys()}

    def load_decoded(self, W: Weights) -> "FreeTwin":
        """Place a decoded phenotype into the free parameters. SPEC: 04 §4.7 P1."""
        with torch.no_grad():
            for (role, layer), v in W.items():
                p = self.W[_key(role, layer)]
                if p.shape != v.shape:
                    raise ValueError(
                        f"role {role} layer {layer}: twin has {tuple(p.shape)}, "
                        f"decode gave {tuple(v.shape)}"
                    )
                p.copy_(v)
        return self

    def forward(self, tokens: torch.Tensor, **kw) -> torch.Tensor:
        return self._host(tokens, weights=self.weights(), **kw)

    def loss(self, tokens: torch.Tensor, **kw) -> torch.Tensor:
        return self._host.loss(tokens, weights=self.weights(), **kw)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def matrix_params(self) -> list[nn.Parameter]:
        """The 2-D weights, for Muon; everything else goes to AdamW."""
        return [p for p in self.parameters() if p.dim() == 2]

    def other_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.dim() != 2]
