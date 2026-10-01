"""`Field` — the expression map: gene centres, widths, and `phi`.

SPEC: 01 §1.2
SPEC: 02 §2.2.1
SPEC: 02 §2.2.2

Gene `g` is a radial basis function on the unit sphere with centre `mu_g` in
`S^{m_dim-1}` and width `sigma_g > 0`:

    phi_g(p) = exp( - || p - mu_g ||^2 / (2 sigma_g^2) )                (1.1)

The germline cost is `(m_dim + 1) * N_G` reals, independent of `|U|`, `d_model`
and `L`. This is the bottleneck property; §09.3 and K9 rest on it, and gate
`G-GERMSIZE` measures it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from soma.config import GERMLINE_SCALE, SIGMA_G_INIT, GermlineScale
from soma.germline.registry import GermlineRegistry
from soma.utils.shapes import check_shape

__all__ = ["Field", "sample_sphere", "phi_from"]


def sample_sphere(
    n: int,
    dim: int,
    generator: Optional[torch.Generator] = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`n` points uniform on `S^{dim-1}`.

    SPEC: 02 §2.2.1

    Normalised isotropic Gaussians. Uniform on the sphere by rotational
    invariance of the Gaussian, which is the only construction that is uniform
    for every `dim` — sampling coordinates uniformly in a box and normalising
    concentrates on the corners' directions.
    """
    if n <= 0 or dim <= 0:
        raise ValueError(f"need n > 0 and dim > 0; got n={n}, dim={dim}")
    device = torch.device(device)
    g = torch.randn(n, dim, generator=generator, device=device, dtype=dtype)
    # A row of exact zeros has no direction; probability ~0 but it would produce
    # NaN, and a NaN centre would poison every phi thereafter.
    norms = g.norm(dim=-1, keepdim=True)
    bad = norms.squeeze(-1) < 1e-12
    if bool(bad.any()):
        g[bad] = torch.nn.functional.one_hot(
            torch.zeros(int(bad.sum()), dtype=torch.long, device=device), dim
        ).to(dtype)
        norms = g.norm(dim=-1, keepdim=True)
    return g / norms


def phi_from(
    p: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """(1.1) evaluated for a batch of positions against every gene.

    SPEC: 01 §1.2

    `p` is `(..., m_dim)`, `mu` is `(N_G, m_dim)`, `sigma` is `(N_G,)`.
    Returns `(..., N_G)` in `[0, 1]`.

    Both arguments lie on the unit sphere, so
    `||p - mu||^2 = 2 - 2 <p, mu>` exactly. Using the inner-product form rather
    than an explicit difference avoids materialising a `(..., N_G, m_dim)`
    intermediate, which at `N_G = 2048` and a probe batch of 4096 units would be
    268 MB for a quantity that reduces immediately.
    """
    if p.shape[-1] != mu.shape[-1]:
        raise ValueError(
            f"position dim {p.shape[-1]} != gene centre dim {mu.shape[-1]}"
        )
    sq = 2.0 - 2.0 * (p @ mu.transpose(-1, -2))
    # <p, mu> can exceed 1 by ~1e-7 in float32, giving a small negative sq and a
    # phi marginally above 1. Clamping keeps phi in its declared range [0, 1].
    sq = sq.clamp_min(0.0)
    return torch.exp(-sq / (2.0 * sigma.square()))


@dataclass
class Field:
    """The germline expression map.

    SPEC: 02 §2.2.1

    `mu` in `R^{N_G x m_dim}`, renormalised to the unit sphere after every
    germline update; `log_sigma` in `R^{N_G}`. Both are germline: gradients are
    prohibited, and `ES_STEP` is the only writer.
    """

    mu: torch.Tensor
    log_sigma: torch.Tensor
    scale: GermlineScale = GERMLINE_SCALE

    def __post_init__(self) -> None:
        check_shape(self.mu, [self.scale.N_G, self.scale.m_dim], name="mu_g")
        check_shape(self.log_sigma, [self.scale.N_G], name="log_sigma_g")
        if self.mu.requires_grad or self.log_sigma.requires_grad:
            raise ValueError(
                "Field tensors are germline; gradients are prohibited (§02.1)"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def initialise(
        cls,
        scale: GermlineScale = GERMLINE_SCALE,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sigma_g: float = SIGMA_G_INIT,
    ) -> "Field":
        """§02.2.1: `mu_g` uniform on `S^{m_dim-1}`, `log_sigma_g = log(1.2)`.

        `sigma_g = 1.2` gives a measured `median[phi] = 0.4995` at `m_dim = 8`.
        Spec v2 used `0.6` with the comment "so a typical phi is near 0.5"; the
        measured median there is **0.062** and `N_eff` falls to 0.79, below
        §01.5's own abort guard (audit A-14).
        """
        mu = sample_sphere(
            scale.N_G, scale.m_dim, generator, device=device, dtype=dtype
        )
        log_sigma = torch.full(
            (scale.N_G,), math.log(sigma_g), device=torch.device(device), dtype=dtype
        )
        return cls(mu, log_sigma, scale)

    def register(self, registry: GermlineRegistry, prefix: str = "") -> None:
        registry.register(f"{prefix}mu_g", self.mu)
        registry.register(f"{prefix}log_sigma_g", self.log_sigma)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def phi(self, p: torch.Tensor) -> torch.Tensor:
        """(1.1): the full expression profile of positions `p`."""
        return phi_from(p, self.mu, self.sigma)

    def phi_at(self, p: torch.Tensor, genes: torch.Tensor) -> torch.Tensor:
        """`phi_g(p_u)` restricted to each unit's own `k_g` genes.

        `p` is `(U, m_dim)`, `genes` is `(U, k_g)` of indices. Returns
        `(U, k_g)`.

        This is the form the Site actually needs: `gate(u, c)` is supported on
        `g_u`, so computing the full `(U, N_G)` profile and then gathering `k_g`
        of it wastes `N_G / k_g = 256x` the work. The gather-first form costs
        `O(U k_g m_dim)`.
        """
        check_shape(p, [None, self.scale.m_dim], name="p")
        check_shape(genes, [p.shape[0], self.scale.k_g], name="g_u")
        mu_sel = self.mu[genes]                       # (U, k_g, m_dim)
        sigma_sel = self.sigma[genes]                 # (U, k_g)
        sq = 2.0 - 2.0 * (mu_sel * p.unsqueeze(1)).sum(-1)
        return torch.exp(-sq.clamp_min(0.0) / (2.0 * sigma_sel.square()))

    # ------------------------------------------------------------------
    # Gene adjacency (§02.2.2)
    # ------------------------------------------------------------------

    def neighbour_table(self) -> torch.Tensor:
        """The `n_nbr` nearest genes to each gene, excluding itself.

        SPEC: 02 §2.2.2

        Shape `(N_G, n_nbr)`. Recomputed whenever `mu` changes (i.e. once per
        germline generation), not per mutation: at `N_G = 2048, m_dim = 8` the
        pairwise distance matrix is 16 MB and the top-k is milliseconds, whereas
        a per-mutation search would be inside the innermost selection loop.

        **[RATIONALE — what geometric locality does and does not buy]** §02.2.2
        v3 narrows the claim: `neighbour(g)` shares only the RBF *centre*, while
        the content `(a_g, b_g, kappa_g)` of the neighbouring gene is an
        independent random draw, so the swap replaces one rank-one part with an
        unrelated one. What locality buys is that the mutated unit's *expression
        profile* moves little, so its routing and metabolic cost are
        approximately preserved while its function changes.
        """
        d = torch.cdist(self.mu, self.mu)
        d.fill_diagonal_(float("inf"))
        _, idx = torch.topk(d, k=self.scale.n_nbr, dim=-1, largest=False)
        return idx

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def renormalise(self) -> None:
        """Project `mu` rows back to the unit sphere (§02.2.1).

        Called after every germline update. The ES perturbation is additive in
        the ambient space, so without this the centres drift off the sphere and
        (1.1)'s `||p - mu||^2 = 2 - 2<p,mu>` identity — which `phi_from` relies
        on — stops holding.
        """
        norms = self.mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.mu.div_(norms)

    def moments(
        self,
        n_probe: int = 400_000,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        """Measured `E[phi]`, `E[phi^2]`, `median[phi]` over uniform positions.

        SPEC: 01 §1.2

        §01.2 records these as `[EMPIRICAL]` with their sample size, because the
        v2 value of `sigma_g` was chosen from an assertion about them that was
        false by a factor of eight. Recomputing rather than trusting the tabled
        figure is the point.
        """
        device = self.mu.device
        chunk = 4096
        total = 0.0
        total_sq = 0.0
        n_done = 0
        sample: list[torch.Tensor] = []
        while n_done < n_probe:
            k = min(chunk, n_probe - n_done)
            p = sample_sphere(
                k, self.scale.m_dim, generator, device=device, dtype=self.mu.dtype
            )
            ph = self.phi(p)
            total += float(ph.sum())
            total_sq += float(ph.square().sum())
            if len(sample) < 8:
                # **[NORMATIVE — v3.1, audit A-85]** A flat stride of
                # `numel // 4096` over a `(chunk, N_G)` tensor is exactly `N_G`
                # when `chunk == 4096`, so it selected **column 0 of every
                # row** — the median of gene 0 alone, over 1 of 2048 genes.
                #
                # It is exact only while `log_sigma` is uniform, i.e. only at
                # generation 0, before ES has touched the field: the diagnostic
                # became wrong precisely when it started carrying information.
                # `median_phi` is the `[EMPIRICAL]` statistic §01.2 used to set
                # `SIGMA_G_INIT`, so it has to be a median over `(probe, gene)`
                # pairs. Sampling a random subset of both axes keeps the cost
                # bounded without collapsing either one.
                flat = ph.flatten()
                take = min(flat.numel(), 4096)
                idx = torch.randint(
                    0, flat.numel(), (take,), generator=generator,
                    device=flat.device,
                )
                sample.append(flat[idx].cpu())
            n_done += k
        n_tot = n_done * self.scale.N_G
        med = float(torch.cat(sample).median()) if sample else float("nan")
        return {
            "E_phi": total / n_tot,
            "E_phi2": total_sq / n_tot,
            "median_phi": med,
            # Gene 0's own width, kept as a named single-gene value rather than
            # reported as if it were the field's (audit A-85).
            "sigma_g_gene0": float(self.sigma[0]),
            "sigma_g_median": float(self.sigma.median()),
            "n_samples": n_done,
            "sigma_g": float(self.sigma[0]),
            "m_dim": self.scale.m_dim,
            "N_G": self.scale.N_G,
        }

    def n_eff(self, p: torch.Tensor, genes: torch.Tensor) -> float:
        """`N_eff = k_g * E[phi]`, a reported diagnostic only (§01.5).

        §01.5 v3 removes `N_eff` from the initialisation formula entirely — the
        v2 derivation used it as a variance prefactor where `k_g` is correct,
        making the realised `d_in Var(W)` `3.94 * gain_act`, outside §02.5's own
        acceptance band (audit A-13). It survives as a diagnostic with the abort
        guard `N_eff >= 1`, which detects a degenerate expression field.
        """
        return float(self.phi_at(p, genes).sum(dim=-1).mean())

    def to(self, device: torch.device | str) -> "Field":
        return Field(self.mu.to(device), self.log_sigma.to(device), self.scale)

    def clone(self) -> "Field":
        return Field(self.mu.clone(), self.log_sigma.clone(), self.scale)
