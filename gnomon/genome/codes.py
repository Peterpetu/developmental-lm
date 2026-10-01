"""Innate identity codes.

SPEC: 02 §2.1
SPEC: 02 §2.2
SPEC: 02 §2.3
SPEC: 02 §2.4
SPEC: 02 §2.5
SPEC: 02 §2.6
SPEC: 02 §2.7
SPEC: 02 §2.9

Nothing in this module is stored or learned. Every code is a pure function of an
integer index and a `GenomeShape`, and contributes zero to `|Gamma|` and zero to
`|Z|` -- which is the reason the genome grows as `H log N` rather than `O(N)`,
and the reason the PNAS paper distinguishes its approach from hypernetworks.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from ..config import GenomeShape, N_ROLES

__all__ = [
    "bits_to_pm1", "seeded_bijection", "channel_codes", "token_codes",
    "frequency_rank",
    "layer_codes", "role_codes", "column_codes", "vector_column",
    "bit_balance", "min_hamming", "STRUCTURED_BALANCE_MAX",
]


def bits_to_pm1(idx: torch.Tensor, n_bits: int) -> torch.Tensor:
    """`(N,)` int64 indices -> `(N, n_bits)` in `{-1., +1.}`.

    SPEC: 02 §2.1 -- `{-1,+1}` rather than `{0,1}`: a zero bit contributes
    nothing to `W b`, so a `{0,1}` code would make low-index identities
    systematically smaller in norm than high-index ones, which is an artifact of
    the encoding rather than of the identity it encodes.
    """
    if idx.numel() and int(idx.max()) >= (1 << n_bits):
        raise ValueError(
            f"index {int(idx.max())} needs more than {n_bits} bits; "
            f"grow the code width rather than truncating (SPEC: 02 §2.2)"
        )
    shifts = torch.arange(n_bits, dtype=torch.int64, device=idx.device)
    bits = (idx.unsqueeze(-1) >> shifts) & 1
    return bits.to(torch.float32) * 2.0 - 1.0


@functools.lru_cache(maxsize=32)
def seeded_bijection(n_bits: int, seed: int) -> torch.Tensor:
    """A fixed permutation of `{0 .. 2^n_bits - 1}`.

    SPEC: 02 §2.4. This is the construction that gives nesting, spread and
    determinism at once, and §02.3 records why the three obvious alternatives
    (prefix, stride, bit-reversal) all fail: each is a *linear* map from
    `log2 d` varying input bits into `n_bits` output bits, so its image lies in
    a `log2 d`-dimensional affine subcube and the remaining bit positions are
    constant. A genome would then never see those bits varied during evolution
    and they would be arbitrary noise at a wider decode.

    Cached because it is a fixed coordinate system, shared by every genome in
    every generation. It is not part of `Gamma` and is never mutated.
    """
    g = torch.Generator(device="cpu").manual_seed(seed ^ (n_bits * 0x9E3779B1))
    return torch.randperm(1 << n_bits, generator=g, dtype=torch.int64)


def _prefix_codes(n: int, n_bits: int, seed: int, device) -> torch.Tensor:
    if n > (1 << n_bits):
        raise ValueError(
            f"cannot address {n} identities with {n_bits} bits; n_bits must grow "
            f"to {max(1, (n - 1).bit_length())} (SPEC: 02 §2.2)"
        )
    perm = seeded_bijection(n_bits, seed)[:n]
    return bits_to_pm1(perm.to(device), n_bits)


def channel_codes(d_model: int, shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(d_model, n_c)` in `{-1,+1}`. SPEC: 02 §2.4."""
    return _prefix_codes(d_model, shape.n_c, shape.seed_codes, device)


def token_codes(vocab: int, shape: GenomeShape, device="cpu",
                rank: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`(vocab, n_v)` in `{-1,+1}`. SPEC: 02 §2.4, §2.9.

    Two constructions, and which one is right was settled by measurement.

    Without `rank`: a seeded bijection of the token id (§2.4), the same
    construction the channel axis uses.

    With `rank` -- a permutation ordering tokens by descending corpus frequency
    -- the code is a **Gray code of relative frequency rank**, exactly §2.5's
    construction with frequency in place of depth:

        q_v  = round( rank(v) * (2^n_v - 1) / (V - 1) )
        e_v  = gray(q_v)

    **Why this is the correct reading of §2.1's own rule.** That rule says Gray
    code where the index has a topology and plain binary where it does not, and
    §2.1 placed the token axis in the second class on the grounds that "token id
    order is an artifact of BPE merge order". BPE merges the most frequent pair
    first, so merge order *is* frequency order -- which is not an artifact but
    information, and frequency rank has exactly the topology the rule asks
    about. Measured on `data/ts_4096`: Spearman(id, frequency rank) = +0.292
    (p ~ 1e-81), and a linear probe recovers `R^2 = 0.035` of log-frequency from
    `log(id)` against `R^2 = 0.004` -- chance -- from §2.4's bijection.

    It matters because §10.7 measured the one component of the decoded
    plasticity rule that pays: a per-token step scaled as `freq^-0.25` beats a
    uniform one by 0.062 nats, 95% BCa [+0.017, +0.210]. A genome cannot decode
    a frequency-shaped field from a code that carries no frequency information,
    so §2.4's bijection was destroying the signal the only working mechanism
    needed.

    Relative rank rather than absolute also *improves* transfer, for §2.5's
    reason: rank 0 means "the most frequent token" at every vocabulary size, so
    a genome evolved at `V = 4096` decodes the same relationship at 8192.
    """
    if rank is None:
        return _prefix_codes(vocab, shape.n_v, shape.seed_codes ^ 0x5EED, device)
    if rank.numel() != vocab:
        raise ValueError(f"rank has {rank.numel()} entries, vocab is {vocab}")
    hi = (1 << shape.n_v) - 1
    r = rank.to(torch.float64)
    q = torch.round(r * hi / max(vocab - 1, 1)).to(torch.int64)
    return bits_to_pm1((q ^ (q >> 1)).to(device), shape.n_v)


def frequency_rank(counts: "torch.Tensor | np.ndarray") -> torch.Tensor:
    """`rank[v] = 0` for the most frequent token. SPEC: 02 §2.9.

    Derived from the training corpus, which is exactly as legitimate as the
    tokenizer itself being fitted on it: this is an *innate* coordinate system
    shaped by the environment the lineage evolved in, fixed before any lifetime
    and costing zero genome reals. It is recorded in the run manifest and
    covered by the corpus hash, because a code that depends on the corpus must
    be reproducible from it.

    **[NORMATIVE — ties are broken by token id, and they are not rare.]**
    Measured on `data/ts_4096`: 508 of 4,096 types share a count with at least
    one other type, the largest tie group being the 173 types that never occur
    in the shard at all. `torch.argsort` without `stable=True` leaves the order
    within a tie unspecified, and on exactly this input its output **differs**
    from the stable one -- so the first version of this function returned a code
    that depended on sort internals rather than on the corpus, which §2.9's
    reproducibility requirement and §2.8's T6 both forbid. The order is
    therefore fixed twice over: a stable sort, applied to a key that already
    orders tied types by ascending id, so the result is a pure function of
    `counts` on any device and any version.
    """
    c = torch.as_tensor(counts, dtype=torch.float64).flatten()
    # Descending count, ties broken by ascending id. `stable=True` alone would
    # do it for a descending sort of the counts themselves only if torch's
    # stability held under `descending`, which is a promise about the argument
    # order rather than about ties; negating the key makes it an ascending sort
    # where stability is the documented behaviour, and leaves nothing implicit.
    order = torch.argsort(-c, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(c.numel(), device=order.device)
    return rank


def layer_codes(L: int, shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(L, n_L)` in `{-1,+1}`: Gray code over **relative** depth.

    SPEC: 02 §2.5. Depth is the one axis with a topology, so it is the one axis
    that takes a Gray code: adjacent depths differ in a single bit, and a
    decoder smooth in its input is then smooth in depth.

    Relative rather than absolute depth is what makes cross-`L` transfer
    interpolation rather than extrapolation: layer 0 of an `L=4` phenotype and
    layer 0 of an `L=12` phenotype receive the same code, as do the last of
    each, so the genome learns "what an early layer looks like" and never "what
    layer 3 looks like".
    """
    if L > (1 << shape.n_L):
        raise ValueError(f"L={L} exceeds 2^n_L={1 << shape.n_L} (SPEC: 02 §2.2)")
    hi = (1 << shape.n_L) - 1
    if L == 1:
        q = torch.zeros(1, dtype=torch.int64, device=device)
    else:
        l = torch.arange(L, dtype=torch.float64, device=device)
        q = torch.round(l * hi / (L - 1)).to(torch.int64)
    gray = q ^ (q >> 1)
    return bits_to_pm1(gray, shape.n_L)


def role_codes(shape: GenomeShape, device="cpu") -> torch.Tensor:
    """`(16, n_M)` in `{-1,+1}`: plain binary. SPEC: 02 §2.6."""
    idx = torch.arange(N_ROLES, dtype=torch.int64, device=device)
    return bits_to_pm1(idx, shape.n_M)


def column_codes(b: torch.Tensor) -> torch.Tensor:
    """Append the "real column" flag: `(n, n_c) -> (n, n_c + 1)`. SPEC: 02 §2.6."""
    flag = torch.ones(b.shape[0], 1, dtype=b.dtype, device=b.device)
    return torch.cat([b, flag], dim=-1)


def vector_column(shape: GenomeShape, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """`(1, n_c + 1)`: the "no column" argument for vector decodes.

    SPEC: 02 §2.6. The `n_c` code bits are zero -- genuinely unused rather than
    ambiguously used -- and the flag bit is `-1`, so the vector case is
    unambiguously distinguishable from every real channel, whose flag is `+1`.
    """
    out = torch.zeros(1, shape.n_c + 1, dtype=dtype, device=device)
    out[0, -1] = -1.0
    return out


# SPEC: 02 §2.8 T4. The spread test applies to §2.4's *random* construction. A
# frequency-ordered code (§2.9) is deliberately structured: its bits correlate
# with rank, which is the entire point, so it is balanced only to within a
# factor that a random code would fail. The looser bound is stated here rather
# than hidden in a test.
STRUCTURED_BALANCE_MAX = 0.40


def bit_balance(codes: torch.Tensor) -> torch.Tensor:
    """Mean of each bit position over the image. SPEC: 02 §2.8 T4.

    A prefix or stride coding shows `|mean| == 1` at the constant positions;
    §2.4's construction gives values within `3/sqrt(n)` of zero.
    """
    return codes.mean(dim=0)


def min_hamming(codes: torch.Tensor, max_n: int = 512) -> int:
    """Minimum pairwise Hamming distance over the image. SPEC: 02 §2.8 T4.

    Capped at `max_n` identities because the pairwise matrix is `O(n^2 * n_bits)`
    and the statistic is a spread check, not an exhaustive certificate.
    """
    c = codes[:max_n]
    diff = (c.unsqueeze(1) != c.unsqueeze(0)).sum(-1)
    n = c.shape[0]
    diff = diff + torch.eye(n, dtype=diff.dtype, device=diff.device) * (c.shape[1] + 1)
    return int(diff.min())
