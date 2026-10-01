"""Data: shards, common-random-number pools, the fixed evaluation set, strata.

SPEC: 02 §2.3
SPEC: 07 §7.2
SPEC: 07 §7.4
SPEC: 08 §8.2

The shards and the tokenizer are GNOMON's (`data/ts_8192`, vocabulary 8192);
the window pool is SCREEN's `WindowPool`, whose contract is the one every
paired measurement rests on: arm `i` seed `s` and arm `j` seed `s` consume the
same windows in the same order.

Strata (§07.4): every target token of the fixed evaluation set is assigned to
exactly one of 12 strata, position-in-document bucket × target-frequency
class, label-free and computed once from the shard's document index and the
training-shard token counts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Optional

import numpy as np
import torch

from gnomon.data.stream import Shard, counts_digest, token_counts
from gnomon.genome.codes import frequency_rank
from screen.crn import WindowPool

__all__ = ["Corpus", "EvalSet", "POS_EDGES", "FREQ_EDGES", "N_STRATA", "stratum_names"]

POS_EDGES = (32, 128, 256)
"""SPEC: 07 §7.4 -- position-in-document buckets [0,32), [32,128), [128,256), [256,inf).

Measured on `data/ts_8192`: no TinyStories document reaches position 512, so an
upper edge of 512 left the last bucket empty; 256 populates all four."""
FREQ_EDGES = (100, 1000)
"""SPEC: 07 §7.4 -- target frequency-rank classes [0,100), [100,1000), [1000,V)."""
N_STRATA = (len(POS_EDGES) + 1) * (len(FREQ_EDGES) + 1)


def stratum_names() -> list[str]:
    pos = ["pos<32", "pos32-128", "pos128-256", "pos>=256"]
    frq = ["freq<100", "freq100-1000", "freq>=1000"]
    return [f"{p}|{f}" for p in pos for f in frq]


class Corpus:
    """The three splits of one prepared corpus plus the token statistics."""

    def __init__(self, root: str | pathlib.Path, vocab: int, *,
                 cache_dir: str | pathlib.Path = "runs/anlage/cache"):
        self.root = pathlib.Path(root)
        self.vocab = vocab
        self.train = Shard.load(self.root, "train")
        self.fitness = Shard.load(self.root, "fitness")
        self.test = Shard.load(self.root, "test")
        cache = pathlib.Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        cpath = cache / f"counts_{self.root.name}_{vocab}.npy"
        if cpath.exists():
            self.counts = np.load(cpath)
        else:
            self.counts = token_counts(self.train, vocab)
            np.save(cpath, self.counts)
        self.counts_digest = counts_digest(self.counts)
        self.rank = frequency_rank(torch.tensor(self.counts))       # (V,) int64
        self.n_unseen = int((self.counts <= 1.0).sum())

    def pool(self, *, n_windows: int, T: int, pool_seed: int, device="cuda") -> WindowPool:
        return WindowPool(self.train, n_windows=n_windows, T=T, pool_seed=pool_seed,
                          device=device)

    def manifest(self) -> dict:
        return {"root": str(self.root), "vocab": self.vocab,
                "tokenizer_hash": self.train.meta.get("tokenizer_hash"),
                "corpus_hash": self.train.meta.get("corpus_hash"),
                "counts_digest": self.counts_digest, "n_unseen_types": self.n_unseen,
                "n_train_tokens": self.train.n_tokens}


@dataclasses.dataclass
class EvalSet:
    """A fixed evaluation set with per-target strata. SPEC: 07 §7.4, 08 §8.2.

    `windows (n, T+1)`, `strata (n, T)` aligned with the targets `windows[:, 1:]`.
    """

    windows: torch.Tensor
    strata: torch.Tensor
    split: str
    digest: str
    offsets: Optional[np.ndarray] = None       # token offset of each window in its shard (v2, v3)
    sources: Optional[np.ndarray] = None       # index of the shard each window comes from (v3)

    @staticmethod
    def _from_offsets(shard: Shard, rank: torch.Tensor, offs: np.ndarray, T: int, split: str,
                      device: str) -> "EvalSet":
        tokens = np.asarray(shard.tokens)
        idx = offs[:, None] + np.arange(T + 1, dtype=np.int64)[None, :]
        win = tokens[idx].astype(np.int64)
        # position in document of every *target* token
        starts = np.asarray(shard.starts)
        tgt_flat = offs[:, None] + np.arange(1, T + 1)[None, :]
        doc = np.searchsorted(starts, tgt_flat, side="right") - 1
        pos = tgt_flat - starts[doc]
        pos_b = np.digitize(pos, POS_EDGES)                          # 0..3
        rk = rank.numpy()[win[:, 1:]]
        frq_b = np.digitize(rk, FREQ_EDGES)                          # 0..2
        strata = pos_b * (len(FREQ_EDGES) + 1) + frq_b
        digest = hashlib.blake2b(win.tobytes(), digest_size=16).hexdigest()
        return EvalSet(torch.from_numpy(win).to(device),
                       torch.from_numpy(strata.astype(np.int64)).to(device), split, digest,
                       offsets=offs.astype(np.int64))

    @staticmethod
    def sample_documents(shards: list, rank: torch.Tensor, *, n_windows: int, T: int, split: str, seed: int,
                         mod: int, residue: int, device="cuda") -> "EvalSet":
        """v3 (SPEC: 08 §8.2): windows at offsets uniform over the union of `shards`' tokens,
        kept when the window's starting document -- indexed globally across the shards in
        order -- satisfies `doc % mod == residue`. Sets with different residues share no
        document, and every set has the union's composition in expectation, which the two
        halves of the validation data do not have on their own (RESULTS.md, R6)."""
        if not 0 <= residue < mod:
            raise ValueError("residue must be in [0, mod)")
        rng = np.random.default_rng(int(seed))
        totals = np.array([s_.n_tokens - T - 1 for s_ in shards], dtype=np.float64)
        if (totals <= 0).any():
            raise ValueError("a shard is too small for T")
        weights = totals / totals.sum()
        bases = np.cumsum([0] + [len(s_.starts) for s_ in shards[:-1]])
        n_cand = 16 * n_windows
        srcs, offs = [], []
        for k, s_ in enumerate(shards):
            m = int(round(n_cand * weights[k]))
            off = rng.integers(0, int(totals[k]), size=m, dtype=np.int64)
            doc = np.searchsorted(np.asarray(s_.starts), off, side="right") - 1
            keep = ((bases[k] + doc) % mod) == residue
            srcs.append(np.full(int(keep.sum()), k, dtype=np.int64))
            offs.append(off[keep])
        srcs = np.concatenate(srcs); offs = np.concatenate(offs)
        if len(offs) < n_windows:
            raise ValueError(f"only {len(offs)} candidate windows with residue {residue}; need {n_windows}")
        order = rng.permutation(len(offs))[:n_windows]
        srcs, offs = srcs[order], offs[order]
        # deterministic layout: by shard, then offset
        o2 = np.lexsort((offs, srcs))
        srcs, offs = srcs[o2], offs[o2]
        parts = []
        for k, s_ in enumerate(shards):
            sel = offs[srcs == k]
            if len(sel):
                parts.append(EvalSet._from_offsets(s_, rank, sel, T, split, "cpu"))
        win = torch.cat([q.windows for q in parts], 0)
        strata = torch.cat([q.strata for q in parts], 0)
        digest = hashlib.blake2b(win.numpy().astype(np.int64).tobytes(), digest_size=16).hexdigest()
        return EvalSet(win.to(device), strata.to(device), split, digest, offsets=offs, sources=srcs)

    @staticmethod
    def build(shard: Shard, rank: torch.Tensor, *, n_windows: int, T: int, split: str,
              device="cuda", offset_windows: int = 0) -> "EvalSet":
        """v1: non-overlapping windows from the head of `shard` (after `offset_windows`).
        Kept for the R1-R4 artifacts; its class composition is that of the shard's head,
        not of the corpus (RESULTS.md, R3)."""
        need = (offset_windows + n_windows) * (T + 1)
        if shard.n_tokens < need:
            raise ValueError(f"shard has {shard.n_tokens} tokens, need {need}")
        offs = (offset_windows + np.arange(n_windows, dtype=np.int64)) * (T + 1)
        return EvalSet._from_offsets(shard, rank, offs, T, split, device)

    @staticmethod
    def sample(shard: Shard, rank: torch.Tensor, *, n_windows: int, T: int, split: str, seed: int,
               region: tuple[float, float] = (0.0, 1.0), device="cuda") -> "EvalSet":
        """v2 (SPEC: 08 §8.2): windows at offsets drawn uniformly over the token range
        `[lo, hi)` of `shard` (`region` in fractions), from `seed` alone -- the same rule the
        training pool uses, so the set's class composition is the corpus's in expectation
        and two sets from disjoint regions share no token."""
        lo = int(region[0] * shard.n_tokens)
        hi = int(region[1] * shard.n_tokens) - T - 1
        if hi - lo < n_windows:
            raise ValueError("region too small for the requested windows")
        rng = np.random.default_rng(int(seed))
        offs = np.sort(rng.integers(lo, hi, size=n_windows, dtype=np.int64))
        return EvalSet._from_offsets(shard, rank, offs, T, split, device)

    @property
    def n_windows(self) -> int:
        return int(self.windows.shape[0])

    def stratum_counts(self) -> torch.Tensor:
        return torch.bincount(self.strata.reshape(-1), minlength=N_STRATA)


# =============================================================================
# Data classes and the mixed pool. SPEC: 12 §12.2
# =============================================================================
import hashlib as _hashlib

RARE_RANK = 1000
"""SPEC: 12 §12.2 -- a token is 'rare' when its frequency rank is >= RARE_RANK."""
CLASS_NAMES_TS = tuple(f"len_{t}|rare_{h}" for t in ("short", "mid", "long") for h in ("lo", "hi"))


def classify_documents(shard: Shard, rank: torch.Tensor, *, rare_rank: int = RARE_RANK,
                       manifest: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """The TinyStories class partition (M1): length tercile × rarity half, per document.

    Edges are deterministic functions of the shard (quantiles with the 'lower' rule,
    the median with 'lower' on ties) and are returned in the manifest so that a
    second build can be checked against the first. With `manifest` given, its edges
    are *applied* (the fitness and test shards are classified by the training
    shard's edges, never by their own). Returns `(doc_class (n_docs,) int64, manifest)`.
    """
    lengths = np.asarray(shard.lengths, dtype=np.int64)
    n_docs = len(lengths)
    if n_docs < 6:
        raise ValueError("need at least 6 documents to form 6 classes")
    rank_np = rank.detach().cpu().numpy().astype(np.int64)
    rare = (rank_np[np.asarray(shard.tokens, dtype=np.int64)] >= rare_rank).astype(np.int64)
    starts = np.asarray(shard.starts, dtype=np.int64)
    rare_per_doc = np.add.reduceat(rare, starts) if n_docs > 0 else np.zeros(0)
    # reduceat sums to the next start; the last document runs to the end
    rarity = rare_per_doc / np.maximum(lengths, 1)
    if manifest is not None:
        if manifest.get("rare_rank") != rare_rank:
            raise ValueError("manifest rare_rank differs")
        e1, e2 = (float(x) for x in manifest["length_edges"])
        med = float(manifest["rarity_median"])
    else:
        e1, e2 = (float(np.quantile(lengths, q, method="lower")) for q in (1 / 3, 2 / 3))
        med = float(np.quantile(rarity, 0.5, method="lower"))
    tercile = np.searchsorted(np.array([e1, e2]), lengths, side="right")   # 0, 1, 2
    half = (rarity > med).astype(np.int64)
    cls = (tercile * 2 + half).astype(np.int64)
    counts = np.bincount(cls, minlength=6).tolist()
    out = {"rule": "length_tercile_x_rarity_half", "n_classes": 6, "names": list(CLASS_NAMES_TS),
           "length_edges": [e1, e2], "rarity_median": med, "rare_rank": rare_rank,
           "doc_counts": counts, "n_docs": int(n_docs)}
    return cls, out


def systematic_counts(B: int, mu: np.ndarray, u: float) -> np.ndarray:
    """`n_c = floor(B S_c + u) - floor(B S_{c-1} + u)`, `S` the cumulative mixture, `u in [0, 1)`:
    sums to `B` exactly and has mean `B mu_c` (M2)."""
    if B < 1:
        raise ValueError("B >= 1")
    if not 0.0 <= u < 1.0:
        raise ValueError("u in [0, 1)")
    S = np.concatenate([[0.0], np.cumsum(np.asarray(mu, dtype=np.float64))])
    S[-1] = 1.0                                               # guard rounding of the last cumulative sum
    marks = np.floor(B * S + u).astype(np.int64)
    n = np.diff(marks)
    assert int(n.sum()) == B, (n, B)
    return n


class ClassifiedPool:
    """A window pool whose windows carry a class, sampled at a mixture `mu`. SPEC: 12 §12.2.

    Windows are drawn exactly as `screen.crn.WindowPool` draws them (offsets from
    `pool_seed` alone), so the unclassified pool and this one hold the same windows
    (M4). The class of a window is the class of the document it starts in.
    """

    def __init__(self, shard: Shard, doc_class: np.ndarray, n_classes: int, *, n_windows: int, T: int,
                 pool_seed: int, device: str = "cuda"):
        if n_windows < 1 or T < 1:
            raise ValueError("n_windows and T must be >= 1")
        tokens = np.asarray(shard.tokens)
        n_tok = int(tokens.shape[0])
        if n_tok < T + 1:
            raise ValueError(f"shard has {n_tok} tokens, need >= {T + 1}")
        rng = np.random.default_rng(pool_seed)
        offs = rng.integers(0, n_tok - T - 1, size=n_windows, dtype=np.int64)
        idx = offs[:, None] + np.arange(T + 1, dtype=np.int64)[None, :]
        win = tokens[idx].astype(np.int64)
        starts = np.asarray(shard.starts, dtype=np.int64)
        doc = np.searchsorted(starts, offs, side="right") - 1
        doc_class = np.asarray(doc_class, dtype=np.int64)
        if doc_class.shape[0] != len(starts):
            raise ValueError("one class per document")
        cls = doc_class[doc]
        if cls.min() < 0 or cls.max() >= n_classes:
            raise ValueError("class out of range")
        self.T, self.n_windows, self.pool_seed, self.device = T, n_windows, pool_seed, device
        self.n_classes = int(n_classes)
        self.windows = torch.from_numpy(win).to(device)
        self.window_class = torch.from_numpy(cls)
        self.members = [torch.nonzero(self.window_class == c).flatten() for c in range(n_classes)]
        self.counts = torch.tensor([int(m.numel()) for m in self.members])
        if int(self.counts.min()) == 0:
            raise ValueError(f"a class has no windows in the pool: counts {self.counts.tolist()}")
        self._digest = _hashlib.blake2b(np.ascontiguousarray(win).tobytes(), digest_size=16).hexdigest()
        self._class_digest = _hashlib.blake2b(np.ascontiguousarray(cls).tobytes(), digest_size=16).hexdigest()

    def digest(self) -> str:
        return self._digest

    def class_digest(self) -> str:
        return self._class_digest

    @property
    def corpus_mixture(self) -> torch.Tensor:
        """`mu^corpus_c = |c| / sum |c'|` over the pool's windows (M4)."""
        return (self.counts.double() / self.counts.sum()).contiguous()

    def at(self, mu) -> "MixedPoolView":
        """The pool at one mixture: `.batches(batch, steps, seed)` as `WindowPool` has."""
        return MixedPoolView(self, mu)

    def window_classes_of(self, windows: torch.Tensor) -> torch.Tensor:
        """Class of each given window `(n, T+1)` by matching its first token offset is not
        possible in general; this looks the windows up by identity in the pool (for tests)."""
        out = torch.empty(windows.shape[0], dtype=torch.long)
        for i in range(windows.shape[0]):
            j = torch.nonzero((self.windows == windows[i].to(self.windows.device)).all(1))
            out[i] = int(self.window_class[int(j[0])]) if j.numel() else -1
        return out

    def batches(self, *, batch: int, steps: int, seed: int, mu):
        """`steps` batches of `(batch, T+1)` at mixture `mu`, ordered by `seed` alone.

        One permutation per class, all seeded from `seed`; a batch takes the next
        `n_c` windows of class `c` with `n_c` from `systematic_counts`. Two calls with
        different `mu` and one `seed` consume identical class streams (M3).
        """
        if batch < 1 or steps < 1:
            raise ValueError("batch and steps must be >= 1")
        mu = np.asarray(torch.as_tensor(mu, dtype=torch.float64).cpu().numpy(), dtype=np.float64)
        if mu.shape != (self.n_classes,) or abs(mu.sum() - 1.0) > 1e-9 or (mu < 0).any():
            raise ValueError("mu must be a simplex vector over the classes")
        # the order seed may be a 64-bit derived seed: reduce it so `base * 1000 + c` fits
        # the generator's range (identical to the raw seed for every seed below 2^31)
        base = int(seed) % (1 << 31)
        gens = [torch.Generator(device="cpu").manual_seed(base * 1000 + c) for c in range(self.n_classes)]
        perms = [self.members[c][torch.randperm(int(self.counts[c]), generator=gens[c])] for c in range(self.n_classes)]
        cursors = [0] * self.n_classes
        epochs = [0] * self.n_classes
        u_rng = np.random.default_rng(base)
        for _ in range(steps):
            n = systematic_counts(batch, mu, float(u_rng.random()))
            parts = []
            for c in range(self.n_classes):
                need = int(n[c])
                while need > 0:
                    avail = int(self.counts[c]) - cursors[c]
                    take = min(need, avail)
                    if take > 0:
                        parts.append(perms[c][cursors[c]:cursors[c] + take])
                        cursors[c] += take
                        need -= take
                    if cursors[c] >= int(self.counts[c]):
                        epochs[c] += 1
                        g = torch.Generator(device="cpu").manual_seed(base * 1000 + c + 7919 * epochs[c])
                        perms[c] = self.members[c][torch.randperm(int(self.counts[c]), generator=g)]
                        cursors[c] = 0
            idx = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
            yield self.windows[idx.to(self.windows.device)]


class MixedPoolView:
    """`ClassifiedPool` bound to one mixture; the interface `lifetime.train` expects."""

    def __init__(self, pool: ClassifiedPool, mu):
        self.pool = pool
        self.mu = torch.as_tensor(mu, dtype=torch.float64).clone()
        self.T = pool.T
        self.n_windows = pool.n_windows

    def digest(self) -> str:
        return self.pool.digest()

    def batches(self, *, batch: int, steps: int, seed: int):
        return self.pool.batches(batch=batch, steps=steps, seed=seed, mu=self.mu)


def classify_eval_windows(ev: "EvalSet", shard, doc_class, *, offset_windows: int = 0) -> torch.Tensor:
    """Class of each window of an `EvalSet`: the class of the document the window starts
    in. v3 sets carry `sources`; then `shard` and `doc_class` are lists, one per source
    shard. v2 sets carry offsets; v1 sets are consecutive from the head."""
    n, T1 = ev.windows.shape
    if ev.sources is not None:
        out = np.empty(n, dtype=np.int64)
        for k, (sh, dc) in enumerate(zip(shard, doc_class)):
            m = ev.sources == k
            if m.any():
                doc = np.searchsorted(np.asarray(sh.starts, dtype=np.int64), ev.offsets[m], side="right") - 1
                out[m] = np.asarray(dc, dtype=np.int64)[doc]
        return torch.from_numpy(out)
    offs = ev.offsets if ev.offsets is not None else (offset_windows + np.arange(n)) * T1
    starts = np.asarray(shard.starts, dtype=np.int64)
    doc = np.searchsorted(starts, offs, side="right") - 1
    return torch.from_numpy(np.asarray(doc_class, dtype=np.int64)[doc])
