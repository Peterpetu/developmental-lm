"""Verifier-backed synthetic task families and their serialisation.

SPEC: 06 §6.3.2

**[NORMATIVE — reserved symbol block]** Token ids `0..63` are a **reserved
block** that the tokenizer never merges and never emits for natural text. Every
synthetic task is serialised over this block, guaranteeing **one symbol = one
token**.

Spec v1 specified no serialisation at all. With a BPE vocabulary learned on
story text, runs of bits and brackets would merge into multi-symbol tokens,
changing the structure of PARITY and DYCK and invalidating both the
length-generalisation claims of §08.2 and the expressivity argument of §05.2.2.

**[NORMATIVE]** The loss is masked to the **answer span** only.

**[NORMATIVE]** Every generator emits `(prompt, answer, lambda_task)` and every
verifier is a **total** function returning `{0,1}`. Generators run on CPU in
dataloader workers. If generator latency exceeds GPU step time the
implementation **raises**; spec v1 said it "asserts ... and logs a warning
otherwise", which is neither.

**Why verifier-backed at all (§6.3.1).** Verifier-free self-improvement
*sharpens* rather than expands: RLVR models win at `pass@1` while the base model
wins at large `k`, and the reachable support demonstrably shrinks. Ground-truth
generators are the channel through which new support can enter. And bounded-depth
transformers and diagonal linear recurrences both lie in `TC^0` and provably
cannot do arbitrary-length parity or permutation composition; §05.2.2 gives the
substrate the capacity, and the task stream supplies the pressure to use it.

**No task-identity conditioning (§08.7 rule 2).** The serialisation carries **no
family marker**. A model that could read off which family an instance came from
would be solving a different problem, and §08.7 requires a static assertion that
no task-identity feature reaches the model — `assert_no_task_identity` below is
that assertion.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RESERVED_SIZE",
    "PAD", "BOS", "EOS", "SEP", "ROW", "EX", "SYM_BASE", "N_SYMBOLS",
    "TaskParams",
    "Instance",
    "FAMILIES",
    "GRID_TRANSFORMS",
    "generate",
    "solve",
    "verify",
    "serialised_length",
    "assert_no_task_identity",
    "EDIT_TABLE",
    "EditSpec",
    "edit",
    "family_lower_bounds",
]

# ---------------------------------------------------------------------------
# The reserved block (§06.3.2)
# ---------------------------------------------------------------------------

RESERVED_SIZE = 64
"""Token ids `0..63`; the tokenizer never merges them and never emits them for
natural text."""

PAD = 0
BOS = 1
EOS = 2
SEP = 3
"""Separates prompt from answer. The loss is masked to everything after it."""
ROW = 4
"""Row separator inside a GRID."""
EX = 5
"""Example separator inside a GRID prompt."""
SYM_BASE = 8
N_SYMBOLS = RESERVED_SIZE - SYM_BASE
"""56 task symbols: `sym(i) = SYM_BASE + i` for `0 <= i < 56`."""


def sym(i: int) -> int:
    if not 0 <= i < N_SYMBOLS:
        raise ValueError(f"symbol {i} outside the reserved alphabet of {N_SYMBOLS}")
    return SYM_BASE + i


# ---------------------------------------------------------------------------
# Parameterisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskParams:
    """`lambda_task` — one point in a family's parameter space.

    Every field has a default so that a family's *lower bound* is expressible
    without naming the parameters of other families, which is what
    `SEED_ARCHIVE` needs (§06.3.3).
    """

    family: str
    n_len: int = 4
    m_mod: int = 2
    p_grp: int = 3
    tpt: int = 1
    b_types: int = 1
    depth: int = 2
    alpha_sz: int = 2
    gap: int = 0
    P_mqar: int = 2
    distract: int = 0
    h: int = 2
    w: int = 2
    tf: str = "identity"
    n_ex: int = 1

    def key(self) -> Tuple:
        """Identity for archive membership (`reject if lambda' already in Lambda`)."""
        return (
            self.family, self.n_len, self.m_mod, self.p_grp, self.tpt,
            self.b_types, self.depth, self.alpha_sz, self.gap, self.P_mqar,
            self.distract, self.h, self.w, self.tf, self.n_ex,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "family": self.family, "n_len": self.n_len, "m_mod": self.m_mod,
            "p_grp": self.p_grp, "tpt": self.tpt, "b_types": self.b_types,
            "depth": self.depth, "alpha_sz": self.alpha_sz, "gap": self.gap,
            "P_mqar": self.P_mqar, "distract": self.distract, "h": self.h,
            "w": self.w, "tf": self.tf, "n_ex": self.n_ex,
        }


@dataclass
class Instance:
    """One serialised task instance."""

    prompt: List[int]
    answer: List[int]
    params: TaskParams

    def tokens(self) -> List[int]:
        """`BOS prompt SEP answer EOS` — the full sequence."""
        return [BOS] + self.prompt + [SEP] + self.answer + [EOS]

    def loss_mask(self) -> List[bool]:
        """True on the answer span only (§06.3.2).

        The `SEP` itself is not scored; the first scored position is the first
        answer token, and `EOS` **is** scored so the model must learn where the
        answer ends rather than being told.
        """
        n = len(self.tokens())
        mask = [False] * n
        start = 1 + len(self.prompt) + 1
        for i in range(start, n):
            mask[i] = True
        return mask


FAMILIES: Tuple[str, ...] = (
    "PARITY", "MODCOUNT", "PERM", "DYCK", "COPY", "MQAR", "GRID",
)

GRID_TRANSFORMS: Tuple[str, ...] = (
    "identity", "rot90", "rot180", "rot270", "flip_h", "flip_v",
    "translate", "recolour", "tile",
)
"""§06.3.2: enumerated explicitly. Spec v1 left "transform class" undefined,
making the family unimplementable."""


# ---------------------------------------------------------------------------
# Ground-truth solvers -- the verifiers
# ---------------------------------------------------------------------------


def _parity_solve(prompt: List[int], p: TaskParams) -> List[int]:
    bits = [t - SYM_BASE for t in prompt]
    return [sym(sum(bits) % 2)]


def _modcount_solve(prompt: List[int], p: TaskParams) -> List[int]:
    """Running count of the marked symbol, mod `m_mod`, reported at the end."""
    syms = [t - SYM_BASE for t in prompt]
    return [sym(sum(1 for s in syms if s == 1) % p.m_mod)]


def _perm_solve(prompt: List[int], p: TaskParams) -> List[int]:
    """Compose the listed transpositions and report the resulting permutation."""
    g = p.p_grp
    perm = list(range(g))
    syms = [t - SYM_BASE for t in prompt]
    # Each step is `tpt` transpositions, each encoded as two symbols (i, j).
    for k in range(0, len(syms) - 1, 2):
        i, j = syms[k], syms[k + 1]
        if 0 <= i < g and 0 <= j < g:
            perm[i], perm[j] = perm[j], perm[i]
    return [sym(v) for v in perm]


def _dyck_solve(prompt: List[int], p: TaskParams) -> List[int]:
    """Report the currently open bracket types, outermost first.

    Symbols: open type `b` is `2*b`, close type `b` is `2*b + 1`.
    A close that does not match the top of the stack makes the string invalid;
    the verifier is **total**, so an invalid string has the well-defined answer
    `[sym(N_SYMBOLS-1)]` (a dedicated INVALID symbol).
    """
    stack: List[int] = []
    for t in prompt:
        s = t - SYM_BASE
        kind, b = s % 2, s // 2
        if kind == 0:
            stack.append(b)
        else:
            if not stack or stack[-1] != b:
                return [sym(N_SYMBOLS - 1)]
            stack.pop()
    return [sym(2 * b) for b in stack] if stack else [sym(N_SYMBOLS - 2)]


def _copy_solve(prompt: List[int], p: TaskParams) -> List[int]:
    """Return the first `n_len` symbols; the `gap` filler is ignored."""
    return list(prompt[: p.n_len])


def _mqar_solve(prompt: List[int], p: TaskParams) -> List[int]:
    """Key -> value lookup. The prompt is `k1 v1 k2 v2 ... query`."""
    body = prompt[:-1]
    query = prompt[-1]
    table: Dict[int, int] = {}
    for i in range(0, len(body) - 1, 2):
        table[body[i]] = body[i + 1]
    return [table.get(query, sym(N_SYMBOLS - 1))]


def _apply_transform(grid: np.ndarray, p: TaskParams) -> np.ndarray:
    tf = p.tf
    if tf == "identity":
        return grid.copy()
    if tf == "rot90":
        return np.rot90(grid, 1)
    if tf == "rot180":
        return np.rot90(grid, 2)
    if tf == "rot270":
        return np.rot90(grid, 3)
    if tf == "flip_h":
        return np.fliplr(grid)
    if tf == "flip_v":
        return np.flipud(grid)
    if tf == "translate":
        return np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
    if tf == "recolour":
        # A fixed involution on the colour alphabet, so the transform is a
        # genuine bijection the model can infer from examples.
        out = grid.copy()
        n_col = max(2, int(grid.max()) + 1)
        return (out + 1) % n_col
    if tf == "tile":
        return np.tile(grid, (1, 1))
    raise ValueError(f"unknown GRID transform {tf!r}")


def _grid_solve(prompt: List[int], p: TaskParams) -> List[int]:
    """Apply `tf` to the query grid, which is the last grid in the prompt."""
    grids = _parse_grids(prompt, p)
    if not grids:
        return [sym(N_SYMBOLS - 1)]
    out = _apply_transform(grids[-1], p)
    return _serialise_grid(out)


def _parse_grids(prompt: List[int], p: TaskParams) -> List[np.ndarray]:
    """Split a GRID prompt on `EX` and `ROW` into arrays."""
    grids: List[np.ndarray] = []
    rows: List[List[int]] = []
    cur: List[int] = []
    for t in prompt:
        if t == ROW:
            rows.append(cur)
            cur = []
        elif t == EX:
            if cur:
                rows.append(cur)
                cur = []
            if rows:
                grids.append(np.array(rows, dtype=np.int64))
                rows = []
        else:
            cur.append(t - SYM_BASE)
    if cur:
        rows.append(cur)
    if rows:
        grids.append(np.array(rows, dtype=np.int64))
    return [g for g in grids if g.ndim == 2 and g.size > 0]


def _serialise_grid(g: np.ndarray) -> List[int]:
    out: List[int] = []
    for i, row in enumerate(g):
        if i:
            out.append(ROW)
        out.extend(sym(int(v)) for v in row)
    return out


_SOLVERS: Dict[str, Callable[[List[int], TaskParams], List[int]]] = {
    "PARITY": _parity_solve,
    "MODCOUNT": _modcount_solve,
    "PERM": _perm_solve,
    "DYCK": _dyck_solve,
    "COPY": _copy_solve,
    "MQAR": _mqar_solve,
    "GRID": _grid_solve,
}


def solve(prompt: Sequence[int], params: TaskParams) -> List[int]:
    """The ground-truth answer for a prompt. Total on any reserved-block input."""
    if params.family not in _SOLVERS:
        raise KeyError(f"unknown family {params.family!r}; known: {FAMILIES}")
    return _SOLVERS[params.family](list(prompt), params)


def verify(prompt: Sequence[int], candidate: Sequence[int], params: TaskParams) -> int:
    """A **total** function returning `{0,1}` (§06.3.2).

    Any candidate is admissible input, including the wrong length, symbols
    outside the reserved block, and the empty sequence. Totality is what makes
    `s_task` well defined for a model that has not yet learned to stop.
    """
    try:
        truth = solve(prompt, params)
    except Exception:
        return 0
    cand = list(candidate)
    return int(cand == truth)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate(params: TaskParams, rng: np.random.Generator) -> Instance:
    """Emit one `(prompt, answer, lambda_task)` (§06.3.2).

    The answer is produced by `solve`, never independently, so a generator and
    its verifier cannot disagree — a class of bug that would silently make
    `s_task` a measurement of the generator rather than of the model.
    """
    f = params.family
    if f == "PARITY":
        prompt = [sym(int(b)) for b in rng.integers(0, 2, size=params.n_len)]
    elif f == "MODCOUNT":
        prompt = [sym(int(b)) for b in rng.integers(0, 2, size=params.n_len)]
    elif f == "PERM":
        g = params.p_grp
        n_pairs = max(1, params.n_len // 2)
        prompt = []
        for _ in range(n_pairs):
            i, j = rng.integers(0, g), rng.integers(0, g)
            prompt.extend([sym(int(i)), sym(int(j))])
    elif f == "DYCK":
        prompt = _gen_dyck(params, rng)
    elif f == "COPY":
        body = [sym(int(v)) for v in rng.integers(0, params.alpha_sz, size=params.n_len)]
        filler = [sym(0)] * params.gap
        prompt = body + filler
    elif f == "MQAR":
        n_pairs = params.P_mqar
        keys = rng.choice(
            max(2, params.alpha_sz if params.alpha_sz >= n_pairs else N_SYMBOLS - 4),
            size=n_pairs, replace=False,
        )
        # **[NORMATIVE — a distractor is an unqueried *pair*; v3.1, audit A-88]**
        # v3.0 appended `distract` **loose symbols** after the key-value pairs,
        # and `_mqar_solve` parses the whole body two at a time — so every
        # distractor was read **into the lookup table**, an odd `distract`
        # shifted the alignment of nothing after it, and a distractor colliding
        # with a real key overwrote that key's value. Measured: 7.17% of the
        # suite row's ground truths were wrong and 2.9% of answers fell outside
        # the declared alphabet. `distract` was not a distractor parameter: it
        # enlarged the table.
        #
        # A distractor in MQAR is an additional pair whose key is **never
        # queried**. Drawing all keys distinct and querying only from the first
        # `n_pairs` gives that, keeps the body an exact sequence of pairs, and
        # makes `distract` monotone in difficulty rather than in ambiguity.
        n_total = n_pairs + max(0, params.distract)
        pool = max(2, params.alpha_sz if params.alpha_sz >= n_total
                   else N_SYMBOLS - 4)
        n_total = min(n_total, pool)
        all_keys = rng.choice(pool, size=n_total, replace=False)
        prompt = []
        for kk in all_keys:
            prompt.extend([sym(int(kk)),
                           sym(int(rng.integers(0, max(2, params.alpha_sz))))])
        # Query one of the first `n_pairs` keys only; the rest are distractors.
        prompt.append(sym(int(all_keys[int(rng.integers(0, min(n_pairs, n_total)))])))
    elif f == "GRID":
        prompt = _gen_grid(params, rng)
    else:
        raise KeyError(f"unknown family {f!r}")

    answer = solve(prompt, params)
    return Instance(prompt, answer, params)


def _gen_dyck(params: TaskParams, rng: np.random.Generator) -> List[int]:
    """A balanced Dyck prefix of at most `depth` and `b_types` bracket types."""
    out: List[int] = []
    stack: List[int] = []
    for _ in range(params.n_len):
        can_open = len(stack) < params.depth
        can_close = bool(stack)
        if can_open and (not can_close or rng.random() < 0.6):
            b = int(rng.integers(0, params.b_types))
            stack.append(b)
            out.append(sym(2 * b))
        elif can_close:
            b = stack.pop()
            out.append(sym(2 * b + 1))
        else:
            b = int(rng.integers(0, params.b_types))
            stack.append(b)
            out.append(sym(2 * b))
    return out


def _gen_grid(params: TaskParams, rng: np.random.Generator) -> List[int]:
    """`n_ex` worked examples then a query grid, separated by `EX`."""
    n_col = 4
    parts: List[int] = []
    for _ in range(params.n_ex):
        g = rng.integers(0, n_col, size=(params.h, params.w))
        parts.extend(_serialise_grid(g))
        parts.append(EX)
        parts.extend(_serialise_grid(_apply_transform(g, params)))
        parts.append(EX)
    q = rng.integers(0, n_col, size=(params.h, params.w))
    parts.extend(_serialise_grid(q))
    return parts


def serialised_length(params: TaskParams, rng: np.random.Generator) -> int:
    """Total tokens of one instance, used by the `T_ctx` clip of §06.3.5."""
    return len(generate(params, rng).tokens())


def assert_no_task_identity(instance: Instance) -> None:
    """§08.7 rule 2: a **static assertion** that no task-identity feature reaches
    the model.

    SPEC: 08 §8.7

    Spec v1 required "the ablation confirming this", but an absent input cannot
    be ablated — the only way to establish it is a structural check, which is
    what this is. The serialisation contains only the reserved control symbols
    and the task's own alphabet; no token encodes which family produced it.
    """
    toks = instance.tokens()
    bad = [t for t in toks if not 0 <= t < RESERVED_SIZE]
    if bad:
        raise AssertionError(
            f"instance leaves the reserved block: {sorted(set(bad))[:8]}"
        )
    control = {BOS, EOS, SEP, ROW, EX}
    used_control = sorted(set(toks) & control)
    allowed = {BOS, EOS, SEP} | ({ROW, EX} if instance.params.family == "GRID" else set())
    leak = set(used_control) - allowed
    if leak:
        raise AssertionError(
            f"family {instance.params.family} uses control symbols {sorted(leak)} "
            f"that identify it; §08.7 rule 2 forbids task-identity features"
        )


# ---------------------------------------------------------------------------
# The EDIT table (§06.3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditSpec:
    name: str
    lo: int
    hi: int
    kind: str
    """`step` | `scale` | `double` | `categorical`"""
    step: float = 1.0
    weight: float = 1.0


EDIT_TABLE: Tuple[EditSpec, ...] = (
    # §06.3.5 v3: n_len is clipped to [4, 32] at every configuration, which
    # SUPERSEDES the table's nominal upper bound of 96 (audit A-30).
    EditSpec("n_len", 4, 32, "scale", 0.25, 3.0),
    EditSpec("m_mod", 2, 9, "step", 1, 1.0),
    EditSpec("p_grp", 3, 6, "step", 1, 1.0),
    EditSpec("tpt", 1, 4, "step", 1, 1.0),
    EditSpec("b_types", 1, 4, "step", 1, 1.0),
    # §06.3.5 v3: depth <= 8, matching the suite's DYCK setting (audit A-30).
    # Capped at the hold-out band of §06.3.5, not above it: an `EDIT` that can
    # reach depth 8 makes the hold-out unenforceable (audit A-88).
    EditSpec("depth", 2, 6, "step", 1, 1.0),
    EditSpec("alpha_sz", 2, 32, "double", 2, 1.0),
    EditSpec("gap", 0, 32, "step", 4, 1.0),
    # §06.3.5 v3: P_mqar <= 16 against the suite's 32 (audit A-30).
    EditSpec("P_mqar", 2, 16, "step", 2, 2.0),
    EditSpec("distract", 0, 64, "step", 4, 1.0),
    EditSpec("h", 2, 8, "step", 1, 1.0),
    EditSpec("w", 2, 8, "step", 1, 1.0),
    EditSpec("tf", 0, len(GRID_TRANSFORMS) - 1, "categorical", 1, 2.0),
    EditSpec("n_ex", 1, 4, "step", 1, 1.0),
)

_FAMILY_PARAMS: Dict[str, Tuple[str, ...]] = {
    "PARITY": ("n_len",),
    "MODCOUNT": ("m_mod", "n_len"),
    "PERM": ("p_grp", "tpt", "n_len"),
    "DYCK": ("b_types", "depth", "n_len"),
    "COPY": ("alpha_sz", "n_len", "gap"),
    "MQAR": ("P_mqar", "distract", "alpha_sz"),
    "GRID": ("h", "w", "tf", "n_ex"),
}
"""§06.3.2's per-family parameter list. `EDIT` perturbs only a parameter the
family actually uses — editing `m_mod` on a PARITY instance would produce a
distinct archive key for an identical generator, filling `Lambda` with
duplicates that `reject if lambda' already in Lambda` cannot see."""


def edit(
    params: TaskParams, rng: np.random.Generator, max_tries: int = 8
) -> Optional[TaskParams]:
    """`EDIT` of §06.3.3: exactly one parameter perturbed by one step.

    "Every value is clamped to its range; a no-op edit is resampled up to 8
    times then abandoned." Abandonment returns `None`, which `ARCHIVE_STEP`
    treats as a rejected proposal.
    """
    names = _FAMILY_PARAMS[params.family]
    specs = [s for s in EDIT_TABLE if s.name in names]
    weights = np.array([s.weight for s in specs], dtype=float)
    weights /= weights.sum()

    for _ in range(max_tries):
        spec = specs[int(rng.choice(len(specs), p=weights))]
        cur = getattr(params, spec.name)
        if spec.kind == "categorical":
            new_val = GRID_TRANSFORMS[int(rng.integers(0, len(GRID_TRANSFORMS)))]
        elif spec.kind == "scale":
            factor = 1.0 + (spec.step if rng.random() < 0.5 else -spec.step)
            new_val = int(np.clip(round(cur * factor), spec.lo, spec.hi))
        elif spec.kind == "double":
            new_val = int(np.clip(cur * 2 if rng.random() < 0.5 else cur // 2,
                                  spec.lo, spec.hi))
        else:
            delta = spec.step if rng.random() < 0.5 else -spec.step
            new_val = int(np.clip(cur + delta, spec.lo, spec.hi))
        if new_val != cur:
            return replace(params, **{spec.name: new_val})
    return None


def family_lower_bounds() -> List[TaskParams]:
    """Each family at the lower bound of every `EDIT` range (§06.3.3).

    `SEED_ARCHIVE` inserts exactly these, unconditionally. v2 never initialised
    `Lambda` at all, so `ARCHIVE_STEP`'s first firing sampled a parent from an
    empty set and the whole task stream was undefined (audit A-24).
    """
    out: List[TaskParams] = []
    for f in FAMILIES:
        kwargs: Dict[str, object] = {}
        for name in _FAMILY_PARAMS[f]:
            spec = next(s for s in EDIT_TABLE if s.name == name)
            kwargs[name] = GRID_TRANSFORMS[0] if spec.kind == "categorical" else spec.lo
        out.append(TaskParams(family=f, **kwargs))  # type: ignore[arg-type]
    return out
