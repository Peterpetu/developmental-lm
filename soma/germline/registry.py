"""The Weismann barrier: the germline registry and its enforcement.

SPEC: 02 §2.1
SPEC: 02 §2.1.1
SPEC: 02 §2.1.2

Parameters are partitioned into two disjoint sets with different write
permissions, enforced in code. `ASSERT_BARRIER` is called immediately before
every optimizer step and raises on a violation.

The guard is deliberately cheap — three checks per registered tensor and one set
intersection — because §07.2 step 6c calls it every step, and a guard that is
too expensive to run every step is a guard that gets moved behind a flag and
then behind a disabled flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple

import torch

__all__ = [
    "BarrierViolation",
    "GermlineRegistry",
    "assert_barrier",
    "GERMLINE_MEMBERS",
    "SOMA_MEMBERS",
    "BOOKKEEPING_MEMBERS",
]


class BarrierViolation(RuntimeError):
    """Raised when a germline tensor is gradient-enabled or optimizer-visible."""


# The exhaustive membership table of §2.1.1. Any tensor not listed anywhere is a
# specification defect; `GermlineRegistry.check_exhaustive` enforces that.
GERMLINE_MEMBERS: Tuple[str, ...] = (
    "mu_g",
    "log_sigma_g",
    "policy",
    "inventory",
    "g_u",
)
"""§2.1.1. `g_u` is *germline-addressed*: discrete, selection-only."""

SOMA_MEMBERS: Tuple[str, ...] = (
    "A",
    "B",
    "kappa",
    "W_shared",
    "W_key",
    "W_ctx",
    "W_pos",
    "K_unit1",
    "K_unit2",
    "p_u",
    "mixer",
    "attention",
    "embedding",
    "norm",
)
"""§2.1.1. `K_unit1, K_unit2` are soma: were they germline, `|Gamma|` would scale
with `U_max` and §09.3's bottleneck claim (9.1) would be false by construction,
firing K9 definitionally without any experiment."""

BOOKKEEPING_MEMBERS: Tuple[str, ...] = (
    "slot_u",
    "live_u",
    "dying_u",
    "m_u",
    "age_u",
    "nu_u",
    "nu_bar_u",
    "h_bar_u",
    "lineage",
)
"""§2.1.1 and §01.1. `slot_u` is bookkeeping, not germline: `BIRTH` assigns a
slot within a lifetime and nothing about it is inherited. (Spec v2 §1.1 called
it germline-addressed while §02.1.1 called it bookkeeping; the latter is
correct, and §01.1 v3 records the resolution.)"""


@dataclass
class GermlineRegistry:
    """The set of tensors on the germline side of the barrier.

    SPEC: 02 §2.1.2

    Registration is by object identity, not by name, so a tensor that is
    reassigned under the same attribute name does not silently leave the
    registry. The registry additionally remembers the name for diagnostics.
    """

    _tensors: Dict[int, Tuple[str, torch.Tensor]] = field(default_factory=dict)

    def register(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Add a tensor to the germline side and enforce its invariants now."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"germline member {name!r} is not a Tensor")
        if tensor.requires_grad:
            raise BarrierViolation(
                f"germline tensor {name!r} was registered with requires_grad=True; "
                f"gradients are prohibited on the germline (§02.1)"
            )
        base = name.split(".")[-1].split("[")[0]
        if base not in GERMLINE_MEMBERS:
            raise BarrierViolation(
                f"{name!r} is not in the exhaustive germline membership table of "
                f"§02.1.1 (members: {GERMLINE_MEMBERS}). Adding a tensor to the "
                f"germline requires a spec change, not a code change."
            )
        self._tensors[id(tensor)] = (name, tensor)
        return tensor

    def unregister(self, tensor: torch.Tensor) -> None:
        self._tensors.pop(id(tensor), None)

    def ids(self) -> Set[int]:
        return set(self._tensors)

    def names(self) -> List[str]:
        return [n for n, _ in self._tensors.values()]

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, t in self._tensors.values():
            yield name, t

    def __len__(self) -> int:
        return len(self._tensors)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _, t in self._tensors.values():
            yield t

    def numel(self) -> int:
        """`|Gamma|` as realised, in elements.

        Used by gate `G-GERMSIZE` to *measure* `d|Gamma|/d|U| = 0` rather than
        assert it (§09.3, K9).
        """
        return sum(int(t.numel()) for _, t in self._tensors.values())


def assert_barrier(
    registry: GermlineRegistry,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    optimizer_param_ids: Optional[Iterable[int]] = None,
) -> None:
    """`ASSERT_BARRIER` of §02.1.2. Called before every optimizer step.

    SPEC: 02 §2.1.2

    ```
    for tensor in GermlineRegistry:
        assert tensor.requires_grad is False
        assert tensor.grad is None
    assert GermlineRegistry.ids() & optimizer.param_ids() == empty
    ```

    The third check is the one that catches the realistic failure. The first two
    catch a tensor that was made trainable; the third catches a tensor that was
    *handed to an optimizer* while still non-trainable — which is silent until
    something later flips `requires_grad`, at which point the germline starts
    receiving updates with no error anywhere.
    """
    for name, tensor in registry.items():
        if tensor.requires_grad:
            raise BarrierViolation(
                f"Weismann barrier violated: germline tensor {name!r} has "
                f"requires_grad=True"
            )
        if tensor.grad is not None:
            raise BarrierViolation(
                f"Weismann barrier violated: germline tensor {name!r} has a "
                f"non-None .grad"
            )

    if optimizer_param_ids is None and optimizer is not None:
        optimizer_param_ids = [
            id(p) for group in optimizer.param_groups for p in group["params"]
        ]
    if optimizer_param_ids is not None:
        overlap = registry.ids() & set(optimizer_param_ids)
        if overlap:
            offending = [n for n, t in registry.items() if id(t) in overlap]
            raise BarrierViolation(
                f"Weismann barrier violated: germline tensors {offending} are "
                f"registered with an optimizer"
            )
