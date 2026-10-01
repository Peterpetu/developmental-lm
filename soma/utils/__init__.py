"""Cross-cutting utilities: RNG streams, shape assertions, statistics, logging."""

from soma.utils.logging import EventCounter, JsonlLogger, RunPaths
from soma.utils.rng import PURPOSES, RngBundle, derive_seed
from soma.utils.shapes import ShapeError, check_dtype, check_finite, check_shape

__all__ = [
    "EventCounter",
    "JsonlLogger",
    "RunPaths",
    "PURPOSES",
    "RngBundle",
    "derive_seed",
    "ShapeError",
    "check_shape",
    "check_finite",
    "check_dtype",
]
