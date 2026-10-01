"""Writing a result file that can be read back.

SPEC: 09 §9.6

Every reported number in this project ends up in a JSON file that some later
script, or some later person, parses. Two ways that goes wrong, both of which
happened here:

* **`NaN` is not JSON.** `json.dumps` emits a bare `NaN` token by default. It is
  not in the grammar, strict parsers reject it, and `json.loads` accepts it only
  because Python's decoder is lenient. A run whose result file cannot be read by
  anything but Python is not a reproducible result.
* **`NaN != NaN`.** A record containing one does not compare equal to itself, so
  a round-trip test over it cannot pass and a determinism test over it cannot
  fail. Both were true of `Archive.history` until v1.1.

An absent quantity is `None`, which is `null` in JSON, round-trips, and compares
equal. `write_json` then serialises with `allow_nan=False`, so a `NaN` that
escapes `finite_or_none` **raises at the point of writing** rather than silently
producing a file that is invalid. A guard that fails loudly at the moment the
mistake is made is worth more than one that has to be remembered.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Optional

__all__ = ["finite_or_none", "clean", "write_json"]


def finite_or_none(x: Any) -> Optional[float]:
    """A finite float, or `None` for `NaN`, `±inf` and `None`.

    `±inf` is included because it has the same two problems: `Infinity` is not
    in the JSON grammar either, and an infinite entry in a table is an absent
    measurement wearing a number's clothes.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def clean(obj: Any) -> Any:
    """Recursively replace every non-finite float with `None`.

    Structure-preserving: dicts, lists and tuples keep their shape, and
    non-numeric leaves (strings, bools, `None`) are passed through untouched.
    `bool` is checked before `float` because `isinstance(True, int)` is true and
    a flag must not be turned into `1.0`.
    """
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        return finite_or_none(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def write_json(path: str | pathlib.Path, obj: Any, *, indent: int = 2) -> pathlib.Path:
    """Write `obj` as strict JSON, creating parents. SPEC: 09 §9.6.

    `allow_nan=False` is the point of this function: it turns "wrote a file no
    strict parser can read" into an exception at the moment of writing.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(obj), indent=indent, allow_nan=False),
                 encoding="utf-8")
    return p
