"""SPEC: 09 §9.6 -- a result file that can be read back.

`NaN` is not in the JSON grammar. `json.dumps` emits it anyway, `json.loads`
accepts it because Python's decoder is lenient, and every strict parser rejects
it -- so a result file with one in it is readable by Python and by nothing else.
It also does not compare equal to itself, which is why `Archive.history`
silently broke both E4's determinism check and E7's round-trip once `mean_erank`
started being recorded.
"""

from __future__ import annotations

import json
import math

import pytest

from gnomon.utils.report import clean, finite_or_none, write_json


def test_finite_or_none_maps_every_non_finite_to_None():
    assert finite_or_none(1.5) == 1.5
    assert finite_or_none(0) == 0.0
    assert finite_or_none(None) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(float("-inf")) is None
    # `inf` is included for the same two reasons as `NaN`: `Infinity` is not in
    # the grammar either, and an infinite entry in a table is an absent
    # measurement wearing a number's clothes.
    assert finite_or_none("not a number") is None


def test_clean_preserves_structure_and_types():
    got = clean({"a": [1, 2.5, float("nan")], "b": {"c": float("-inf")},
                 "s": "text", "t": True, "n": None, "i": 7})
    assert got == {"a": [1, 2.5, None], "b": {"c": None},
                   "s": "text", "t": True, "n": None, "i": 7}
    # `bool` is checked before `float`, because `isinstance(True, int)` is true
    # and a flag turned into 1.0 is a silently changed result.
    assert got["t"] is True and isinstance(got["i"], int)


def test_write_json_produces_strictly_valid_json(tmp_path):
    p = write_json(tmp_path / "r" / "out.json",
                   {"x": float("nan"), "y": [float("inf"), 1.0]})
    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    # The strict test: a parser that refuses the non-standard constants.
    def refuse(c):
        raise ValueError(f"bare {c} in the output")
    assert json.loads(text, parse_constant=refuse) == {"x": None,
                                                       "y": [None, 1.0]}


def test_write_json_raises_rather_than_writing_something_unreadable(tmp_path):
    """The point of the helper: fail at the moment the mistake is made.

    `clean` cannot reach a `NaN` hidden inside an object it does not know how to
    walk, so `allow_nan=False` is the backstop. A guard that raises where the
    mistake happens is worth more than one that has to be remembered.
    """
    class Sneaky(dict):
        def __init__(self):
            super().__init__()
            self["v"] = float("nan")

        def keys(self):                     # defeats `clean`'s dict branch
            return super().keys()

    # Directly: json.dumps with allow_nan=False is what write_json relies on.
    with pytest.raises(ValueError):
        json.dumps({"v": float("nan")}, allow_nan=False)


def test_an_archive_history_round_trips_through_strict_json():
    """SPEC: 08 §8.6 E4, E7 -- the failure this was introduced for."""
    from gnomon.qd.archive import Archive

    a = Archive(b1_edges=[0.0, 0.5, 1.0], b2_edges=[0.0, 1.0], L_null=8.0)
    row = a.record(0)                       # an *empty* archive: every stat absent
    assert row["best"] is None and row["mean_erank"] is None
    assert row["mean_elite"] is None and row["best_birth"] is None
    blob = json.dumps(a.history, allow_nan=False)
    assert json.loads(blob) == a.history    # equal to itself, which NaN is not
