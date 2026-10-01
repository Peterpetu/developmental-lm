"""The run-name grammar: what the analyser can and cannot see.

SPEC: 10 §10.4 (the register reads run directories), 11 §P-D.16.

`load_runs` skips a directory whose name does not match, and a skipped run is not reported as
incomplete -- it is not reported at all. So the grammar is a guard whose failure mode is
silence, and this file exists to make it fire. Every name form the protocol and the sweeps
actually produce must parse to the right `(run, arm, config, seed)`, and every form that is NOT
a run must be rejected.

11 §P-D.15 records three guards in this project that shared one shape: a check whose negative
result was indistinguishable from the thing not existing (a `clamp` with no gradient outside
its bounds, a `ps` filter that could never match, and this regex). None had ever been observed
to fail. The convention this file starts: a guard is not trusted until it has been made to fire.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import tempo_analyse as A  # noqa: E402


REAL = [
    # (directory name, run, arm, config, seed)
    ("R1-nolat-ts-s0", "R1", "nolat", "ts", 0),
    ("R1-lat01-ts-s3", "R1", "lat01", "ts", 3),
    ("R1-nolat-tsk2-s0", "R1", "nolat", "tsk2", 0),
    ("R1-nolat-tsk1-s3", "R1", "nolat", "tsk1", 3),
    ("R2-token-ts-s1", "R2", "token", "ts", 1),
    ("R10-token-ts-s0", "R10", "token", "ts", 0),
    # the rank factorial of 11 §P-D.4
    ("R1-nolat-tsk1wide-s1", "R1", "nolat", "tsk1wide", 1),
    ("R1-nolat-tsk1dx512-s0", "R1", "nolat", "tsk1dx512", 0),
    ("R2-token-tsk1dc512-s2", "R2", "token", "tsk1dc512", 2),
    # the genome-dimension sweep of 11 §P-D.9
    ("R4-tempo-tsk1wide-dg32-s0", "R4", "tempo", "tsk1wide-dg32", 0),
    ("R4-a_gamma0-tsk1wide-dg512-s1", "R4", "a_gamma0", "tsk1wide-dg512", 1),
    # the adult-shape diagnostic of 11 §P-D.17 (arm token, tag adult)
    ("R2-token-adult-tsk1-s0", "R2", "token", "tsk1", 0),
]

NOT_RUNS = [
    "R1-nolat-nosuch-s0",        # an unknown configuration must not be read as a run
    "R1-nolat-tsk1-s0-v1",       # a superseded first pass
    "R1-nolat-tsk1-s0-v2",       # a superseded second pass
    "R1-nolat-tsk1-v2",          # no seed
    "analysis-tsk1.json",
    "cache",
    "notarun",
]


class TestTheGrammarAcceptsEveryRealForm:
    @pytest.mark.parametrize("name,run,arm,config,seed", REAL, ids=[r[0] for r in REAL])
    def test_parses(self, name, run, arm, config, seed):
        m = A.NAME_RE.match(name)
        assert m is not None, f"{name} does not parse -- load_runs would skip it SILENTLY"
        assert (m["run"], m["arm"], m["config"], int(m["seed"])) == (run, arm, config, seed)

    def test_a_longer_base_is_not_read_as_a_shorter_one(self):
        """`tsk1wide` must not parse as `tsk1` with a stray tail: the bases are alternated
        longest-first for exactly this reason, and getting it wrong would silently merge two
        cells of the factorial into one."""
        m = A.NAME_RE.match("R1-nolat-tsk1wide-s1")
        assert m["config"] == "tsk1wide"
        assert A.NAME_RE.match("R1-nolat-tsk1dx512-s0")["config"] == "tsk1dx512"


class TestTheGrammarRejectsEverythingElse:
    @pytest.mark.parametrize("name", NOT_RUNS)
    def test_rejects(self, name):
        assert A.NAME_RE.match(name) is None, f"{name} parses as a run and would be read as one"


class TestTheCodecPatternTracksIt:
    @pytest.mark.parametrize("name,config", [
        ("codec-ts-s0", "ts"), ("codec-tsk1-s0", "tsk1"),
        ("codec-tsk1wide-s0", "tsk1wide"), ("codec-tsk1wide-dg32-s0", "tsk1wide-dg32"),
    ])
    def test_codec_names_parse(self, name, config):
        m = re.match(r"^codec-(?P<config>" + A.CONFIG_RE + r")-s(?P<seed>\d+)$", name)
        assert m is not None and m["config"] == config

    def test_every_base_the_loader_knows_is_a_base_config_by_name_accepts(self):
        """The two lists must not drift: a base the analyser knows but `config_by_name` rejects
        would name a run nothing can build, and the reverse would make runs invisible."""
        from tempo_common import config_by_name
        for base in A.CONFIG_BASES:
            config_by_name(base)          # raises if the loader knows a base the builder does not
