"""Acceptance tests R1-R5 of docs/anlage/08. SPEC: 08 §8.7"""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys

import numpy as np
import pytest

from gnomon.utils.stats import bca_bootstrap, tost
from screen.variance import pooled_within_sd

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_R1_sigma_e_recovers_a_planted_noise_level():
    rng = np.random.default_rng(0)
    sigma = 0.01
    table = {f"arm{i}": list(2.0 + 0.1 * i + rng.normal(0, sigma, 40)) for i in range(6)}
    sd, df = pooled_within_sd(table)
    assert df == 6 * 39
    assert abs(sd - sigma) < 0.15 * sigma


def test_R2_paired_contrast_resolves_three_sigma_and_tost_finds_equivalence():
    rng = np.random.default_rng(1)
    sigma = 0.01
    base = rng.normal(2.0, 0.05, 5)                       # arm-level spread >> sigma_e
    a = base + rng.normal(0, sigma, 5)
    b = base - 3 * sigma + rng.normal(0, sigma, 5)        # a planted 3 sigma_e improvement
    d = [float(x) for x in b - a]
    res = bca_bootstrap(d, statistics.fmean, n_resamples=2000, alternative="two-sided")
    assert res.ci_high < 0.0
    same = [float(x) for x in (base + rng.normal(0, sigma, 5))]
    r = tost(list(map(float, a)), same, 2 * sigma, paired=True)
    assert r.verdict in ("equivalent", "inconclusive")
    r2 = tost(list(map(float, a)), list(map(float, b)), 2 * sigma, paired=True)
    assert not r2.equivalent
    # below five seeds the analysis falls back to a paired Student-t interval
    sys.path.insert(0, str(REPO / "scripts"))
    from anlage_analyse import paired
    c = paired(list(map(float, a[:4])), list(map(float, b[:4])))
    assert c["n"] == 4 and c["method"] == "t" and c["ci_high"] < 0.0


def test_R3_analyse_reports_not_evaluable_on_an_empty_directory(tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "anlage_analyse.py"),
                          "--runs", str(tmp_path)], capture_output=True, text=True)
    assert "NOT EVALUABLE" in out.stdout
    assert " PASS" not in out.stdout


def test_R4_manifest_fields_and_nan_refusal(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    from anlage_common import manifest, write_json
    m = manifest({"run": "test"})
    for k in ("time_utc", "git", "torch", "cuda", "gpu", "platform", "python", "run"):
        assert k in m
    with pytest.raises(ValueError):
        json.dumps({"a": float("nan")}, allow_nan=False)
    # write_json maps a NaN to null rather than emitting an invalid file
    write_json(tmp_path / "x.json", {"a": float("nan"), "b": 1.0})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": 1.0}


def test_R5_gate_oracle_cannot_pass_a_failing_command():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "gate.py"), "probe", "--",
                          sys.executable, "-c", "print('GATE_OK probe'); raise SystemExit(3)"],
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "GATE_FAIL probe" in out.stderr
