"""G-SPECNUM: every configured constant of docs/anlage recomputed from the module tree.

SPEC: 03 §3.6
SPEC: 04 §4.5
SPEC: 02 §2.6

GNOMON's headline ratio was stated three times and agreed with the code in none
of them (`docs/gnomon/AUDIT-v1.1 A-3`). Every number a specification table
carries is asserted here as an equality against the module tree, and the
tables of §04.5 are regenerated from this test's output.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from anlage.config import (
    GenomeShape, MATRIX_ROLES, N_ROLES, PILE, ROLE_AXES, TINYSTORIES, VECTOR_ROLES,
    assert_scale_free,
)
from anlage.genome import Genome
from anlage.latents import LatentBank
from anlage.model import Phenotype

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_genome_count_table_of_03_6():
    shape = GenomeShape()
    t = Genome(shape).count_table()
    rows = {
        "W_1": 256 * 45, "c_1": 256, "W_1_tok": 256 * 47, "c_1_tok": 256,
        "W_1b": 256 * 256, "c_1b": 256, "W_2": 4 * 64 * 256, "c_2": 4 * 64,
        "O": 16 * 4 * 64 * 64, "kappa_hat": 16, "o": 4 * 64,
        "A_sigma": 16 * 12, "b_sigma": 16, "w_sigma": 16,
    }
    assert t == rows
    assert sum(rows.values()) == 418_288 == shape.n_params()


def test_roles_and_axes_of_04_2():
    assert N_ROLES == 20 and len(MATRIX_ROLES) == 16 and len(VECTOR_ROLES) == 4
    assert set(ROLE_AXES) == set(range(N_ROLES))


def test_scale_free_over_both_lineages():
    assert_scale_free(GenomeShape(), [TINYSTORIES, PILE])


@pytest.mark.parametrize("lin", [TINYSTORIES, PILE], ids=lambda l: l.name)
def test_phenotype_counts_of_04_5_are_the_module_trees(lin):
    """The table of §04.5 is written from these numbers; a drift is a defect."""
    table = {}
    for st in lin.stages:
        m = Phenotype(st, lin.vocab, seed=0)
        n = m.n_params()
        assert n == st.n_params(lin.vocab)
        z = LatentBank(GenomeShape(), st, lin.vocab).n_params()
        table[st.summary()] = {"params": n, "nonemb": m.n_params_nonemb(), "latents": z,
                               "ratio_genome": n / 418_288,
                               "ratio_genome_latents": n / (418_288 + z)}
    out = REPO / "runs" / "anlage" / "cache" / f"specnum_{lin.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    # the numbers recorded in docs/anlage/04 §4.5 after v0.1
    expected = {
        "tinystories": {"s0:d256/L8/h4/kv2/F1024": 13_011_201,
                        "s1:d384/L12/h6/kv2/F1536": 35_875_201,
                        "s2:d512/L16/h8/kv2/F2048": 78_307_841},
        "pile": {"s0:d512/L12/h8/kv4/F2048": 73_804_289,
                 "s1:d768/L18/h12/kv4/F3072": 207_116_033,
                 "s2:d1024/L24/h16/kv4/F4096": 461_186_049},
    }[lin.name]
    for k, v in expected.items():
        assert table[k]["params"] == v, (k, table[k]["params"], v)
    latents = {
        "tinystories": {"s0:d256/L8/h4/kv2/F1024": 300_096,
                        "s1:d384/L12/h6/kv2/F1536": 508_256,
                        "s2:d512/L16/h8/kv2/F2048": 798_848},
        "pile": {"s0:d512/L12/h8/kv4/F2048": 765_024,
                 "s1:d768/L18/h12/kv4/F3072": 1_387_408,
                 "s2:d1024/L24/h16/kv4/F4096": 2_257_088},
    }[lin.name]
    for k, v in latents.items():
        assert table[k]["latents"] == v, (k, table[k]["latents"], v)
