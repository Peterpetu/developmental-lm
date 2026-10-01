"""SPEC: 08 §8.6 -- the acceptance tests E1..E9 for the archive and the loop."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gnomon.qd.archive import Archive, Member, bin_index
from gnomon.qd.mapelites import map_elites
from gnomon.utils.stats import tost


def _member(fitness, d1, d2, archive, gen=0):
    d = (d1, d2)
    return Member(state={}, fitness=fitness, descriptors=d,
                  niche=archive.niche_of(d), generation=gen, n_decoded=1000,
                  arch_summary="", per_seed=[fitness])


def _archive(n=4, L_null=10.0):
    return Archive(b1_edges=[i / n for i in range(n + 1)],
                   b2_edges=[float(i) for i in range(n + 1)], L_null=L_null)


# ---------------------------------------------------------------------- E1

def test_E1_elitism_a_worse_genome_never_displaces_an_elite():
    a = _archive()
    assert a.insert(_member(5.0, 0.1, 0.5, a))
    assert not a.insert(_member(6.0, 0.1, 0.5, a))
    assert a.cells[(0, 0)].fitness == 5.0
    assert a.insert(_member(4.0, 0.1, 0.5, a))
    assert a.cells[(0, 0)].fitness == 4.0
    assert a.n_attempts == 3 and a.n_inserts == 2


def test_E1_qd_score_is_non_decreasing_over_a_run():
    rng = np.random.default_rng(0)
    a = _archive()
    scores = []
    for i in range(300):
        a.insert(_member(float(rng.uniform(2, 9)), float(rng.uniform(0, 1)),
                         float(rng.uniform(0, 4)), a))
        scores.append(a.qd_score())
    assert all(b >= s - 1e-12 for s, b in zip(scores, scores[1:])), (
        "qd_score fell; with per-niche elitism that is a bug, not a fluctuation"
    )


def test_E1_qd_score_is_zero_for_an_empty_archive_and_ignores_worse_than_null():
    a = _archive(L_null=10.0)
    assert a.qd_score() == 0.0
    a.insert(_member(12.0, 0.1, 0.5, a))       # worse than the null model
    assert a.qd_score() == 0.0


# ---------------------------------------------------------------------- E2

def test_E2_bins_are_closed_open_with_a_closed_top():
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert bin_index(0.0, edges) == 0
    assert bin_index(0.24, edges) == 0
    assert bin_index(0.25, edges) == 1
    assert bin_index(0.99, edges) == 3
    assert bin_index(1.0, edges) == 3, "the top bin must be closed"
    assert bin_index(-5.0, edges) == 0 and bin_index(5.0, edges) == 3


def test_E2_a_genome_lands_in_the_niche_its_descriptors_give():
    a = _archive(n=4)
    m = _member(3.0, 0.6, 2.5, a)
    assert m.niche == (2, 2)
    a.insert(m)
    assert list(a.cells) == [(2, 2)]


# ---------------------------------------------------------------------- E5, E6
#
# The two halves of the `KG-QD` control. Both are required: a test that only
# shows MAP-Elites winning cannot distinguish "the algorithm works" from "the
# comparison is rigged", and one that only shows a tie cannot detect a search
# that has stopped working.

def _run(strategy: str, *, landscape, seed: int, n_pop=12, G_gen=25):
    """`strategy` in {"mapelites", "random"}; identical budget and archive.

    The genome is `(x, y, u, v)`: fitness depends only on `(x, y)` and the
    behaviour descriptors only on `(u, v)`. Keeping them disjoint is what makes
    this a test of the *algorithm* rather than of the landscape -- with a
    descriptor that is also a fitness coordinate, only the niche containing the
    optimum can ever reach it, and the comparison measures the binning instead.
    """
    rng = torch.Generator().manual_seed(seed)
    npr = np.random.default_rng(seed + 991)
    a = _archive(n=5, L_null=10.0)

    def random_genome(i):
        return npr.uniform(0.0, 1.0, size=4)

    def vary(p1, p2, r):
        g = np.array(p1.state["g"])
        take = npr.random(4) < 0.5
        g = np.where(take, g, np.array(p2.state["g"]))
        return np.clip(g + npr.normal(0, 0.06, size=4), 0.0, 1.0)

    def evaluate(g, gen):
        f = landscape(g)
        d = (float(g[2]), float(g[3]) * 5.0)
        return Member(state={"g": list(g)}, fitness=f, descriptors=d,
                      niche=a.niche_of(d), generation=gen, n_decoded=1,
                      arch_summary="", per_seed=[f])

    return map_elites(a, n_init=n_pop, n_pop=n_pop, G_gen=G_gen,
                      p_random=1.0 if strategy == "random" else 0.10,
                      random_genome=random_genome, vary=vary,
                      evaluate=evaluate, rng=rng)


def _structureless(g):
    """White noise in the genome: nothing local refinement can exploit.

    Not a constant -- a constant would make every arm trivially equal and test
    nothing. A deterministic hash means neighbouring genomes have unrelated
    fitness, so the best-of-N distribution is the same however the N genomes
    were chosen, which is exactly the null `KG-QD` has to be able to accept.
    """
    key = tuple(int(v * 1e6) for v in g)
    return 2.0 + 7.0 * np.random.default_rng(abs(hash(key)) % (2**32)).random()


def _stepping_stones(g):
    """A narrow well at the bottom of a broad basin that leads into it.

    Random sampling almost never lands in the well (radius ~0.028 in a unit
    square); local refinement from anywhere in the basin walks downhill into it.
    That is the structure quality-diversity is supposed to exploit.
    """
    d = (float(g[0]) - 0.8) ** 2 + (float(g[1]) - 0.3) ** 2
    return 8.0 - 2.0 * math.exp(-d / 0.15) - 4.0 * math.exp(-d / 0.0008)


def test_E5_on_a_structureless_landscape_the_two_are_indistinguishable():
    """SPEC: 08 §8.6 E5 -- the control for `KG-QD`'s null direction."""
    me = [_run("mapelites", landscape=_structureless, seed=s).best().fitness
          for s in range(10)]
    rs = [_run("random", landscape=_structureless, seed=s).best().fitness
          for s in range(10)]
    # Unpaired: MAP-Elites and random search consume their generators
    # differently, so seed s does not name the same draws in both arms.
    r = tost(me, rs, 0.60, paired=False)
    assert r.equivalent, (r, float(np.mean(me)), float(np.mean(rs)))


def test_E6_on_a_stepping_stone_landscape_map_elites_wins():
    """SPEC: 08 §8.6 E6 -- the control for `KG-QD`'s alternative direction."""
    me = [_run("mapelites", landscape=_stepping_stones, seed=s).best().fitness
          for s in range(10)]
    rs = [_run("random", landscape=_stepping_stones, seed=s).best().fitness
          for s in range(10)]
    assert np.mean(me) < np.mean(rs) - 0.5, (float(np.mean(me)), float(np.mean(rs)))
    assert sum(x < y for x, y in zip(me, rs)) >= 8, (me, rs)


def test_E5_E6_together_show_the_control_fires_in_both_directions():
    """Neither half alone is a control; this asserts the pair."""
    null_me = np.mean([_run("mapelites", landscape=_structureless, seed=s)
                       .best().fitness for s in range(6)])
    null_rs = np.mean([_run("random", landscape=_structureless, seed=s)
                       .best().fitness for s in range(6)])
    str_me = np.mean([_run("mapelites", landscape=_stepping_stones, seed=s)
                      .best().fitness for s in range(6)])
    str_rs = np.mean([_run("random", landscape=_stepping_stones, seed=s)
                      .best().fitness for s in range(6)])
    assert abs(null_me - null_rs) < 0.35 < (str_rs - str_me)


# ---------------------------------------------------------------------- E4

def test_E4_a_whole_run_is_reproducible_from_its_seed():
    a = _run("mapelites", landscape=_stepping_stones, seed=3)
    b = _run("mapelites", landscape=_stepping_stones, seed=3)
    assert a.history == b.history
    assert {k: v.fitness for k, v in a.cells.items()} == \
           {k: v.fitness for k, v in b.cells.items()}


def test_E4_different_seeds_differ():
    a = _run("mapelites", landscape=_stepping_stones, seed=3)
    b = _run("mapelites", landscape=_stepping_stones, seed=4)
    assert a.history != b.history


# ---------------------------------------------------------------------- E7

def test_E7_archive_round_trip(tmp_path):
    a = _run("mapelites", landscape=_stepping_stones, seed=5)
    a.save(tmp_path / "arch.pt")
    b = Archive.load(tmp_path / "arch.pt")
    assert b.n_attempts == a.n_attempts and b.n_inserts == a.n_inserts
    assert b.history == a.history
    assert set(b.cells) == set(a.cells)
    for k in a.cells:
        assert b.cells[k].fitness == a.cells[k].fitness
        assert b.cells[k].descriptors == a.cells[k].descriptors
        assert b.cells[k].state == a.cells[k].state
    assert b.qd_score() == a.qd_score() and b.coverage() == a.coverage()


# ---------------------------------------------------------------------- E8

def test_E8_coverage_counts_filled_niches_not_attempts():
    a = _archive(n=4)
    for _ in range(50):
        a.insert(_member(5.0, 0.1, 0.5, a))          # all into one niche
    assert a.n_attempts == 50 and len(a.cells) == 1
    assert a.coverage() == pytest.approx(1 / 16)


# ---------------------------------------------------------------------- E9

def test_E9_the_budget_is_exact():
    calls = []
    a = _archive()

    def ev(g, gen):
        calls.append(gen)
        return _member(float(len(calls)), 0.1, 0.5, a)

    map_elites(a, n_init=7, n_pop=5, G_gen=4, p_random=1.0,
               random_genome=lambda i: i, vary=lambda p1, p2, r: 0,
               evaluate=ev, rng=torch.Generator().manual_seed(0))
    assert len(calls) == 7 + 4 * 5
    assert calls.count(0) == 7 and calls.count(1) == 5


def test_parents_are_sampled_uniformly_over_filled_niches():
    """SPEC: 08 §8.2. Uniform over niches, not over fitness."""
    a = _archive(n=4)
    a.insert(_member(1.0, 0.1, 0.5, a))
    a.insert(_member(9.0, 0.6, 2.5, a))
    rng = torch.Generator().manual_seed(0)
    picks = [p.fitness for p in a.sample_parents(600, rng)]
    good = picks.count(1.0)
    assert 250 < good < 350, (
        f"{good}/600 draws took the better elite; sampling is not uniform over "
        f"niches, which is what keeps stepping stones alive"
    )


def test_sample_parents_on_an_empty_archive_returns_nothing():
    assert _archive().sample_parents(2, torch.Generator().manual_seed(0)) == []


def test_history_records_what_the_spec_says():
    a = _run("mapelites", landscape=_stepping_stones, seed=1)
    keys = {"generation", "coverage", "n_filled", "qd_score", "best",
            "mean_elite", "n_attempts", "n_inserts", "best_arch",
            "best_n_decoded",
            # SPEC: 07 §7.4 -- `KG-RANK` is a trend test over generations, so
            # the series must exist in the history. Until v1.1 `erank` was
            # computed for every member and written nowhere, and the criterion
            # could not be evaluated from a completed run at all.
            "mean_erank", "best_erank",
            # SPEC: 06 §6.4 -- birth_share's trajectory across generations.
            "mean_birth", "best_birth",
            # SPEC: 01 §1.7 -- which arm wrote this history.
            "lamarck"}
    assert set(a.history[0]) == keys
    assert [r["generation"] for r in a.history] == list(range(26))


def test_history_rows_hold_no_NaN_and_survive_json():
    """SPEC: 08 §8.4. A row with `NaN` in it is not JSON and is not equal to itself.

    `json.dumps` emits a bare `NaN`, which strict parsers reject, and `NaN !=
    NaN` means E4's determinism and E7's round-trip cannot be asserted over a
    history containing one. Absent values are `None`.
    """
    import json

    a = _run("mapelites", landscape=_stepping_stones, seed=1)
    blob = json.dumps(a.history, allow_nan=False)      # raises if any NaN
    assert json.loads(blob) == a.history
    for row in a.history:
        for k, v in row.items():
            assert v == v, f"{k} is NaN"


# ---------------------------------------------------------------------- E3

def test_E3_the_seed_set_is_a_function_of_the_generation_alone():
    """SPEC: 08 §8.6 E3. Seed-paired within a generation.

    A function of the generation and nothing else: not of which genome is being
    evaluated, not of the order genomes arrive in, not of any running generator.
    An unpaired comparison does not produce a wrong number, it produces a
    noisier one, which is far harder to notice than a wrong one.
    """
    from gnomon.qd.evolve import generation_seeds

    assert generation_seeds(0, 2) == [0, 1]
    assert generation_seeds(3, 2) == [3000, 3001]
    assert generation_seeds(3, 2) == generation_seeds(3, 2)
    # Distinct across generations, so a genome re-evaluated in a later
    # generation is not scored on the tokens it was selected on.
    seen = [s for g in range(40) for s in generation_seeds(g, 4)]
    assert len(set(seen)) == len(seen)


def test_E3_every_genome_in_a_generation_sees_the_same_seeds():
    """SPEC: 08 §8.6 E3, at the level of the loop rather than the formula.

    Driven through `evolve` itself with a stub runner, so what is asserted is
    what the evolutionary loop actually does with the seeds -- the formula being
    right and the loop ignoring it is precisely the failure this must catch.
    """
    import gnomon.qd.evolve as E
    from gnomon.config import GenomeShape, M_PICO
    from gnomon.lifetime.train import LifetimeResult

    calls: list[tuple[int, int]] = []          # (generation, seed)

    class StubRunner:
        def __init__(self, cfg, shape, **kw):
            self.model = None
            self.token_code, self.token_provenance = "freq", {}
            self._gen = 0

        def set_arch(self, arch):
            self.arch = arch

        def load_genome(self, g):
            pass

        def run(self, genome, *, seed, **kw):
            calls.append(seed)
            return LifetimeResult(final=1.0 + 1e-6 * seed, birth=2.0, curve=[],
                                  train_curve=[], seconds=0.0, n_genome=1,
                                  n_soma=1, n_decoded=1)

    real_runner, real_erank = E.LifetimeRunner, E._decoded_erank
    E.LifetimeRunner = StubRunner
    E._decoded_erank = lambda r: 0.3
    try:
        ev = E.EvolveConfig(n_init=4, n_pop=3, n_seed=2, G_gen=2, seed=0)
        E.evolve(M_PICO, GenomeShape(), ev, train_shard=None, eval_shard=None,
                 device="cpu", log=lambda s: None)
    finally:
        E.LifetimeRunner, E._decoded_erank = real_runner, real_erank

    # generation 0 evaluates n_init genomes, then G_gen generations of n_pop.
    per_gen = [calls[:4 * 2], calls[4 * 2:4 * 2 + 3 * 2],
               calls[4 * 2 + 3 * 2:]]
    assert [len(c) for c in per_gen] == [8, 6, 6]
    for g, chunk in enumerate(per_gen):
        want = E.generation_seeds(g, 2)
        genomes = [chunk[i:i + 2] for i in range(0, len(chunk), 2)]
        assert all(seeds == want for seeds in genomes), (g, genomes)


# --------------------------------------------------------------------- E10

def test_E10_the_lamarckian_arm_is_labelled_on_every_archive_artifact():
    """SPEC: 08 §8.6 E10, SPEC: 01 §1.7.

    §01.7 requires *every* artifact of `A-LAMARCK` to carry the label, and the
    artifacts §08 owns are the config, the member and the history row -- the
    three things that outlive the process. A genome reloaded from an archive
    months later must still say which arm made it, because the arm is reported
    as an upper bound and an unlabelled one would be read as GNOMON's result.

    The lifetime-level half of E10 -- that the arm moves every tensor of `Gamma`
    and closes the path again -- lives in `test_lifetime.py`, with the runner
    that implements it.
    """
    from gnomon.qd.archive import Archive, Member
    from gnomon.qd.evolve import EvolveConfig

    assert EvolveConfig(lamarck=True).label() == "A-LAMARCK"
    assert EvolveConfig(lamarck=True, random_search=True).label() == "A-LAMARCK"
    assert EvolveConfig().label() == "GNOMON"

    a = _archive(n=2)
    for lam in (True, False):
        m = _member(5.0, 0.1, 0.5, a)
        m.lamarck = lam
        a.cells.clear()
        a.insert(m)
        assert a.record(0)["lamarck"] is lam
    # An archive written before the arm existed loads with the label absent,
    # which must read as "not Lamarckian" rather than raising: `Member` gives
    # `lamarck` a default for exactly that reason.
    assert Member(state={}, fitness=1.0, descriptors=(0.0, 1.0), niche=(0, 0),
                  generation=0, n_decoded=1, arch_summary="",
                  per_seed=[1.0]).lamarck is False


def test_the_arm_table_and_the_arms_agree():
    """SPEC: 08 §8.5. A table of arms that disagrees with the arms hides one.

    §09.6 rule 4 requires any arm that failed to run to be listed with its
    reason, and "listed" is only meaningful against a table that matches what
    can be run. Until v1.1 §08.5 named `A-ETA`, which is v0.9's vocabulary --
    `eta` stopped being a scalar gain at v1.0 -- and omitted `A-FIXSIG`, which
    was implemented, and `A-LAMARCK`, which was not.

    Parsed from the specification rather than restated, so the two cannot drift.
    """
    import pathlib
    import re

    spec = pathlib.Path("docs/gnomon/08-archive.md").read_text(encoding="utf-8")
    # The whole of §8.5 up to the next heading. Splitting on the first
    # `**[NORMATIVE` -- as this first did -- cuts at the sentence *introducing*
    # the table and yields an empty block, so the test compared nothing against
    # everything and "passed" only by failing loudly.
    block = spec.split("## 8.5 Ablations", 1)[1].split("\n## ", 1)[0]
    named = set(re.findall(r"^\|\s*([AR]-[A-Z0-9-]+)\s*\|", block, re.M))
    assert named, "no arms parsed out of §08.5; the table's shape has changed"

    src = pathlib.Path("scripts/gnomon_evolve.py").read_text(encoding="utf-8")
    arms = set(re.findall(r'^\s*"([A-Z][A-Z0-9-]*)":\s*\{', src, re.M))
    arms.discard("GNOMON")            # the unablated run is not an ablation

    assert named == arms, (
        f"§08.5 names {sorted(named - arms)} that cannot be run, and "
        f"{sorted(arms - named)} can be run and is not named"
    )


def test_every_arm_produces_its_own_label():
    """SPEC: 01 §1.7, SPEC: 08 §8.5. An unlabelled arm is an unattributed result."""
    import pathlib
    import re

    from gnomon.qd.evolve import EvolveConfig

    src = pathlib.Path("scripts/gnomon_evolve.py").read_text(encoding="utf-8")
    body = src.split("ARMS = {", 1)[1].split("\n}", 1)[0]
    for name, kwargs in re.findall(r'"([A-Z][A-Z0-9-]*)":\s*\{([^}]*)\}', body):
        kw = dict(re.findall(r'"(\w+)":\s*(True|False|"\w[\w-]*")', kwargs))
        parsed = {k: (v == "True" if v in ("True", "False") else v.strip('"'))
                  for k, v in kw.items()}
        assert EvolveConfig(**parsed).label() == name, name
