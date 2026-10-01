"""The evolutionary loop, wired to GNOMON. SPEC: 01 §1.5, SPEC: 08 §8.2, §8.3, §8.5.

C4 of the constitution: the only thing that improves `Gamma`. The algorithm
itself lives in `mapelites.py`, driven here by a real lifetime; keeping them
apart is what lets §08.6's E5 and E6 exercise the same loop against a synthetic
landscape with no GPU, and what makes `A-RAND` an exactly matched control rather
than a separately written one.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from typing import Callable, Optional

import torch

from ..config import GenomeShape, PhenoConfig
from ..data.stream import Shard
from ..genome.arch import ArchHead, Strategy, crossover, mutate
from ..genome.decoder import Decoder
from ..lifetime.runner import LifetimeRunner
from ..utils.stats import effective_rank
from .archive import Archive, Member
from .mapelites import map_elites

__all__ = ["EvolveConfig", "Genome", "evolve"]


@dataclasses.dataclass(frozen=True)
class EvolveConfig:
    """SPEC: 08 §8.3 -- the defaults and their justification live in the spec."""

    n_init: int = 32
    n_pop: int = 32
    n_seed: int = 2
    G_gen: int = 60
    sigma_rel: float = 0.15
    p_mut: float = 0.25
    p_random: float = 0.10
    n_b1_bins: int = 6
    n_b2_bins: int = 6
    seed: int = 0
    # SPEC: 08 §8.5 -- ablation switches. Each removes exactly one mechanism,
    # so an arm differs from GNOMON in one place and the comparison is clean.
    random_search: bool = False        # A-RAND
    freeze_arch: bool = False          # A-NOARCH
    no_crossover: bool = False         # A-NOX
    one_descriptor: bool = False       # A-1D
    rule: str = "dpr"                  # SPEC: 10 §10.6 -- the lifetime arm
    fixed_sigma: bool = False          # A-FIXSIG: no self-adaptation (§10.4)
    anneal: bool = True                # A-ANNEAL
    # SPEC: 08 §8.5 -- A-LAMARCK. The paper's own inner step: the lifetime loss
    # back-propagates through the decode into `Gamma`, so acquired change is
    # inherited. Reported as an **upper bound**, never as GNOMON's result.
    lamarck: bool = False

    @property
    def n_lifetimes(self) -> int:
        return (self.n_init + self.G_gen * self.n_pop) * self.n_seed

    def label(self) -> str:
        # A-LAMARCK first: §01.7 requires every artifact of that arm to carry
        # the label, so it must win over any other flag set alongside it.
        if self.lamarck:
            return "A-LAMARCK"
        for flag, name in (("random_search", "A-RAND"), ("freeze_arch", "A-NOARCH"),
                           ("no_crossover", "A-NOX"), ("one_descriptor", "A-1D")):
            if getattr(self, flag):
                return name
        if self.rule != "dpr":
            return f"R-{self.rule.upper()}"
        if self.fixed_sigma:
            return "A-FIXSIG"
        if not self.anneal:
            return "A-ANNEAL"
        return "GNOMON"


class Genome:
    """`Gamma = (Gamma_w, Gamma_a)`, the unit of variation. SPEC: 07 §7.5."""

    def __init__(self, shape: GenomeShape, *, seed: int = 0):
        self.shape = shape
        self.w = Decoder(shape, seed=seed)
        self.a = ArchHead(shape, seed=seed)
        # SPEC: 10 §10.4 -- the mutation strengths are part of the genome, so
        # how a lineage varies is heritable and under selection.
        self.s = Strategy([n for n, _ in self.w.named_parameters()]
                          + [n for n, _ in self.a.named_parameters()])

    @property
    def modules(self) -> list:
        return [self.w, self.a]

    def n_params(self) -> int:
        return self.w.n_params() + self.a.n_params() + self.s.n_params()

    def state(self) -> dict:
        return {**{f"w.{k}": v.detach().clone()
                   for k, v in self.w.state_dict().items()},
                **{f"a.{k}": v.detach().clone()
                   for k, v in self.a.state_dict().items()},
                **{f"s.{k}": v.detach().clone()
                   for k, v in self.s.state_dict().items()}}

    def load(self, state: dict) -> "Genome":
        self.w.load_state_dict({k[2:]: v for k, v in state.items()
                                if k.startswith("w.")})
        self.a.load_state_dict({k[2:]: v for k, v in state.items()
                                if k.startswith("a.")})
        if any(k.startswith("s.") for k in state):
            self.s.load_state_dict({k[2:]: v for k, v in state.items()
                                    if k.startswith("s.")})
        return self


def generation_seeds(generation: int, n_seed: int) -> list[int]:
    """The lifetime seeds every genome of one generation is evaluated on.

    SPEC: 08 §8.2, §8.6 E3. A *function of the generation alone*, so it cannot
    depend on which genome is being evaluated, on the order genomes arrive in,
    or on any running generator. That is what makes every within-generation
    comparison seed-paired: two genomes differ by their genomes and by nothing
    else in the draw. E3 plants a mutant that unpairs them and requires the
    suite to reject it, because an unpaired comparison is not a wrong number --
    it is a noisier one, which is far harder to notice.
    """
    return [1000 * int(generation) + s for s in range(int(n_seed))]


def _edges(lo: float, hi: float, n: int) -> list[float]:
    if hi <= lo:
        hi = lo + 1e-6
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def descriptor_range(cfg: PhenoConfig, shape: GenomeShape) -> tuple[float, float]:
    """The achievable `log10(decoded params)` range, from the menus alone.

    Computed rather than observed, so the map's axes do not depend on which
    genomes a run happened to draw. Rescaling them mid-run would move every
    genome's niche and make the coverage series meaningless.
    """
    from ..genome.arch import MENU_F, MENU_H, MENU_L
    from ..pheno.plan import Arch

    def extreme(pick, op):
        L = min(pick(MENU_L), cfg.L)
        h = max(cfg.n_kv, int(round(pick(MENU_H) * cfg.n_h)))
        h = max(cfg.n_kv, h - h % cfg.n_kv)
        while h * cfg.d_head > shape.d_max and h > cfg.n_kv:
            h -= cfg.n_kv
        f = min(pick(MENU_F) * cfg.d_model, shape.d_max)
        return math.log10(Arch((op,) * L, (h,) * L, (f,) * L,
                               (cfg.T,) * L).n_decoded(cfg))

    return extreme(min, "delta"), extreme(max, "attn")


def evolve(
    cfg: PhenoConfig,
    shape: GenomeShape,
    ev: EvolveConfig,
    *,
    train_shard: Shard,
    eval_shard: Shard,
    device: str = "cuda",
    out_dir: Optional[str] = None,
    compile_mode: Optional[str] = None,
    n_eval_batches: int = 4,
    token_code: str = "freq",
    log: Callable[[str], None] = print,
) -> Archive:
    """SPEC: 08 §8.2. Returns the archive; writes it and its history to `out_dir`.

    **`compile_mode` defaults to `None` (eager), and that is a measured
    decision, not an oversight.** `torch.compile(mode="reduce-overhead")` takes
    a fixed-architecture step from 79.5 to 11.5 ms, but every genome here
    decodes its own graph, so each distinct architecture pays a ~30 s inductor
    compile and some of them fail outright on a machine without MSVC (inductor
    emits a CPU kernel it cannot build). Eight architectures took over ten
    minutes just to bind. The throughput is recovered instead by batch size:
    the step is dispatch-bound, so at M-pico batch 128 costs 100 ms/step against
    batch 32's 82 ms while carrying four times the tokens -- 6.15 ms per
    thousand tokens against 20.1, a 3.3x gain for free.
    """
    t0 = time.time()
    rng = torch.Generator().manual_seed(ev.seed)
    runner = LifetimeRunner(cfg, shape, train_shard=train_shard,
                            eval_shard=eval_shard, device=device,
                            compile_mode=compile_mode,
                            n_eval_batches=n_eval_batches,
                            token_code=token_code)
    lo, hi = descriptor_range(cfg, shape)
    archive = Archive(
        b1_edges=_edges(0.0, 1.0, ev.n_b1_bins),
        b2_edges=[lo, hi] if ev.one_descriptor else _edges(lo, hi, ev.n_b2_bins),
        L_null=math.log(cfg.vocab),
    )
    scratch = Genome(shape, seed=0)
    pa, pb = Genome(shape, seed=1), Genome(shape, seed=2)
    n_lifetimes = 0

    def random_genome(i: int) -> Genome:
        return scratch.load(Genome(shape, seed=ev.seed * 100003 + 7919 + i).state())

    def vary(p1: Member, p2: Member, r: torch.Generator) -> Genome:
        pa.load(p1.state)
        pb.load(p2.state)
        if ev.no_crossover:
            scratch.load(pa.state())
        else:
            crossover(scratch.modules, pa.modules, pb.modules, generator=r)
        mutate([scratch.w] if ev.freeze_arch else scratch.modules,
               sigma_rel=ev.sigma_rel, p_mut=ev.p_mut, generator=r,
               strategy=None if ev.fixed_sigma else scratch.s)
        if ev.freeze_arch:
            # A-NOARCH: the architecture head is inherited untouched, so every
            # genome in the run decodes the graph its ancestor did.
            scratch.a.load_state_dict(pa.a.state_dict())
        return scratch

    def evaluate(g: Genome, generation: int) -> Member:
        nonlocal n_lifetimes
        arch = g.a.decode_arch(cfg)
        runner.set_arch(arch)
        w_state = {k[2:]: v for k, v in g.state().items() if k.startswith("w.")}
        runner.load_genome(w_state)
        per_seed, births = [], []
        for i, seed in enumerate(generation_seeds(generation, ev.n_seed)):
            # SPEC: 08 §8.2 -- seeds are shared across every genome in a
            # generation, so within-generation comparisons are seed-paired and
            # the differences between genomes are not differences in the draw.
            #
            # SPEC: 08 §8.5 -- in `A-LAMARCK` the genome is loaded once and the
            # seeds run *consecutively on it*: they are successive lives of one
            # lineage, and reloading between them would discard exactly the
            # acquired change the arm exists to inherit. The consequence is
            # stated rather than hidden: the later seeds start from a genome the
            # earlier ones improved, so the mean over seeds is not an unbiased
            # estimate of the genome's fitness the way it is in every other arm.
            r = runner.run(w_state, seed=seed,
                           rule=ev.rule, anneal=ev.anneal,
                           lamarck=ev.lamarck, load=(i == 0 or not ev.lamarck))
            per_seed.append(r.final)
            births.append(r.birth)
            n_lifetimes += 1
        state = g.state()
        if ev.lamarck:
            # Read the acquired genome back out. Without this the arm would
            # compute a gradient, apply it, and then throw it away at the niche
            # boundary -- Lamarckian in the lifetime and Darwinian in the
            # archive, which is neither arm.
            state = {**state,
                     **{f"w.{k}": v for k, v in runner.export_genome().items()}}
        d = arch.descriptors(cfg)
        return Member(
            state=state, fitness=sum(per_seed) / len(per_seed),
            descriptors=d, niche=archive.niche_of(d), generation=generation,
            n_decoded=arch.n_decoded(cfg), arch_summary=arch.summary(),
            per_seed=per_seed, birth=sum(births) / len(births),
            erank=_decoded_erank(runner), lamarck=ev.lamarck,
        )

    def on_generation(g: int, row: dict) -> None:
        log(f"  gen {g:3d}  cov {row['coverage']:.2f}  QD {row['qd_score']:7.2f}  "
            f"best {row['best']:.4f}  mean {row['mean_elite']:.4f}  "
            f"[{row['best_arch'][:38]}]  ({time.time()-t0:6.0f}s)")
        if out_dir:
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            archive.save(pathlib.Path(out_dir) / "archive.pt")
            pathlib.Path(out_dir, "history.json").write_text(
                json.dumps(archive.history, indent=2))
            # SPEC: 02 §2.9 -- the token code is derived from the corpus, so
            # what it was derived from is written beside the archive rather
            # than left to be reconstructed from the run's command line.
            pathlib.Path(out_dir, "provenance.json").write_text(
                json.dumps(runner.token_provenance, indent=2))

    log(f"{ev.label()}  seed {ev.seed}  {cfg.name}  |Gamma| = "
        f"{scratch.n_params():,}  {archive.n_niches} niches  "
        f"{ev.n_lifetimes} lifetimes  B2 in [{lo:.2f}, {hi:.2f}]  "
        f"token code {runner.token_code} (§02.9)")
    map_elites(archive, n_init=ev.n_init, n_pop=ev.n_pop, G_gen=ev.G_gen,
               p_random=1.0 if ev.random_search else ev.p_random,
               random_genome=random_genome, vary=vary, evaluate=evaluate,
               rng=rng, on_generation=on_generation)
    assert n_lifetimes == ev.n_lifetimes, (n_lifetimes, ev.n_lifetimes)
    log(f"\n{archive.summary()}")
    log(f"{n_lifetimes} lifetimes in {time.time()-t0:.0f}s "
        f"({(time.time()-t0)/n_lifetimes:.1f}s each)")
    return archive


def _decoded_erank(runner: LifetimeRunner) -> float:
    """Mean decoded `erank` over the layers' `Q`, normalised. SPEC: 07 §7.4.

    A diagnostic, not a descriptor: it is a property of `Gamma_w` rather than of
    the architecture. `KG-RANK` fires if the archive's elites show it collapsing.
    """
    from ..config import ROLE_Q

    m = runner.model
    with torch.no_grad():
        W = m.decode()
        vals = []
        for l in range(m.arch.L):
            q = W[(ROLE_Q, l)].detach().float()
            vals.append(effective_rank(torch.linalg.svdvals(q).tolist())
                        / min(q.shape))
    return float(sum(vals) / len(vals))
