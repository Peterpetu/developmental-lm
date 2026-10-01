"""The checkpoint a run writes is the checkpoint the evaluator reads.

SPEC: 07 §7.4.2
SPEC: 06 §6.2

`theta_m0` — the frozen, better reference §06.2's Rho-1 token selection needs —
is produced by the M0 run and consumed by `scripts/eval_tinystories.py` and by
the M1 lifetime. Audit A-70 found the writer missing entirely (the report
recorded a path to a file nothing created); audit A-71 found the writer and the
reader disagreeing on the top-level key, which would have surfaced as a
`KeyError` at the *end* of an M1 run, hours after the run that produced the file.

Neither defect is visible to a unit test of either side. This module writes a
checkpoint through the real `lifetime` path and loads it through the reader's
own expectations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.germline import Genome
from soma.model.model import SomaModel
from soma.train import LifetimeConfig, lifetime

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def written(tiny_setup, tmp_path_factory):
    cfg, corpus, eval_batches, probe = tiny_setup
    root = tmp_path_factory.mktemp("ckpt")
    genome = Genome.initialise(
        cfg, generator=torch.Generator().manual_seed(1), lineage_id="ck"
    )
    lc = LifetimeConfig(
        cfg=cfg, run_id="ck", runs_root=str(root), device="cpu",
        corpus=corpus, eval_batches=eval_batches, probe_tokens=probe,
        L_star=4.0, slope_ref=1.0, rows_per_batch=2, micro_batches=1,
        eval_every=30, verbose=False, log_to_disk=True,
        save_final_state=True, max_steps=30,
    )
    lifetime(genome, lc, seed=1)
    return cfg, probe, Path(root) / "ck" / "checkpoints" / "final.pt"


def test_the_run_writes_the_file_its_report_names(written) -> None:
    """Audit A-70: the report recorded the path and nothing created the file."""
    _, _, path = written
    assert path.exists(), f"{path} was not written"
    assert path.stat().st_size > 0


def test_the_blob_uses_the_keys_the_evaluator_reads(written) -> None:
    """Audit A-71. The reader does `blob["model"]` and `blob.get("genome")`;
    a writer using any other key fails at load, at the end of a long run."""
    _, _, path = written
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in blob, sorted(blob)
    assert "genome" in blob and blob["genome"] is not None
    for k in ("config", "seed", "steps", "dense_sites"):
        assert k in blob, sorted(blob)


def test_the_checkpoint_carries_the_unit_populations(written) -> None:
    """The population is state, not derived: `p_u`, `g_u`, `live`, `m_u` and the
    anneal counters are buffers in the state dict, so a reload resumes the
    *same* population rather than a freshly seeded one."""
    cfg, _, path = written
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pop_keys = [k for k in blob["model"] if k.startswith("populations.")]
    assert len(pop_keys) >= 6 * cfg.L, len(pop_keys)
    for field in ("p_u", "g_u", "live", "m_u", "age_u"):
        assert any(k.endswith("." + field) for k in pop_keys), field


def test_a_reloaded_model_runs_a_forward_pass(written) -> None:
    """The end-to-end claim: load, and the phenotype works. A state dict that
    loads but cannot run is not a reference model."""
    cfg, probe, path = written
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SomaModel(cfg)
    model.load_state_dict(blob["model"])
    out = model(probe, genome=blob["genome"], want_metabolic=False)
    assert out.logits.shape == (probe.shape[0], probe.shape[1], cfg.V)
    assert torch.isfinite(out.logits).all()


def test_the_reload_reproduces_the_original_logits(written, tiny_setup) -> None:
    """Bit-for-bit on CPU: if the reload differed, `theta_m0`'s cached per-token
    losses would be a reference to a model that no longer exists."""
    cfg, probe, path = written
    blob = torch.load(path, map_location="cpu", weights_only=False)
    a = SomaModel(cfg)
    a.load_state_dict(blob["model"])
    b = SomaModel(cfg)
    b.load_state_dict(blob["model"])
    a.eval()
    b.eval()
    with torch.no_grad():
        la = a(probe, genome=blob["genome"], want_metabolic=False).logits
        lb = b(probe, genome=blob["genome"], want_metabolic=False).logits
    assert torch.equal(la, lb)
