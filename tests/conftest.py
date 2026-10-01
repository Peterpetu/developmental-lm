"""Suite-wide determinism.

SPEC: 02 §2.6.3
SPEC: 07 §7.7

`nn.Linear.reset_parameters`, `nn.Embedding.reset_parameters` and
`nn.init.normal_` accept no `generator`, so **every** `SomaModel(cfg)` in the
suite draws its weights from the process-global stream (audit A-64). Twenty-one
such constructions exist across the test modules, and none of them seeded it.

The consequence is not theoretical. Running the gate ledger produced a single
failure of `G-GRAD` that did not reproduce in fourteen subsequent runs — a test
that had passed on every model it had happened to be given, and failed on one it
had not. A suite whose passes are not reproducible is not evidence, whatever its
green count says, and the defect it hides is exactly the one it is least able to
find: an assertion with a tolerance, holding for most initialisations.

Seeding here rather than in twenty-one call sites is deliberate. A convention
that must be remembered at every construction will be forgotten at one of them,
and the failure is silent — a passing test, on a model nobody chose.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

SUITE_SEED = 20260828


@pytest.fixture(autouse=True)
def _deterministic_global_streams():
    """Reset every process-global stream before each test.

    Autouse and unconditional: a test that wants a *different* model builds one
    with an explicit generator, which is unaffected by this. What it removes is
    the case where a test builds a model with **no** generator and is therefore
    handed whatever the previous test left behind.
    """
    torch.manual_seed(SUITE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SUITE_SEED)
    np.random.seed(SUITE_SEED)
    random.seed(SUITE_SEED)
    yield
