"""ANLAGE -- a language model grown from a neural genome.

SPEC: 01 §1.1

The package is organised by the four parameter classes of `docs/anlage/01`:

``config``      the genome shape and the stage configurations of both lineages
``codes``       innate codes (pure functions of index; zero parameters)
``latents``     the latent bank `Z` and its prior
``genome``      `Gamma` and the neural-kernel decode `D`
``ops``         the functional operators of the phenotype
``model``       the phenotype: free weights per role, growth-aware
``recipe``      optimisers, schedules, the loss
``data``        streams, strata, common random numbers
``lifetime``    one lifetime (§06's loop)
``regress``     the generation-level consolidation (§05)
``grow``        growth events (§06)
``population``  siblings, lexicase, merging (§07)
"""

from __future__ import annotations

__version__ = "0.1.0"
