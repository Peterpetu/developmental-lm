"""SOMA - Selected Ontogeny of Modular Architecture.

A language-model architecture whose parameters are partitioned by write
permission (germline vs soma) and in which a population of mortal, positioned
computational units is subjected to case-based selection inside a single
training run.

The normative specification is `docs/spec/00..11` at version 3.0. Every module
here carries `SPEC: nn §x.y` markers tying it to the section it implements;
`scripts/spec_trace.py` checks the mapping is total.
"""

from soma.config import CONFIGS, M1, M_NANO, SomaConfig, get_config

__all__ = ["SomaConfig", "M_NANO", "M1", "CONFIGS", "get_config"]
