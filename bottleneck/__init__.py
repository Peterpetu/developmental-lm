"""The genomic bottleneck of Shuvaev, Lachi, Koulakov & Zador (PNAS 2024), run as printed on
the 13M stage-0 language model of ANLAGE.

SPEC: docs/bottleneck/00-replication.md

`genome.py`  the paper's g-networks on this repository's codes (§1, §2, §4)
`loop.py`    intermittent training, Algorithm 1 with Eq. 3, as a lifetime hook (§3, §5)
"""

from .genome import BottleneckGenome, PairMLP, gray_rank_codes, index_codes, n_bits
from .loop import Intermittent, epsilon, regenerate, shuffle_rows
from .meta import AdvanceOnly, MetaIntermittent, MetaPool, TaskDistribution
from .transfer import (Source, initialise, layer_keys, load_source, task_eval_set, task_pool,
                       task_token_array)

__all__ = ["BottleneckGenome", "PairMLP", "gray_rank_codes", "index_codes", "n_bits",
           "Intermittent", "epsilon", "regenerate", "shuffle_rows",
           "AdvanceOnly", "MetaIntermittent", "MetaPool", "TaskDistribution",
           "Source", "initialise", "layer_keys", "load_source", "task_eval_set", "task_pool",
           "task_token_array"]
