"""Selection: epsilon-lexicase over units, with tokens as cases.

SPEC: 04 §4.1
"""

from soma.selection.buffer import CaseBatch, CaseReservoir
from soma.selection.credit import CreditAudit, case_weights, spearman
from soma.selection.lexicase import (
    LexicaseResult,
    case_win_histogram,
    credit_matrix,
    epsilon_lexicase,
    epsilon_per_case,
    mean_utility_parents,
)
from soma.selection.select import SelectionEvent, select

__all__ = [
    "CaseReservoir",
    "CaseBatch",
    "case_weights",
    "CreditAudit",
    "spearman",
    "credit_matrix",
    "epsilon_per_case",
    "epsilon_lexicase",
    "mean_utility_parents",
    "case_win_histogram",
    "LexicaseResult",
    "select",
    "SelectionEvent",
]
