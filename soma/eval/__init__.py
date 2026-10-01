"""Evaluation: metrics, Q_route, the behavioural suite, the kill criteria.

SPEC: 08 §8.1
SPEC: 10 §10.1
"""

from soma.eval.metrics import (
    CurveTracker,
    EvalPoint,
    Fitness,
    bits_per_byte,
    compute_fitness,
    erank_probe,
    held_out_ce,
)
from soma.eval.killcriteria import KillVerdict, Verdicts, evaluate_all
from soma.eval.qroute import barber_modularity, brim, q_route
from soma.eval.regression import SuiteResult, greedy_decode, run_suite, solve_rate

__all__ = [
    "CurveTracker", "EvalPoint", "Fitness", "compute_fitness",
    "held_out_ce", "erank_probe", "bits_per_byte",
    "q_route", "barber_modularity", "brim",
    "evaluate_all", "KillVerdict", "Verdicts",
    "run_suite", "solve_rate", "greedy_decode", "SuiteResult",
]
