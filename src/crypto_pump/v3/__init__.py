"""V3 execution-aware research and shadow-trading components."""

from .labels import TripleBarrierSpec, add_triple_barrier_labels
from .execution import ExecutionCostModel, ExecutionBacktester
from .risk import RiskLimits, RiskManager, RiskDecision
from .shadow import ShadowExecutor
from .metrics import financial_report, stress_test_trades

__all__ = [
    "TripleBarrierSpec",
    "add_triple_barrier_labels",
    "ExecutionCostModel",
    "ExecutionBacktester",
    "RiskLimits",
    "RiskManager",
    "RiskDecision",
    "ShadowExecutor",
    "financial_report",
    "stress_test_trades",
]
