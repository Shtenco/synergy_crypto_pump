"""V3 execution-aware research, live-data and shadow-trading components."""

from .combined_oos import (
    combine_oos_predictions,
    combine_oos_trades,
    combined_oos_report,
    pooled_ml_report,
)
from .execution import ExecutionBacktester, ExecutionCostModel
from .labels import TripleBarrierSpec, add_triple_barrier_labels
from .live import ShadowSignalEngine, ShadowStrategyConfig
from .metrics import financial_report, stress_test_trades
from .monitor import HealthReport, HealthThresholds, health_report, population_stability_index
from .panel import PanelConfig, normalize_pumpportal_trades, trades_to_panel
from .risk import RiskDecision, RiskLimits, RiskManager
from .shadow import ShadowExecutor
from .store import V3Store
from .stream import PumpStreamCollector

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
    "combine_oos_trades",
    "combine_oos_predictions",
    "pooled_ml_report",
    "combined_oos_report",
    "V3Store",
    "PumpStreamCollector",
    "PanelConfig",
    "normalize_pumpportal_trades",
    "trades_to_panel",
    "HealthThresholds",
    "HealthReport",
    "health_report",
    "population_stability_index",
    "ShadowStrategyConfig",
    "ShadowSignalEngine",
]
