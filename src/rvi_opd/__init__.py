"""Core, framework-independent primitives for RvI-OPD experiments."""

from .budget import BudgetLedger, paired_repair_span
from .router import Action, RouteThresholds, calibrate_thresholds, route_state
from .signals import RawSignals, StateSignals, compute_raw_signals, robust_normalize

__all__ = [
    "Action",
    "BudgetLedger",
    "RawSignals",
    "RouteThresholds",
    "StateSignals",
    "calibrate_thresholds",
    "compute_raw_signals",
    "paired_repair_span",
    "robust_normalize",
    "route_state",
]
