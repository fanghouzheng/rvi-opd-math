from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .signals import StateSignals


class Action(str, Enum):
    REPAIR = "repair"
    INTERVENE = "intervene"
    DISCARD = "discard"


@dataclass(frozen=True)
class RouteThresholds:
    repair: float
    intervene: float


def calibrate_thresholds(
    states: Sequence[StateSignals],
    *,
    repair_quantile: float = 0.90,
    intervene_quantile: float = 0.90,
) -> RouteThresholds:
    """Freeze thresholds from calibration states only; never call this on evaluation data."""

    if not states:
        raise ValueError("at least one calibration state is required")
    if not 0 < repair_quantile < 1 or not 0 < intervene_quantile < 1:
        raise ValueError("quantiles must lie strictly between 0 and 1")
    repair_scores = np.asarray([max(s.d_learnable, s.d_incompatible) for s in states])
    s2_scores = np.asarray([s.epistemic_mass for s in states])
    return RouteThresholds(
        repair=float(np.quantile(repair_scores, repair_quantile)),
        intervene=float(np.quantile(s2_scores, intervene_quantile)),
    )


def route_state(state: StateSignals, thresholds: RouteThresholds) -> Action:
    """High state damage has priority over local absorbability."""

    if state.epistemic_mass >= thresholds.intervene:
        return Action.INTERVENE
    if max(state.d_learnable, state.d_incompatible) >= thresholds.repair:
        return Action.REPAIR
    return Action.DISCARD
