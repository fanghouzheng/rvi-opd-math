import pytest

from rvi_opd.budget import BudgetLedger, paired_repair_span
from rvi_opd.metrics import paired_problem_bootstrap, residual_auc
from rvi_opd.router import Action, RouteThresholds, route_state
from rvi_opd.signals import StateSignals


def state(dl=0.0, di=0.0, s2=0.0):
    return StateSignals(dl + di, 0.5, dl, di, s2)


def test_router_prioritises_intervention():
    thresholds = RouteThresholds(repair=0.7, intervene=0.8)
    assert route_state(state(dl=0.9, s2=0.9), thresholds) is Action.INTERVENE
    assert route_state(state(di=0.9, s2=0.1), thresholds) is Action.REPAIR
    assert route_state(state(dl=0.1, s2=0.1), thresholds) is Action.DISCARD


def test_budget_span_and_validation():
    assert list(paired_repair_span(4, 3, 20)) == [4, 5, 6]
    assert list(paired_repair_span(4, 3, 5)) == [4]
    ledger = BudgetLedger(1, 2, 3)
    ledger.add(BudgetLedger(4, 5, 6))
    assert ledger == BudgetLedger(5, 7, 9)
    with pytest.raises(ValueError):
        BudgetLedger(-1, 0, 0).validate()


def test_problem_bootstrap_clusters_repeated_samples():
    estimate, low, high = paired_problem_bootstrap(
        ["p1", "p1", "p2", "p2"], [1, 1, 0, 0], [0, 0, 0, 0], samples=1000, seed=1
    )
    assert estimate == pytest.approx(0.5)
    assert low <= estimate <= high
    assert residual_auc([0.2, 0.4]) == pytest.approx(0.3)
