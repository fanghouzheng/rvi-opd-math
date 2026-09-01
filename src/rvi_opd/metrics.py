from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np


def paired_problem_bootstrap(
    problem_ids: Sequence[str],
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Problem-clustered paired bootstrap CI for a treatment-control delta."""

    if not (len(problem_ids) == len(treatment) == len(control)) or not problem_ids:
        raise ValueError("inputs must be non-empty and have equal length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for pid, t, c in zip(problem_ids, treatment, control, strict=True):
        grouped[pid].append(float(t) - float(c))
    cluster_means = np.asarray([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = rng.choice(cluster_means, size=(samples, cluster_means.size), replace=True).mean(axis=1)
    estimate = float(cluster_means.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def residual_auc(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite sequence")
    return float(array.mean())
