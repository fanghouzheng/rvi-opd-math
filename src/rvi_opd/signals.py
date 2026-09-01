from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawSignals:
    """Unnormalised per-state diagnostics."""

    disagreement: float
    coverage: float
    epistemic_mass: float


@dataclass(frozen=True)
class StateSignals:
    """Batch-normalised signals used by the router."""

    disagreement: float
    coverage: float
    d_learnable: float
    d_incompatible: float
    epistemic_mass: float


def _as_probabilities(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def compute_raw_signals(
    teacher_probs: Sequence[float],
    student_probs: Sequence[float],
    *,
    top_k: int,
    epistemic_token_ids: Iterable[int],
    eps: float = 1e-12,
) -> RawSignals:
    """Compute TA-OPD disagreement/coverage and TRD epistemic-onset mass.

    `teacher_probs` and `student_probs` are full-vocabulary probabilities in the
    same token order. Disagreement is forward KL after renormalising both models
    on the union of their top-K supports. Coverage is the *full* teacher mass on
    the student's top-K support.
    """

    teacher = _as_probabilities(teacher_probs, "teacher_probs")
    student = _as_probabilities(student_probs, "student_probs")
    if teacher.shape != student.shape:
        raise ValueError("teacher_probs and student_probs must have equal length")
    if not 1 <= top_k <= teacher.size:
        raise ValueError("top_k must lie in [1, vocabulary_size]")

    student_top = np.argpartition(student, -top_k)[-top_k:]
    teacher_top = np.argpartition(teacher, -top_k)[-top_k:]
    union = np.union1d(student_top, teacher_top)
    t_union = teacher[union]
    s_union = student[union]
    t_union = t_union / t_union.sum()
    s_union = s_union / s_union.sum()
    disagreement = float(np.sum(t_union * (np.log(t_union + eps) - np.log(s_union + eps))))
    coverage = float(teacher[student_top].sum())

    epi_ids = np.asarray(sorted(set(epistemic_token_ids)), dtype=np.int64)
    if epi_ids.size and (np.any(epi_ids < 0) or np.any(epi_ids >= teacher.size)):
        raise ValueError("epistemic token id is outside the vocabulary")
    epistemic_mass = float(teacher[epi_ids].sum()) if epi_ids.size else 0.0
    return RawSignals(disagreement, coverage, epistemic_mass)


def robust_normalize(
    values: Sequence[float], low_q: float = 0.05, high_q: float = 0.95
) -> np.ndarray:
    """TA-OPD-style quantile normalisation with deterministic degenerate handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite 1D sequence")
    if not 0 <= low_q < high_q <= 1:
        raise ValueError("quantiles must satisfy 0 <= low_q < high_q <= 1")
    low, high = np.quantile(array, [low_q, high_q])
    width = float(high - low)
    if width <= np.finfo(np.float64).eps:
        return np.zeros_like(array)
    return np.clip((array - low) / width, 0.0, 1.0)


def combine_normalised_signals(
    disagreement: Sequence[float], coverage: Sequence[float], epistemic_mass: Sequence[float]
) -> list[StateSignals]:
    if not (len(disagreement) == len(coverage) == len(epistemic_mass)):
        raise ValueError("all signal vectors must have equal length")
    d_norm = robust_normalize(disagreement)
    c_norm = robust_normalize(coverage)
    s2_norm = robust_normalize(epistemic_mass)
    return [
        StateSignals(float(d), float(c), float(d * c), float(d * (1 - c)), float(s2))
        for d, c, s2 in zip(d_norm, c_norm, s2_norm, strict=True)
    ]
