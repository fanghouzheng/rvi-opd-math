import numpy as np
import pytest

from rvi_opd.signals import combine_normalised_signals, compute_raw_signals, robust_normalize


def test_raw_signals_have_expected_support_behavior():
    teacher = [0.05, 0.10, 0.80, 0.05]
    student = [0.70, 0.20, 0.05, 0.05]
    result = compute_raw_signals(teacher, student, top_k=2, epistemic_token_ids=[2])
    assert result.coverage == pytest.approx(0.15)
    assert result.epistemic_mass == pytest.approx(0.80)
    assert result.disagreement > 0


def test_normalization_is_bounded_and_handles_constant_vector():
    values = robust_normalize([0, 1, 2, 3, 100])
    assert np.all((0 <= values) & (values <= 1))
    assert np.array_equal(robust_normalize([2, 2, 2]), np.zeros(3))


def test_decomposition_sums_to_disagreement():
    states = combine_normalised_signals([1, 2, 4], [0.1, 0.6, 0.9], [0.2, 0.3, 0.8])
    for state in states:
        assert state.d_learnable + state.d_incompatible == pytest.approx(state.disagreement)
