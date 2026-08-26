"""Scoring-rule properties, checked against values derived by hand."""

from __future__ import annotations

import numpy as np
import pytest

from pitchcast.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    ranked_probability_score,
    skill_score,
)


def test_perfect_forecast_scores_zero():
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    actual = np.array([0, 1, 2])
    assert ranked_probability_score(probs, actual) == pytest.approx(0.0)
    assert brier_score(probs, actual) == pytest.approx(0.0)


def test_rps_respects_outcome_ordering():
    """Predicting an away win when home wins must cost more than predicting a draw.

    This is the property that makes RPS the right metric for football and
    distinguishes it from log loss, which treats the three classes as unordered.
    """
    actual = np.array([0])
    near_miss = ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), actual)
    far_miss = ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), actual)
    assert far_miss > near_miss


def test_rps_known_value():
    # Certain draw forecast, home win occurs: cumulative preds (0, 1) vs (1, 1).
    assert ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), np.array([0])) == pytest.approx(0.5)


def test_log_loss_matches_definition():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert log_loss(probs, np.array([0])) == pytest.approx(-np.log(0.5))


def test_confident_and_wrong_is_punished_but_finite():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert np.isfinite(log_loss(probs, np.array([2])))


def test_calibration_error_zero_for_calibrated_forecast():
    rng = np.random.default_rng(0)
    probs = np.tile([0.5, 0.25, 0.25], (40000, 1))
    actual = rng.choice(3, size=40000, p=[0.5, 0.25, 0.25])
    assert expected_calibration_error(probs, actual) < 0.01


def test_skill_score_sign():
    actual = np.array([0, 0, 0, 0])
    good = np.tile([0.9, 0.05, 0.05], (4, 1))
    poor = np.tile([0.34, 0.33, 0.33], (4, 1))
    assert skill_score(good, poor, actual) > 0
    assert skill_score(poor, good, actual) < 0


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        ranked_probability_score(np.array([[0.5, 0.5]]), np.array([0]))
    with pytest.raises(ValueError):
        ranked_probability_score(np.array([[0.3, 0.3, 0.4]]), np.array([0, 1]))
