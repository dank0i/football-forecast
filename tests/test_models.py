"""Model-level invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_forecast.models.dixon_coles import fit_dixon_coles


def test_score_matrix_is_a_distribution(synthetic_matches: pd.DataFrame):
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    matrix = fit.score_matrix(1, 4)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_outcome_probabilities_sum_to_one(synthetic_matches: pd.DataFrame):
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    assert fit.outcome_probabilities(1, 4).sum() == pytest.approx(1.0)


def test_stronger_team_is_favoured(synthetic_matches: pd.DataFrame):
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    strong_at_home = fit.outcome_probabilities(1, 4)
    weak_at_home = fit.outcome_probabilities(4, 1)
    assert strong_at_home[0] > strong_at_home[2]
    assert weak_at_home[2] > weak_at_home[0]


def test_home_advantage_is_positive(synthetic_matches: pd.DataFrame):
    """The fixture generates home teams with a 1.3x scoring multiplier."""
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    assert fit.gamma > 0


def test_same_team_both_sides_is_a_coin_flip_plus_home_edge(synthetic_matches: pd.DataFrame):
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    probs = fit.outcome_probabilities(2, 2)
    assert probs[0] > probs[2]


def test_unknown_team_returns_nan(synthetic_matches: pd.DataFrame):
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    assert np.isnan(fit.outcome_probabilities(999, 1)).all()


def test_derived_markets_are_consistent_with_the_matrix(synthetic_matches: pd.DataFrame):
    """Over-2.5 and the complement of under-2.5 must agree with the matrix."""
    fit = fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2012-01-01"))
    matrix = fit.score_matrix(1, 4)
    derived = fit.derived_markets(1, 4)
    goals = np.arange(matrix.shape[0])
    totals = goals[:, None] + goals[None, :]
    assert derived["over_2_5"] == pytest.approx(matrix[totals > 2].sum())
    assert derived["nil_nil"] == pytest.approx(matrix[0, 0])
    assert derived["btts"] == pytest.approx(matrix[1:, 1:].sum())


def test_time_decay_shifts_weight_toward_recent_form(synthetic_matches: pd.DataFrame):
    """A large decay must produce a different fit from no decay at all."""
    as_of = pd.Timestamp("2012-01-01")
    flat = fit_dixon_coles(synthetic_matches, as_of=as_of, xi=0.0)
    decayed = fit_dixon_coles(synthetic_matches, as_of=as_of, xi=0.01)
    assert not np.allclose(flat.attack, decayed.attack, atol=1e-6)


def test_fit_before_any_data_raises(synthetic_matches: pd.DataFrame):
    with pytest.raises(ValueError, match="no matches before"):
        fit_dixon_coles(synthetic_matches, as_of=pd.Timestamp("2000-01-01"))
