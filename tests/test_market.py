"""Devigging correctness, including a regression test for the original bug."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_forecast.evaluation.replication import correct_odds_normalisation, original_odds_normalisation
from football_forecast.features.market import (
    devig,
    devig_multiplicative,
    devig_power,
    devig_shin,
    odds_to_raw,
)

FAIR_ODDS = np.array([[3.0, 3.0, 3.0]])
TYPICAL = np.array([[2.20, 3.40, 3.30], [1.30, 5.50, 11.0], [4.50, 3.60, 1.85]])


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devigged_probabilities_sum_to_one(method: str):
    probs = devig(TYPICAL, method=method)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devigged_probabilities_are_valid(method: str):
    probs = devig(TYPICAL, method=method)
    assert (probs > 0).all() and (probs < 1).all()


def test_margin_free_odds_are_unchanged():
    """Odds with no overround already are a distribution."""
    np.testing.assert_allclose(devig_multiplicative(FAIR_ODDS), 1 / 3, atol=1e-9)


def test_devigging_preserves_favourite_ordering():
    for method in (devig_multiplicative, devig_power, devig_shin):
        probs = method(TYPICAL)
        assert (np.argsort(probs, axis=1) == np.argsort(odds_to_raw(TYPICAL), axis=1)).all()


def test_shin_shades_the_longshot_more_than_multiplicative():
    """Shin's correction should take proportionally more off the longest price."""
    lopsided = np.array([[1.20, 7.0, 15.0]])
    multiplicative = devig_multiplicative(lopsided)[0]
    shin = devig_shin(lopsided)[0]
    assert shin[2] < multiplicative[2]
    assert shin[0] > multiplicative[0]


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown devig method"):
        devig(TYPICAL, method="nonsense")


def test_original_normalisation_bug_is_reproduced():
    """Regression guard for the in-place overwrite in the source notebook.

    The first assignment replaces the home column with a probability, and the
    next two then take its reciprocal as though it were still odds. The result
    is a triple that does not sum to 1, with the away outcome collapsed by an
    order of magnitude. This test pins the behaviour so the corrected path
    cannot silently drift back into it.
    """
    frame = pd.DataFrame({"B365H": [2.20], "B365D": [3.40], "B365A": [3.30]})
    for column in ("B365H", "B365D", "B365A"):
        frame[column] = frame[column].astype(float)

    wrong = original_odds_normalisation(frame)[["B365H", "B365D", "B365A"]].to_numpy()[0]
    right = correct_odds_normalisation(frame)[["B365H", "B365D", "B365A"]].to_numpy()[0]

    assert wrong.sum() < 0.75
    assert right.sum() == pytest.approx(1.0)
    assert wrong[0] == pytest.approx(right[0])  # home is computed first, so it survives
    assert wrong[2] < right[2] / 5  # away is destroyed
