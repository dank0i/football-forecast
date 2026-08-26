"""Invariants for the team-embedding network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_forecast.models.embedding_net import UNKNOWN_TEAM, EmbeddingForecaster

FEATURES = ["elo_diff", "elo_home", "elo_away"]


@pytest.fixture
def prepared(synthetic_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from football_forecast.features.elo import compute_elo

    frame = synthetic_matches.merge(compute_elo(synthetic_matches), on="match_index")
    cut = int(len(frame) * 0.8)
    return frame.iloc[:cut], frame.iloc[cut:]


def test_predictions_are_a_distribution(prepared):
    train, test = prepared
    model = EmbeddingForecaster(features=FEATURES, max_epochs=5).fit(train.iloc[:-40], train.iloc[-40:])
    probs = model.predict(test)
    assert probs.shape == (len(test), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_unseen_team_falls_back_to_the_unknown_slot(prepared):
    """A club absent from training must predict, not crash.

    Promotion guarantees this happens in a real walk-forward run, and silently
    indexing past the embedding table would either raise or read a neighbouring
    club's vector.
    """
    train, test = prepared
    model = EmbeddingForecaster(features=FEATURES, max_epochs=5).fit(train.iloc[:-40], train.iloc[-40:])
    stranger = test.copy()
    stranger["home_team_api_id"] = 999999
    probs = model.predict(stranger)
    assert np.isfinite(probs).all()
    assert model.team_index.get(999999) is None


def test_preprocessing_is_fitted_on_training_data_only(prepared):
    """Scaler statistics must not move when the test frame changes.

    This is the neural equivalent of the imputation leak in the original
    analysis, where ``fillna(X.mean())`` was computed over the full frame before
    splitting.
    """
    train, test = prepared
    model = EmbeddingForecaster(features=FEATURES, max_epochs=3).fit(train.iloc[:-40], train.iloc[-40:])
    mean_before, scale_before = model.mean.copy(), model.scale.copy()
    wild = test.copy()
    wild[FEATURES] = wild[FEATURES] * 1000
    model.predict(wild)
    np.testing.assert_array_equal(model.mean, mean_before)
    np.testing.assert_array_equal(model.scale, scale_before)


def test_missing_features_are_handled(prepared):
    train, test = prepared
    train = train.copy()
    train.loc[train.index[:50], "elo_diff"] = np.nan
    model = EmbeddingForecaster(features=FEATURES, max_epochs=3).fit(train.iloc[:-40], train.iloc[-40:])
    holey = test.copy()
    holey.loc[holey.index[:10], "elo_diff"] = np.nan
    assert np.isfinite(model.predict(holey)).all()


def test_team_vectors_cover_every_trained_club(prepared):
    train, _ = prepared
    model = EmbeddingForecaster(features=FEATURES, max_epochs=3).fit(train.iloc[:-40], train.iloc[-40:])
    vectors = model.team_vectors()
    assert len(vectors) == len(model.team_index)
    assert UNKNOWN_TEAM not in set(model.team_index.values())


def test_predict_before_fit_raises():
    model = EmbeddingForecaster(features=FEATURES)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(pd.DataFrame({"home_team_api_id": [1], "away_team_api_id": [2], "elo_diff": [0.0]}))
