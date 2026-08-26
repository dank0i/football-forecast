"""Reference forecasts every real model has to beat.

The original analysis reported ~65% accuracy with no baseline attached, which
makes the number impossible to interpret. These make the bar explicit, and the
third one is the bar that actually matters: a bookmaker's published prices are a
liquid, incentive-backed forecast produced by people with more data than this
database contains. Beating the market is the real test; beating a coin flip is
not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DENSE_BOOKMAKERS, REFERENCE_BOOKMAKER
from ..features import market


def prior_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict the training-set outcome frequencies for every match."""
    counts = train["result"].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0.0)
    return np.tile(counts.to_numpy(dtype=float), (len(test), 1))


def home_baseline(test: pd.DataFrame) -> np.ndarray:
    """Always predict a home win with certainty.

    Included because it is the implicit baseline behind "the home team wins 46%
    of the time", and it scores catastrophically on any proper scoring rule,
    which is the point.
    """
    probs = np.zeros((len(test), 3))
    probs[:, 0] = 1.0
    return probs


def market_baseline(test: pd.DataFrame, book: str = REFERENCE_BOOKMAKER, method: str = "shin") -> np.ndarray:
    return market.bookmaker_probabilities(test, book=book, method=method)


def consensus_baseline(
    test: pd.DataFrame, books: tuple[str, ...] = DENSE_BOOKMAKERS, method: str = "shin"
) -> np.ndarray:
    return market.consensus_probabilities(test, books=books, method=method)
