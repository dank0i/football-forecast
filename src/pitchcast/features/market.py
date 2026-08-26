"""Turn bookmaker odds into probabilities.

Decimal odds do not sum to a probability distribution: the implied
probabilities sum to roughly 1.06, and that 6% excess is the bookmaker's
margin ("overround", or vig). Removing it is called devigging, and *how* you
remove it changes the resulting probabilities by more than most feature
engineering does, because the margin is not spread evenly across outcomes.
Longshots are systematically overpriced, so a flat normalisation leaves the
favourite underrated and the longshot overrated. Three methods are implemented
so the choice can be measured rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ..config import DENSE_BOOKMAKERS, REFERENCE_BOOKMAKER

Method = str


def odds_to_raw(odds: np.ndarray) -> np.ndarray:
    """Decimal odds -> raw implied probabilities (which sum to > 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / np.asarray(odds, dtype=float)
    return raw


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    """Scale raw probabilities down until they sum to 1.

    The naive method. Assumes the margin is proportional to each outcome's
    probability, which is known to be wrong, but it is the standard reference.
    """
    raw = odds_to_raw(odds)
    return raw / raw.sum(axis=1, keepdims=True)


def devig_power(odds: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Find k such that sum(raw_i ** k) == 1.

    Applies more correction to longshots than to favourites, which matches the
    observed shape of the favourite-longshot bias.
    """
    raw = odds_to_raw(odds)
    out = np.empty_like(raw)
    for i, row in enumerate(raw):
        if not np.isfinite(row).all():
            out[i] = np.nan
            continue

        def gap(k: float, row: np.ndarray = row) -> float:
            return float(np.sum(row**k) - 1.0)

        try:
            k = optimize.brentq(gap, 0.5, 2.0, xtol=tol)
            out[i] = row**k
        except ValueError:
            out[i] = row / row.sum()
    return out


def devig_shin(odds: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Shin (1993): price the margin as protection against informed bettors.

    Models the book as facing a proportion z of insiders, and inverts the
    resulting quoted prices back to the bookmaker's true beliefs. Standard in
    the forecasting literature and usually the best-calibrated of the three.
    """
    raw = odds_to_raw(odds)
    out = np.empty_like(raw)
    for i, row in enumerate(raw):
        if not np.isfinite(row).all():
            out[i] = np.nan
            continue
        total = row.sum()

        def gap(z: float, row: np.ndarray = row, total: float = total) -> float:
            p = (np.sqrt(z**2 + 4 * (1 - z) * row**2 / total) - z) / (2 * (1 - z))
            return float(p.sum() - 1.0)

        try:
            z = optimize.brentq(gap, 1e-9, 0.35, xtol=tol)
            p = (np.sqrt(z**2 + 4 * (1 - z) * row**2 / total) - z) / (2 * (1 - z))
            out[i] = p / p.sum()
        except ValueError:
            out[i] = row / row.sum()
    return out


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds: np.ndarray, method: Method = "shin") -> np.ndarray:
    if method not in DEVIG_METHODS:
        raise ValueError(f"unknown devig method {method!r}; choose from {sorted(DEVIG_METHODS)}")
    return DEVIG_METHODS[method](odds)


def bookmaker_probabilities(
    matches: pd.DataFrame, book: str = REFERENCE_BOOKMAKER, method: Method = "shin"
) -> np.ndarray:
    """Devigged H/D/A probabilities for one bookmaker; NaN rows where odds are missing."""
    cols = [book + s for s in "HDA"]
    odds = matches[cols].to_numpy(dtype=float)
    probs = np.full_like(odds, np.nan)
    complete = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    if complete.any():
        probs[complete] = devig(odds[complete], method=method)
    return probs


def consensus_probabilities(
    matches: pd.DataFrame,
    books: tuple[str, ...] = DENSE_BOOKMAKERS,
    method: Method = "shin",
) -> np.ndarray:
    """Average the devigged probabilities across the densely-covered bookmakers.

    Devig first, then average. Averaging odds first and devigging once would
    blend prices carrying different margins, which biases the result toward
    whichever book is greediest.
    """
    stack, weights = [], []
    for book in books:
        probs = bookmaker_probabilities(matches, book=book, method=method)
        stack.append(np.nan_to_num(probs, nan=0.0))
        weights.append(np.isfinite(probs).all(axis=1).astype(float))
    total = np.sum(weights, axis=0)
    summed = np.sum(stack, axis=0)
    out = np.full(summed.shape, np.nan)
    ok = total > 0
    out[ok] = summed[ok] / total[ok, None]
    return out


def best_available_odds(matches: pd.DataFrame, books: tuple[str, ...] = DENSE_BOOKMAKERS) -> np.ndarray:
    """Highest decimal odds offered on each outcome across books.

    A bettor shops for the best line, so a staking simulation that assumes a
    single book understates achievable returns. NaNs are ignored per outcome.
    """
    stacked = np.stack([matches[[b + s for s in "HDA"]].to_numpy(dtype=float) for b in books], axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmax(stacked, axis=0)


def overround(matches: pd.DataFrame, book: str = REFERENCE_BOOKMAKER) -> np.ndarray:
    odds = matches[[book + s for s in "HDA"]].to_numpy(dtype=float)
    return odds_to_raw(odds).sum(axis=1)
