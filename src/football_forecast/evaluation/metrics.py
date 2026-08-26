"""Scoring rules for three-way match forecasts.

Accuracy is the wrong headline metric for this problem and it is worth being
precise about why. Football outcomes are irreducibly uncertain: even a perfect
forecaster would be "wrong" on most matches, because the true probability of
the most likely outcome is often only 45-55%. Accuracy also throws away
everything except the argmax, so a model that says 34/33/33 and a model that
says 90/5/5 score identically when the favourite wins, and it cannot distinguish
a confident correct call from a lucky one.

The metrics here judge the whole distribution:

* **RPS** (ranked probability score) is the standard in football forecasting.
  It respects the natural ordering home > draw > away, so predicting an away win
  when the home side wins is penalised more than predicting a draw. Lower is
  better; a perfect forecast scores 0.
* **Log loss** is the strictly proper scoring rule, brutal about confident
  mistakes. Reported alongside RPS because they occasionally disagree.
* **Brier score** is the multiclass squared error, less sensitive to tail events.
* **Calibration** asks a different question from all three: when the model says
  60%, does it happen 60% of the time? A model can rank well and still be badly
  calibrated, and calibration is what matters if the probabilities are ever used
  to size a bet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def _check(probs: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    actual = np.asarray(actual, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"probs must be (n, 3), got {probs.shape}")
    if len(probs) != len(actual):
        raise ValueError(f"length mismatch: {len(probs)} probs vs {len(actual)} outcomes")
    return probs, actual


def ranked_probability_score(probs: np.ndarray, actual: np.ndarray) -> float:
    """Mean RPS over the ordered outcomes (home, draw, away)."""
    probs, actual = _check(probs, actual)
    onehot = np.eye(3)[actual]
    cum_pred = np.cumsum(probs, axis=1)[:, :2]
    cum_true = np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1) / 2.0))


def log_loss(probs: np.ndarray, actual: np.ndarray) -> float:
    probs, actual = _check(probs, actual)
    picked = probs[np.arange(len(actual)), actual]
    return float(-np.mean(np.log(np.clip(picked, EPS, 1.0))))


def brier_score(probs: np.ndarray, actual: np.ndarray) -> float:
    probs, actual = _check(probs, actual)
    onehot = np.eye(3)[actual]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, actual: np.ndarray) -> float:
    probs, actual = _check(probs, actual)
    return float(np.mean(probs.argmax(axis=1) == actual))


def expected_calibration_error(probs: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Weighted gap between predicted confidence and observed frequency.

    Computed over all three outcomes flattened, so it measures the calibration
    of the probabilities themselves rather than only the top-1 confidence.
    """
    probs, actual = _check(probs, actual)
    onehot = np.eye(3)[actual].ravel()
    flat = probs.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(flat, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.mean() * abs(flat[mask].mean() - onehot[mask].mean())
    return float(error)


def calibration_table(probs: np.ndarray, actual: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability data: predicted vs observed frequency per confidence bin."""
    probs, actual = _check(probs, actual)
    onehot = np.eye(3)[actual].ravel()
    flat = probs.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(flat, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(mask.sum()),
                "predicted": float(flat[mask].mean()),
                "observed": float(onehot[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(probs: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """All headline metrics for one set of forecasts."""
    probs, actual = _check(probs, actual)
    valid = np.isfinite(probs).all(axis=1)
    if not valid.any():
        return dict.fromkeys(("n", "rps", "log_loss", "brier", "accuracy", "ece"), np.nan)
    probs, actual = probs[valid], actual[valid]
    # Renormalise defensively: a model that emits probabilities summing to
    # 0.999 would otherwise be silently rewarded by log loss.
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "n": len(actual),
        "rps": ranked_probability_score(probs, actual),
        "log_loss": log_loss(probs, actual),
        "brier": brier_score(probs, actual),
        "accuracy": accuracy(probs, actual),
        "ece": expected_calibration_error(probs, actual),
    }


def skill_score(probs: np.ndarray, reference: np.ndarray, actual: np.ndarray) -> float:
    """Fractional RPS improvement over a reference forecast.

    Positive means better than the reference; 0 means indistinguishable. This is
    the number that matters when the reference is the betting market.
    """
    valid = np.isfinite(probs).all(axis=1) & np.isfinite(reference).all(axis=1)
    model = ranked_probability_score(probs[valid], actual[valid])
    base = ranked_probability_score(reference[valid], actual[valid])
    return float((base - model) / base) if base > 0 else np.nan
