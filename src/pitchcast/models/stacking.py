"""Blend several forecasters through a meta-learner.

The component models make different mistakes. Dixon-Coles knows about scoring
rates and handles draws structurally but sees nothing except goals; the boosted
model sees squads, form and history but has to learn the shape of a football
result from scratch; Elo is a crude single number that is nonetheless hard to
beat for its cost. Averaging them fixed-weight is leaving value on the table,
because the right weighting is not uniform and is not obvious.

A multinomial logistic regression over their out-of-fold predictions learns the
weights. It is deliberately a weak learner: with three inputs and a proper
scoring rule as the target, anything more flexible would start fitting the
meta-features rather than combining them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-9


def _logit(probs: np.ndarray) -> np.ndarray:
    """Log-probabilities make a linear blend behave like a weighted geometric mean.

    Blending raw probabilities linearly pulls everything toward the mean and
    blunts confident, correct forecasts; blending in log space preserves them.
    """
    return np.log(np.clip(probs, EPS, 1.0))


@dataclass
class Stacker:
    """Multinomial blend of several three-way forecasts.

    scikit-learn dropped the ``multi_class`` argument in 1.9; multinomial is now
    the default for a multiclass target, so it is no longer passed explicitly.
    """

    sources: list[str]
    meta: LogisticRegression = field(default_factory=lambda: LogisticRegression(C=1.0, max_iter=2000))

    def _design(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([_logit(frames[name]) for name in self.sources])

    def fit(self, frames: dict[str, np.ndarray], actual: np.ndarray) -> Stacker:
        design = self._design(frames)
        valid = np.isfinite(design).all(axis=1)
        self.meta.fit(design[valid], actual[valid])
        return self

    def predict(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        design = self._design(frames)
        out = np.full((len(design), 3), np.nan)
        valid = np.isfinite(design).all(axis=1)
        if valid.any():
            out[valid] = self.meta.predict_proba(design[valid])
        # Where a component is missing, fall back to the first source that has a
        # forecast rather than dropping the match entirely.
        for name in self.sources:
            missing = ~np.isfinite(out).all(axis=1)
            if not missing.any():
                break
            candidate = frames[name]
            usable = missing & np.isfinite(candidate).all(axis=1)
            out[usable] = candidate[usable]
        return out


def blend_weights(stacker: Stacker) -> pd.DataFrame:
    """Learned coefficients, for reporting which component earned its place."""
    if stacker.meta is None:
        raise RuntimeError("fit() must be called first")
    coefs = stacker.meta.coef_
    rows = []
    for i, name in enumerate(stacker.sources):
        block = coefs[:, i * 3 : (i + 1) * 3]
        rows.append({"source": name, "mean_abs_coef": float(np.abs(block).mean())})
    return pd.DataFrame(rows).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)
