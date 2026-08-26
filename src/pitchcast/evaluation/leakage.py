"""Quantify what a random train/test split is worth on chronological data.

The original analysis used ``train_test_split(X, y, test_size=0.2,
random_state=42)`` across eight seasons. This module runs the same model under
both protocols so the difference is a measured number rather than an assertion.

Two distinct effects are separated:

``random_split``      shuffled split, features built leak-free. Isolates the
                      damage from the split alone.
``random_plus_impute`` shuffled split *and* ``fillna(X.mean())`` computed over
                      the full frame before splitting, which is what the
                      original did. Isolates the additional damage from letting
                      test-set statistics reach the imputer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..evaluation.metrics import evaluate
from ..models.gbm import GBMForecaster


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    params = dict(GBMForecaster.__dataclass_fields__["params"].default_factory())
    params["seed"] = seed
    cutoff = int(len(train) * 0.85)
    model = GBMForecaster(features=features, params=params).fit(train.iloc[:cutoff], train.iloc[cutoff:])
    return model.predict(test)


def compare_protocols(
    frame: pd.DataFrame, features: list[str], seed: int = 42, test_size: float = 0.2
) -> pd.DataFrame:
    """Score identical features under leaky and honest evaluation protocols."""
    frame = frame.dropna(subset=["result"]).reset_index(drop=True)
    rows = []

    # 1. Random split, features untouched.
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, shuffle=True)
    probs = _fit_predict(train.reset_index(drop=True), test, features, seed)
    rows.append({"protocol": "random_split", **evaluate(probs, test["result"].to_numpy())})

    # 2. Random split with imputation fitted on everything, as in the original.
    leaked = frame.copy()
    leaked[features] = leaked[features].fillna(leaked[features].mean())
    ltrain, ltest = train_test_split(leaked, test_size=test_size, random_state=seed, shuffle=True)
    probs = _fit_predict(ltrain.reset_index(drop=True), ltest, features, seed)
    rows.append({"protocol": "random_split_global_impute", **evaluate(probs, ltest["result"].to_numpy())})

    # 3. The honest protocol: a single chronological cut at the same ratio.
    split_at = int(len(frame) * (1 - test_size))
    ctrain, ctest = frame.iloc[:split_at], frame.iloc[split_at:]
    probs = _fit_predict(ctrain, ctest, features, seed)
    rows.append({"protocol": "chronological_split", **evaluate(probs, ctest["result"].to_numpy())})

    return pd.DataFrame(rows)
