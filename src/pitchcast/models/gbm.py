"""Gradient-boosted forecaster over the engineered features.

LightGBM is used rather than the notebook's untuned ``RandomForestClassifier``
for two reasons that matter here and not in general:

* It splits on NaN natively. Roughly two thirds of matches have no event feed
  and 15% have an incomplete lineup, and mean-imputing those would invent a
  league-average team where the data says "unknown". Letting the tree learn a
  direction for missing values keeps the distinction.
* It is calibrated by default. The notebook's ``class_weight='balanced'``
  deliberately distorts the class priors to flatter the draw recall, which
  wrecks the probabilities: a 25% base-rate event gets reweighted as if it were
  33%, and every downstream probability is then wrong by construction. Since
  the target here is a proper scoring rule, the priors are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 120,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "num_threads": 0,
}


@dataclass
class GBMForecaster:
    """Thin wrapper that keeps feature order and class mapping honest."""

    features: list[str]
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    booster: lgb.Booster | None = None
    best_iteration: int | None = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> GBMForecaster:
        dtrain = lgb.Dataset(train[self.features], label=train["result"], free_raw_data=False)
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = None
        if valid is not None and len(valid):
            valid_sets = [lgb.Dataset(valid[self.features], label=valid["result"], reference=dtrain)]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.best_iteration = self.booster.best_iteration or self.num_boost_round
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("fit() must be called before predict()")
        probs = self.booster.predict(frame[self.features], num_iteration=self.best_iteration)
        return np.asarray(probs, dtype=float)

    def importances(self) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("fit() must be called before importances()")
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "gain": self.booster.feature_importance("gain"),
                    "split": self.booster.feature_importance("split"),
                }
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
