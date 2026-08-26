"""Hyperparameter search under the walk-forward protocol.

The parameters used until now were hand-picked. Searching them has to respect
the same time ordering as evaluation, or the search itself becomes the leak:
picking parameters by their score on the test seasons is just a slower way of
fitting to the test set.

So the search runs on *inner* folds carved out of the training seasons only. The
held-out seasons are never touched, and the winning configuration is fixed
before they are scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BURN_IN_SEASONS, SEASONS
from ..evaluation.metrics import ranked_probability_score
from .gbm import GBMForecaster

SEARCH_SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [30, 60, 120, 250, 500],
    "feature_fraction": [0.3, 0.4, 0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 20.0, 50.0],
    "lambda_l1": [0.0, 0.5, 2.0],
}


# LightGBM rejects a float where it wants an int ("num_leaves should be of type
# int, got 7.0"), and numpy turns a mixed list into floats on selection, so the
# integer-valued parameters are cast back explicitly.
INTEGER_PARAMS = frozenset({"num_leaves", "min_child_samples"})


def _coerce(key: str, value) -> float | int:
    return int(value) if key in INTEGER_PARAMS else float(value)


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({key: _coerce(key, rng.choice(values)) for key, values in SEARCH_SPACE.items()})
    return configs


def coerce_params(params: dict) -> dict:
    """Cast a stored config back to the types LightGBM expects."""
    return {key: _coerce(key, value) for key, value in params.items()}


def inner_folds(train_seasons: tuple[str, ...], n_folds: int = 2):
    """Carve validation folds from the tail of the training seasons."""
    usable = [s for s in train_seasons if s not in SEASONS[:BURN_IN_SEASONS]]
    for season in usable[-n_folds:]:
        yield tuple(s for s in train_seasons if s < season), season


def search(
    frame: pd.DataFrame,
    features: list[str],
    n_configs: int = 40,
    seed: int = 0,
    n_folds: int = 2,
) -> tuple[dict, pd.DataFrame]:
    """Random search scored by RPS on inner folds of the training window.

    The training window here is every season before the first held-out one, so
    no configuration is ever chosen using a season it will later be scored on.
    """
    train_seasons = SEASONS[:BURN_IN_SEASONS] + SEASONS[BURN_IN_SEASONS : BURN_IN_SEASONS + 2]
    folds = list(inner_folds(train_seasons, n_folds=n_folds))
    if not folds:
        raise ValueError("no inner folds available for tuning")

    rows = []
    for i, overrides in enumerate(sample_configs(n_configs, seed=seed)):
        scores = []
        for fit_seasons, valid_season in folds:
            fit = frame[frame["season"].isin(fit_seasons)]
            valid = frame[frame["season"] == valid_season]
            if fit.empty or valid.empty:
                continue
            params = dict(GBMForecaster.__dataclass_fields__["params"].default_factory())
            params.update(overrides)
            params["seed"] = seed
            cutoff = int(len(fit) * 0.85)
            model = GBMForecaster(features=features, params=params).fit(fit.iloc[:cutoff], fit.iloc[cutoff:])
            scores.append(ranked_probability_score(model.predict(valid), valid["result"].to_numpy()))
        if scores:
            rows.append({"config_id": i, **overrides, "rps": float(np.mean(scores))})

    table = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    best = coerce_params({k: v for k, v in table.iloc[0].items() if k in SEARCH_SPACE})
    return best, table
