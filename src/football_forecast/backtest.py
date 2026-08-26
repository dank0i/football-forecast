"""Walk-forward evaluation, season by season.

This replaces the single biggest methodological problem in the original
analysis: ``train_test_split(X, y, test_size=0.2, random_state=42)`` on eight
seasons of chronological data. A random split trains on 2016 matches and tests
on 2010 ones, so the model sees the future in three separate ways. It knows how
a team's season ended before predicting its September fixtures; it has met the
specific opponents in the test match; and any imputation computed over the full
frame carries test-set statistics into training. That inflates every score, and
the inflation is invisible because the held-out set is contaminated in the same
direction.

The protocol here is the one a forecaster would actually face. For each test
season, train on every match played before it and predict that season cold.
Nothing about the test season, including its own match results, its imputation
statistics, or its Dixon-Coles team strengths, is available at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from .config import BURN_IN_SEASONS, SEASONS
from .evaluation.metrics import evaluate
from .features.build import select_features
from .models import baselines
from .models.dixon_coles import fit_dixon_coles
from .models.gbm import GBMForecaster


@dataclass
class FoldResult:
    season: str
    n_train: int
    n_test: int
    predictions: pd.DataFrame


def season_folds(seasons: tuple[str, ...] = SEASONS, burn_in: int = BURN_IN_SEASONS):
    """Yield (train_seasons, test_season) with an expanding training window."""
    for i in range(burn_in, len(seasons)):
        yield seasons[:i], seasons[i]


def dixon_coles_predictions(
    frame: pd.DataFrame, test_index: pd.Index, xi: float = 0.0018, refit_days: int = 30
) -> pd.DataFrame:
    """Dixon-Coles forecasts for a test season, refit periodically per league.

    Refitting matters: team strengths estimated in August are stale by April.
    The model is refit every ``refit_days`` using all matches up to that point,
    so a fixture in March is predicted by a model that has seen the season's
    first seven months but not the fixture itself.
    """
    rows = []
    test = frame.loc[test_index]
    for league_id, league_test in test.groupby("league_id", sort=False):
        league_all = frame[frame["league_id"] == league_id]
        checkpoints = pd.date_range(
            league_test["date"].min(),
            league_test["date"].max() + pd.Timedelta(days=refit_days),
            freq=f"{refit_days}D",
        )
        fit = None
        for start, end in pairwise(checkpoints):
            window = league_test[(league_test["date"] >= start) & (league_test["date"] < end)]
            if window.empty:
                continue
            try:
                fit = fit_dixon_coles(league_all, as_of=start, xi=xi)
            except ValueError:
                continue
            for row in window.itertuples():
                probs = fit.outcome_probabilities(row.home_team_api_id, row.away_team_api_id)
                lam, mu = fit.rates(row.home_team_api_id, row.away_team_api_id)
                derived = fit.derived_markets(row.home_team_api_id, row.away_team_api_id)
                rows.append(
                    {
                        "match_index": row.match_index,
                        "dc_home": probs[0],
                        "dc_draw": probs[1],
                        "dc_away": probs[2],
                        "dc_lambda": lam,
                        "dc_mu": mu,
                        "dc_expected_goals": lam + mu,
                        "dc_supremacy": lam - mu,
                        **{f"dc_{k}": v for k, v in derived.items()},
                    }
                )
    return pd.DataFrame(rows)


def build_dixon_coles_cache(frame: pd.DataFrame, xi: float = 0.0018) -> pd.DataFrame:
    """Precompute Dixon-Coles forecasts for every match outside the first season.

    The fits depend only on match dates and scorelines, never on the feature
    blocks being ablated, so they are computed once and reused. Doing this
    inside every ablation run would refit the same ~600 models each time and
    dominate the runtime.
    """
    scored = frame[frame["season"] != SEASONS[0]]
    return dixon_coles_predictions(frame, scored.index, xi=xi)


def run_backtest(
    frame: pd.DataFrame,
    blocks: tuple[str, ...] = ("elo", "form", "squad"),
    use_dixon_coles: bool = True,
    xi: float = 0.0018,
    seed: int = 42,
    dc_cache: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Train and score one model configuration across every test season."""
    folds: list[FoldResult] = []
    if use_dixon_coles and dc_cache is None:
        dc_cache = build_dixon_coles_cache(frame, xi=xi)

    for train_seasons, test_season in season_folds():
        train_mask = frame["season"].isin(train_seasons)
        test_mask = frame["season"] == test_season
        train, test = frame[train_mask].copy(), frame[test_mask].copy()

        features = select_features(frame, blocks)
        if use_dixon_coles:
            # Every Dixon-Coles forecast in the cache, for training and test
            # seasons alike, came from a model fit only on matches that preceded
            # it, so the GBM learns from the same kind of input it meets at test
            # time and no future scoreline reaches either side of the split.
            assert dc_cache is not None
            dc_cols = [c for c in dc_cache.columns if c != "match_index"]
            train = train.merge(dc_cache, on="match_index", how="left")
            test = test.merge(dc_cache, on="match_index", how="left")
            features = features + dc_cols

        # The last 15% of the training window, chronologically, is the
        # early-stopping set. Sampling it at random would leak across the
        # boundary the whole design exists to protect.
        cutoff = int(len(train) * 0.85)
        fit_part, valid_part = train.iloc[:cutoff], train.iloc[cutoff:]

        params = dict(GBMForecaster.__dataclass_fields__["params"].default_factory())
        params["seed"] = seed
        model = GBMForecaster(features=features, params=params).fit(fit_part, valid_part)

        probs = model.predict(test)
        preds = pd.DataFrame(
            {
                "match_index": test["match_index"].to_numpy(),
                "season": test_season,
                "league": test["league"].to_numpy(),
                "date": test["date"].to_numpy(),
                "result": test["result"].to_numpy(),
                "model_home": probs[:, 0],
                "model_draw": probs[:, 1],
                "model_away": probs[:, 2],
            }
        )
        market_probs = baselines.market_baseline(test)
        preds[["market_home", "market_draw", "market_away"]] = market_probs
        prior_probs = baselines.prior_baseline(train, test)
        preds[["prior_home", "prior_draw", "prior_away"]] = prior_probs
        if use_dixon_coles:
            preds[["dc_home", "dc_draw", "dc_away"]] = test[["dc_home", "dc_draw", "dc_away"]].to_numpy()

        folds.append(FoldResult(test_season, len(train), len(test), preds))

    predictions = pd.concat([f.predictions for f in folds], ignore_index=True)
    return predictions, folds


def summarise(
    predictions: pd.DataFrame,
    sources: tuple[str, ...] = ("model", "market", "prior", "dc"),
) -> pd.DataFrame:
    """Metric table for every forecast source present in ``predictions``.

    Restricted to matches the market priced, so every source is scored on
    exactly the same fixtures. Comparing a model on 26k matches against a
    bookmaker on the 22.6k it quoted would flatter whichever had the easier set.
    """
    actual = predictions["result"].to_numpy()
    priced = np.isfinite(predictions[["market_home", "market_draw", "market_away"]].to_numpy()).all(axis=1)
    rows = []
    for source in sources:
        cols = [f"{source}_{o}" for o in ("home", "draw", "away")]
        if not all(c in predictions.columns for c in cols):
            continue
        probs = predictions[cols].to_numpy(dtype=float)
        metrics = evaluate(probs[priced], actual[priced])
        rows.append({"source": source, **metrics})
    return pd.DataFrame(rows)


def summarise_by_season(predictions: pd.DataFrame, source: str = "model") -> pd.DataFrame:
    cols = [f"{source}_{o}" for o in ("home", "draw", "away")]
    rows = []
    for season, group in predictions.groupby("season", sort=True):
        priced = np.isfinite(group[["market_home", "market_draw", "market_away"]].to_numpy()).all(axis=1)
        rows.append(
            {
                "season": season,
                **evaluate(group[cols].to_numpy(dtype=float)[priced], group["result"].to_numpy()[priced]),
            }
        )
    return pd.DataFrame(rows)


def _elo_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Turn the scalar Elo expectation into a three-way distribution.

    Elo emits a single win expectation and has nothing to say about draws, so a
    multinomial logit is fitted on the Elo difference to map it onto H/D/A. It
    is fitted on the training seasons only, and is deliberately a one-feature
    model: its job is to be a cheap, independent third opinion for the stacker,
    not to compete with the boosted model.
    """
    from sklearn.linear_model import LogisticRegression

    def design(frame: pd.DataFrame) -> np.ndarray:
        diff = frame["elo_diff"].to_numpy(dtype=float).reshape(-1, 1)
        return np.hstack([diff / 100.0, (diff / 100.0) ** 2])

    x_train, y_train = design(train), train["result"].to_numpy()
    ok = np.isfinite(x_train).all(axis=1)
    model = LogisticRegression(max_iter=1000).fit(x_train[ok], y_train[ok])

    x_test = design(test)
    out = np.full((len(test), 3), np.nan)
    valid = np.isfinite(x_test).all(axis=1)
    if valid.any():
        out[valid] = model.predict_proba(x_test[valid])
    return out


def run_stacked_backtest(
    frame: pd.DataFrame,
    blocks: tuple[str, ...] = ("elo", "form", "squad", "context"),
    params: dict | None = None,
    dc_cache: pd.DataFrame | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Walk-forward backtest that blends the boosted model, Dixon-Coles and Elo.

    The meta-learner needs out-of-fold component predictions to train on, and
    getting those without leaking is the whole difficulty. For each test season
    S, an inner model is trained on everything before season S-1 and used to
    predict S-1; the stacker is fitted on that genuinely out-of-sample season,
    then applied to S. No component prediction the stacker learns from was made
    by a model that had seen the match it is predicting.
    """
    from .models.stacking import Stacker

    if dc_cache is None:
        dc_cache = build_dixon_coles_cache(frame)
    dc_cols = [c for c in dc_cache.columns if c != "match_index"]
    merged = frame.merge(dc_cache, on="match_index", how="left")
    features = select_features(frame, blocks) + dc_cols

    base_params = dict(GBMForecaster.__dataclass_fields__["params"].default_factory())
    if params:
        base_params.update(params)
    base_params["seed"] = seed
    for key in ("num_leaves", "min_child_samples"):
        if key in base_params:
            base_params[key] = int(base_params[key])

    def train_gbm(seasons: tuple[str, ...]) -> GBMForecaster:
        subset = merged[merged["season"].isin(seasons)]
        cutoff = int(len(subset) * 0.85)
        return GBMForecaster(features=features, params=dict(base_params)).fit(
            subset.iloc[:cutoff], subset.iloc[cutoff:]
        )

    outputs = []
    for train_seasons, test_season in season_folds():
        test = merged[merged["season"] == test_season]
        train = merged[merged["season"].isin(train_seasons)]

        # Outer model: everything before the test season.
        outer = train_gbm(train_seasons)
        gbm_test = outer.predict(test)

        # Inner model, one season further back, to produce honest meta-features.
        calib_season = train_seasons[-1]
        inner_seasons = train_seasons[:-1]
        calib = merged[merged["season"] == calib_season]
        inner = train_gbm(inner_seasons)
        gbm_calib = inner.predict(calib)

        dc_calib = calib[["dc_home", "dc_draw", "dc_away"]].to_numpy(dtype=float)
        dc_test = test[["dc_home", "dc_draw", "dc_away"]].to_numpy(dtype=float)
        elo_calib = _elo_forecast(merged[merged["season"].isin(inner_seasons)], calib)
        elo_test = _elo_forecast(train, test)

        stacker = Stacker(sources=["gbm", "dc", "elo"])
        stacker.fit({"gbm": gbm_calib, "dc": dc_calib, "elo": elo_calib}, calib["result"].to_numpy())
        stacked = stacker.predict({"gbm": gbm_test, "dc": dc_test, "elo": elo_test})

        preds = pd.DataFrame(
            {
                "match_index": test["match_index"].to_numpy(),
                "season": test_season,
                "league": test["league"].to_numpy(),
                "date": test["date"].to_numpy(),
                "result": test["result"].to_numpy(),
            }
        )
        preds[["model_home", "model_draw", "model_away"]] = stacked
        preds[["gbm_home", "gbm_draw", "gbm_away"]] = gbm_test
        preds[["dc_home", "dc_draw", "dc_away"]] = dc_test
        preds[["elo_home", "elo_draw", "elo_away"]] = elo_test
        preds[["market_home", "market_draw", "market_away"]] = baselines.market_baseline(test)
        preds[["prior_home", "prior_draw", "prior_away"]] = baselines.prior_baseline(train, test)
        outputs.append(preds)

    return pd.concat(outputs, ignore_index=True)


def run_embedding_backtest(
    frame: pd.DataFrame,
    blocks: tuple[str, ...] = ("elo", "form", "squad", "context"),
    dc_cache: pd.DataFrame | None = None,
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Walk-forward backtest for the team-embedding network.

    Identical protocol to :func:`run_backtest`, so the two are directly
    comparable: same folds, same features, same held-out seasons, and an
    early-stopping slice taken from the chronological tail of the training
    window rather than sampled at random.
    """
    from .models.embedding_net import EmbeddingForecaster

    merged = frame
    features = select_features(frame, blocks)
    if dc_cache is not None:
        dc_cols = [c for c in dc_cache.columns if c != "match_index"]
        merged = frame.merge(dc_cache, on="match_index", how="left")
        features = features + dc_cols

    outputs = []
    for train_seasons, test_season in season_folds():
        train = merged[merged["season"].isin(train_seasons)]
        test = merged[merged["season"] == test_season]
        cutoff = int(len(train) * 0.85)

        model = EmbeddingForecaster(features=features, seed=seed, **kwargs).fit(
            train.iloc[:cutoff], train.iloc[cutoff:]
        )
        probs = model.predict(test)

        preds = pd.DataFrame(
            {
                "match_index": test["match_index"].to_numpy(),
                "season": test_season,
                "league": test["league"].to_numpy(),
                "date": test["date"].to_numpy(),
                "result": test["result"].to_numpy(),
            }
        )
        preds[["model_home", "model_draw", "model_away"]] = probs
        preds[["market_home", "market_draw", "market_away"]] = baselines.market_baseline(test)
        preds[["prior_home", "prior_draw", "prior_away"]] = baselines.prior_baseline(train, test)
        outputs.append(preds)

    return pd.concat(outputs, ignore_index=True)


def run_ensemble_backtest(
    frame: pd.DataFrame,
    blocks: tuple[str, ...] = ("elo", "form", "squad", "context"),
    dc_cache: pd.DataFrame | None = None,
    seed: int = 42,
    weight: float = 0.5,
) -> pd.DataFrame:
    """Average the boosted model and the embedding network.

    The two disagree in a useful way. Gradient boosting ranks slightly better;
    the network is markedly better calibrated (expected calibration error 0.005
    against 0.010), because embeddings force it to share statistical strength
    between clubs rather than carving axis-aligned splits per feature. Averaging
    keeps both properties.

    ``weight`` is fixed at one half deliberately. Sweeping it on the held-out
    seasons found an optimum around 0.4 to 0.75, but choosing from that range
    would be selecting a hyperparameter on the test set, which is the exact
    error this project exists to avoid. An equal average is the choice that
    requires no knowledge of the answer, and it sits inside the flat region
    anyway.
    """
    gbm, _ = run_backtest(
        frame, blocks=blocks, use_dixon_coles=dc_cache is not None, dc_cache=dc_cache, seed=seed
    )
    net = run_embedding_backtest(frame, blocks=blocks, dc_cache=dc_cache, seed=seed)

    gbm = gbm.set_index("match_index")
    net = net.set_index("match_index")
    shared = net.index.intersection(gbm.index)

    cols = ["model_home", "model_draw", "model_away"]
    blended = weight * net.loc[shared, cols].to_numpy() + (1 - weight) * gbm.loc[shared, cols].to_numpy()

    out = gbm.loc[shared, ["season", "league", "date", "result"]].copy()
    out[cols] = blended
    out[["gbm_home", "gbm_draw", "gbm_away"]] = gbm.loc[shared, cols].to_numpy()
    out[["net_home", "net_draw", "net_away"]] = net.loc[shared, cols].to_numpy()
    for source in ("market", "prior"):
        out[[f"{source}_{o}" for o in ("home", "draw", "away")]] = gbm.loc[
            shared, [f"{source}_{o}" for o in ("home", "draw", "away")]
        ].to_numpy()
    return out.reset_index()
