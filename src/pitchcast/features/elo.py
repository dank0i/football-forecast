"""Margin-aware Elo ratings.

Elo is the workhorse team-strength estimate in football forecasting and it is
the single feature the original analysis most needed: one-hot team identifiers
give every team one fixed strength for eight seasons, which is wrong for any
club that got promoted, relegated, bought, or rebuilt.

Three details separate a useful football Elo from the chess original:

* **Margin of victory.** A 4-0 win is stronger evidence than a 1-0 win, so the
  update is scaled by goal difference. Without it, ratings converge slowly and
  underrate dominant sides.
* **Home advantage.** Applied as a rating bonus to the home side when forming
  the expectation, so the update is not systematically biased against away teams.
* **Season regression.** Squads turn over each summer, and promoted teams
  inherit the rating of whoever went down. Pulling every rating part-way back to
  the league mean between seasons stops a dynasty's rating from ossifying.

Ratings are produced by one forward pass in kickoff order. Each match records
the ratings *as they stood before it was played*, so no future information can
reach a feature by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 65.0
    mov_exponent: float = 0.8
    season_regression: float = 0.25
    initial: float = 1500.0


def expected_home_score(rating_home: float, rating_away: float, home_advantage: float) -> float:
    """Probability-like expectation in [0, 1] for the home side."""
    return 1.0 / (1.0 + 10.0 ** ((rating_away - rating_home - home_advantage) / 400.0))


def _mov_multiplier(goal_diff: int, exponent: float) -> float:
    """Weight the update by margin, with diminishing returns past a rout."""
    return float((abs(goal_diff) + 1.0) ** exponent)


def compute_elo(matches: pd.DataFrame, params: EloParams | None = None) -> pd.DataFrame:
    """Return pre-match Elo ratings and derived features, one row per match.

    Ratings are kept in separate pools per league. The dataset contains only
    domestic fixtures, so no match ever links two leagues and a shared pool
    would imply comparisons the data cannot support.
    """
    params = params or EloParams()
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by kickoff before computing Elo")

    ratings: dict[tuple[int, int], float] = {}
    league_season: dict[int, str] = {}

    n = len(matches)
    home_pre = np.empty(n)
    away_pre = np.empty(n)
    home_played = np.zeros(n, dtype="int32")
    away_played = np.zeros(n, dtype="int32")
    played: dict[tuple[int, int], int] = {}

    cols = matches[["league_id", "season", "home_team_api_id", "away_team_api_id", "goal_diff"]].to_numpy()

    for i, (league_id, season, home_id, away_id, goal_diff) in enumerate(cols):
        league_id, home_id, away_id = int(league_id), int(home_id), int(away_id)

        # A new season in this league: pull every rating in the pool part-way
        # back toward the mean before the first fixture is processed.
        if league_season.get(league_id) != season:
            league_season[league_id] = season
            pool = [key for key in ratings if key[0] == league_id]
            if pool:
                mean = float(np.mean([ratings[key] for key in pool]))
                for key in pool:
                    ratings[key] += params.season_regression * (mean - ratings[key])

        hk, ak = (league_id, home_id), (league_id, away_id)
        r_home = ratings.setdefault(hk, params.initial)
        r_away = ratings.setdefault(ak, params.initial)
        home_pre[i], away_pre[i] = r_home, r_away
        home_played[i] = played.get(hk, 0)
        away_played[i] = played.get(ak, 0)

        expected = expected_home_score(r_home, r_away, params.home_advantage)
        actual = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
        delta = params.k * _mov_multiplier(int(goal_diff), params.mov_exponent) * (actual - expected)
        ratings[hk] = r_home + delta
        ratings[ak] = r_away - delta
        played[hk] = home_played[i] + 1
        played[ak] = away_played[i] + 1

    out = pd.DataFrame(
        {
            "match_index": matches["match_index"].to_numpy(),
            "elo_home": home_pre,
            "elo_away": away_pre,
            "elo_diff": home_pre - away_pre,
            "elo_matches_home": home_played,
            "elo_matches_away": away_played,
        }
    )
    out["elo_expected_home"] = 1.0 / (
        1.0 + 10.0 ** ((out["elo_away"] - out["elo_home"] - params.home_advantage) / 400.0)
    )
    return out


def tune_elo(
    matches: pd.DataFrame,
    k_grid: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0),
    hfa_grid: tuple[float, ...] = (40.0, 55.0, 65.0, 80.0, 95.0, 110.0, 125.0, 140.0),
    regression_grid: tuple[float, ...] = (0.0, 0.25, 0.5),
) -> tuple[EloParams, pd.DataFrame]:
    """Grid-search Elo hyperparameters by binary log-loss on decided matches.

    Scored on wins and losses only: plain Elo emits a single win expectation,
    not a three-way distribution, so it has nothing to say about a draw. The
    caller is responsible for restricting ``matches`` to the tuning window;
    passing the full dataset would tune on matches later used for evaluation.
    """
    rows = []
    best, best_loss = EloParams(), np.inf
    for k in k_grid:
        for hfa in hfa_grid:
            for reg in regression_grid:
                params = EloParams(k=k, home_advantage=hfa, season_regression=reg)
                elo = compute_elo(matches, params)
                decided = matches["result"] != 1
                warm = (elo["elo_matches_home"] >= 10) & (elo["elo_matches_away"] >= 10)
                mask = (decided & warm).to_numpy()
                if mask.sum() < 500:
                    continue
                p = np.clip(elo.loc[mask, "elo_expected_home"].to_numpy(), 1e-9, 1 - 1e-9)
                y = (matches.loc[mask, "result"] == 0).to_numpy(dtype=float)
                loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
                rows.append({"k": k, "home_advantage": hfa, "season_regression": reg, "logloss": loss})
                if loss < best_loss:
                    best, best_loss = params, loss
    return best, pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
