"""Rolling team form, computed from previous matches only.

Every column here answers "what did we know about this team the moment before
kickoff?". The implementation reshapes the fixture list into one row per team
per match, sorts by kickoff, and takes rolling windows that are **shifted by
one** so a team's own current result can never enter its own feature. That
shift is the whole ballgame: without it, "goals scored in the last 5 matches"
silently contains the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOWS = (3, 5, 10)
EVENT_ROLLING = ("shoton", "shotoff", "corner", "possession")


def to_team_rows(matches: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Explode one match into two team-perspective rows (home and away)."""
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        cols = {
            "match_index": matches["match_index"],
            "date": matches["date"],
            "season": matches["season"],
            "league_id": matches["league_id"],
            "team_api_id": matches[f"{side}_team_api_id"],
            "opponent_api_id": matches[f"{opp}_team_api_id"],
            "is_home": side == "home",
            "goals_for": matches[f"{side}_team_goal"],
            "goals_against": matches[f"{opp}_team_goal"],
        }
        frame = pd.DataFrame(cols)
        if events is not None:
            merged = matches[["match_index"]].merge(events, on="match_index", how="left")
            for col in EVENT_ROLLING:
                frame[f"ev_{col}_for"] = merged[f"{side}_{col}"].to_numpy()
                frame[f"ev_{col}_against"] = merged[f"{opp}_{col}"].to_numpy()
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    rows["goal_diff"] = rows["goals_for"] - rows["goals_against"]
    rows["points"] = np.select([rows["goal_diff"] > 0, rows["goal_diff"] == 0], [3.0, 1.0], default=0.0)
    rows["won"] = (rows["goal_diff"] > 0).astype(float)
    rows["drew"] = (rows["goal_diff"] == 0).astype(float)
    rows["clean_sheet"] = (rows["goals_against"] == 0).astype(float)
    rows["failed_to_score"] = (rows["goals_for"] == 0).astype(float)
    return rows.sort_values(["team_api_id", "date", "match_index"], kind="mergesort").reset_index(drop=True)


def _shifted_rolling(group: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Mean of the previous ``window`` matches, excluding the current one."""
    return group[column].shift(1).rolling(window, min_periods=1).mean()


def compute_form(matches: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build lagged form features and fold them back to one row per match."""
    rows = to_team_rows(matches, events)
    grouped = rows.groupby("team_api_id", sort=False, group_keys=False)

    base_stats = ["points", "goals_for", "goals_against", "goal_diff", "clean_sheet", "failed_to_score"]
    for window in FORM_WINDOWS:
        for stat in base_stats:
            rows[f"form{window}_{stat}"] = grouped.apply(
                lambda g, s=stat, w=window: _shifted_rolling(g, s, w)
            )

    if events is not None:
        for col in EVENT_ROLLING:
            for suffix in ("for", "against"):
                rows[f"form10_ev_{col}_{suffix}"] = grouped.apply(
                    lambda g, c=col, s=suffix: _shifted_rolling(g, f"ev_{c}_{s}", 10)
                )

    # Season-to-date points-per-game: a longer memory than the rolling windows
    # but reset each August, so a strong side that just lost its squad is not
    # carried by last season's league position.
    season_group = rows.groupby(["team_api_id", "season"], sort=False, group_keys=False)
    rows["season_ppg"] = season_group.apply(lambda g: g["points"].shift(1).expanding(min_periods=1).mean())
    rows["season_matches"] = season_group.cumcount()

    # Days since the team last played. Fixture congestion is a real effect and
    # the first match of a season is left as NaN rather than an invented number.
    rows["days_rest"] = grouped.apply(lambda g: (g["date"] - g["date"].shift(1)).dt.days)
    rows["matches_last_14d"] = grouped.apply(_congestion)

    # Venue-specific form: home advantage is not uniform across clubs.
    venue_group = rows.groupby(["team_api_id", "is_home"], sort=False, group_keys=False)
    rows["venue_form5_points"] = venue_group.apply(
        lambda g: g["points"].shift(1).rolling(5, min_periods=1).mean()
    )

    rows["career_matches"] = grouped.cumcount()

    feature_cols = [c for c in rows.columns if c.startswith(("form", "season_", "venue_"))]
    feature_cols += ["days_rest", "matches_last_14d", "career_matches"]

    home = rows[rows["is_home"]].set_index("match_index")[feature_cols].add_prefix("home_")
    away = rows[~rows["is_home"]].set_index("match_index")[feature_cols].add_prefix("away_")
    out = home.join(away, how="outer").reset_index()

    # Differences carry most of the signal: a model comparing two teams cares
    # about the gap, and giving it the gap directly saves it from learning
    # subtraction through axis-aligned splits.
    for col in feature_cols:
        out[f"diff_{col}"] = out[f"home_{col}"] - out[f"away_{col}"]
    return out


def _congestion(group: pd.DataFrame) -> pd.Series:
    """Matches played by this team in the 14 days before each kickoff."""
    dates = group["date"].to_numpy("datetime64[ns]")
    window = np.timedelta64(14, "D")
    counts = np.empty(len(dates), dtype="float64")
    for i, day in enumerate(dates):
        counts[i] = np.count_nonzero((dates[:i] > day - window) & (dates[:i] < day))
    return pd.Series(counts, index=group.index)
