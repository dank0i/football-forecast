"""Contextual features: history between the two clubs, and squad churn.

These cover ground the other blocks miss. Elo and form describe each team
against the league as a whole; nothing so far describes *this fixture*, whether
these two clubs have a lopsided history, whether one of them was in a lower
division last year, or whether the eleven taking the field is the eleven that
has been playing.

A note on what "known before kickoff" means here. The squad block and
:func:`squad_continuity` both read the starting eleven, which is published
roughly an hour before the match. That is a real assumption and it is stated
rather than hidden: these forecasts are made *after* team news, not days ahead.
It keeps the market comparison fair, since closing odds also move on lineup
announcements, but a model intended to price a fixture midweek could not use
this block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import lineup_columns

H2H_WINDOW = 6


def head_to_head(matches: pd.DataFrame, window: int = H2H_WINDOW) -> pd.DataFrame:
    """Record between these two clubs over their previous meetings.

    Meetings are counted from the home team's perspective regardless of which
    ground they were played at, then a venue-specific count is kept alongside,
    because some fixtures are lopsided only at one of the two grounds.
    """
    pair_history: dict[frozenset, list[tuple[int, int]]] = {}
    venue_history: dict[tuple[int, int], list[int]] = {}

    n = len(matches)
    h2h_points = np.full(n, np.nan)
    h2h_goal_diff = np.full(n, np.nan)
    h2h_played = np.zeros(n)
    h2h_venue_points = np.full(n, np.nan)

    cols = matches[["home_team_api_id", "away_team_api_id", "goal_diff"]].to_numpy()
    for i, (home_id, away_id, goal_diff) in enumerate(cols):
        home_id, away_id, goal_diff = int(home_id), int(away_id), int(goal_diff)
        pair = frozenset((home_id, away_id))
        past = pair_history.get(pair, [])

        if past:
            recent = past[-window:]
            # Stored from the lower-id team's perspective, so flip when needed.
            points, diffs = [], []
            for stored_for, stored_diff in recent:
                sign = 1 if stored_for == home_id else -1
                diff = sign * stored_diff
                diffs.append(diff)
                points.append(3.0 if diff > 0 else (1.0 if diff == 0 else 0.0))
            h2h_points[i] = float(np.mean(points))
            h2h_goal_diff[i] = float(np.mean(diffs))
            h2h_played[i] = len(recent)

        venue_key = (home_id, away_id)
        venue_past = venue_history.get(venue_key, [])
        if venue_past:
            recent = venue_past[-window:]
            h2h_venue_points[i] = float(np.mean([3.0 if d > 0 else (1.0 if d == 0 else 0.0) for d in recent]))

        pair_history.setdefault(pair, []).append((home_id, goal_diff))
        venue_history.setdefault(venue_key, []).append(goal_diff)

    return pd.DataFrame(
        {
            "match_index": matches["match_index"].to_numpy(),
            "h2h_points": h2h_points,
            "h2h_goal_diff": h2h_goal_diff,
            "h2h_played": h2h_played,
            "h2h_venue_points": h2h_venue_points,
        }
    )


def promotion_flags(matches: pd.DataFrame) -> pd.DataFrame:
    """Whether each side is new to this league this season.

    Promoted clubs are systematically weaker than their Elo suggests, because
    the rating they carry was earned against different opposition, or is the
    default 1500 if they have never appeared. Seasons are ordered by their first
    kickoff so the check only ever looks backwards.
    """
    seasons = matches.groupby("season")["date"].min().sort_values().index.tolist()
    seen: dict[int, set[int]] = {}

    new_home = np.zeros(len(matches))
    new_away = np.zeros(len(matches))
    seasons_in_league_home = np.zeros(len(matches))
    seasons_in_league_away = np.zeros(len(matches))
    tenure: dict[tuple[int, int], int] = {}

    for season in seasons:
        mask = matches["season"] == season
        block = matches[mask]
        for side, new_col, tenure_col in (
            ("home", new_home, seasons_in_league_home),
            ("away", new_away, seasons_in_league_away),
        ):
            ids = block[f"{side}_team_api_id"].to_numpy()
            leagues = block["league_id"].to_numpy()
            positions = np.flatnonzero(mask.to_numpy())
            for pos, team_id, league_id in zip(positions, ids, leagues, strict=True):
                known = seen.get(int(league_id), set())
                new_col[pos] = 0.0 if int(team_id) in known else 1.0
                tenure_col[pos] = tenure.get((int(league_id), int(team_id)), 0)

        # Only after the whole season is scored is it folded into history.
        for league_id, group in block.groupby("league_id"):
            members = set(group["home_team_api_id"]) | set(group["away_team_api_id"])
            seen.setdefault(int(league_id), set()).update(int(t) for t in members)
            for team_id in members:
                key = (int(league_id), int(team_id))
                tenure[key] = tenure.get(key, 0) + 1

    return pd.DataFrame(
        {
            "match_index": matches["match_index"].to_numpy(),
            "home_newly_promoted": new_home,
            "away_newly_promoted": new_away,
            "home_seasons_in_league": seasons_in_league_home,
            "away_seasons_in_league": seasons_in_league_away,
            "diff_seasons_in_league": seasons_in_league_home - seasons_in_league_away,
        }
    )


def squad_continuity(matches: pd.DataFrame) -> pd.DataFrame:
    """How much of the starting eleven also started the team's previous match.

    A proxy for rotation, injury and suspension, none of which this database
    records directly. A heavily changed side is usually a weakened or rested
    one, and the squad-rating features cannot see that on their own.
    """
    last_eleven: dict[int, set[int]] = {}
    n = len(matches)
    home_overlap = np.full(n, np.nan)
    away_overlap = np.full(n, np.nan)

    home_cols = matches[lineup_columns("home")].to_numpy()
    away_cols = matches[lineup_columns("away")].to_numpy()
    home_ids = matches["home_team_api_id"].to_numpy()
    away_ids = matches["away_team_api_id"].to_numpy()

    for i in range(n):
        for players, team_id, out in (
            (home_cols[i], int(home_ids[i]), home_overlap),
            (away_cols[i], int(away_ids[i]), away_overlap),
        ):
            eleven = {int(p) for p in players if np.isfinite(p)}
            if not eleven:
                continue
            previous = last_eleven.get(team_id)
            if previous:
                out[i] = len(eleven & previous) / 11.0
            last_eleven[team_id] = eleven

    return pd.DataFrame(
        {
            "match_index": matches["match_index"].to_numpy(),
            "home_squad_continuity": home_overlap,
            "away_squad_continuity": away_overlap,
            "diff_squad_continuity": home_overlap - away_overlap,
        }
    )


def elo_momentum(matches: pd.DataFrame, elo: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Change in a team's Elo over its previous ``window`` matches.

    Level and trajectory are different things: a club rated 1600 and climbing is
    not the same bet as one rated 1600 and falling.
    """
    frame = matches[["match_index", "home_team_api_id", "away_team_api_id"]].merge(
        elo[["match_index", "elo_home", "elo_away"]], on="match_index"
    )
    records = []
    for side in ("home", "away"):
        records.append(
            pd.DataFrame(
                {
                    "match_index": frame["match_index"],
                    "team_api_id": frame[f"{side}_team_api_id"],
                    "rating": frame[f"elo_{side}"],
                    "side": side,
                }
            )
        )
    long = pd.concat(records, ignore_index=True).sort_values(["team_api_id", "match_index"])
    long["momentum"] = long.groupby("team_api_id", sort=False)["rating"].transform(
        lambda s: s - s.shift(window)
    )

    home = long[long["side"] == "home"].set_index("match_index")["momentum"]
    away = long[long["side"] == "away"].set_index("match_index")["momentum"]
    out = pd.DataFrame({"elo_momentum_home": home, "elo_momentum_away": away}).reset_index()
    out["elo_momentum_diff"] = out["elo_momentum_home"] - out["elo_momentum_away"]
    return out


def compute_context(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """All contextual features joined on ``match_index``."""
    out = head_to_head(matches)
    for block in (promotion_flags(matches), squad_continuity(matches), elo_momentum(matches, elo)):
        out = out.merge(block, on="match_index", how="left")
    return out
