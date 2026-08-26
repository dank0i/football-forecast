"""Load the European Soccer Database into tidy frames.

The source is a 313 MB SQLite dump whose ``Match`` table is 115 columns wide and
mixes four different kinds of information: the fixture itself, the starting
lineups as player IDs, ten bookmakers' closing odds, and post-match event feeds
stored as XML blobs. This module splits that into frames that each mean one
thing, and normalises the pieces that are inconsistent at source.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BOOKMAKERS, SQLITE_PATH, SQLITE_SHA256

LINEUP_SLOTS = range(1, 12)
EVENT_COLUMNS = ("goal", "shoton", "shotoff", "foulcommit", "card", "cross", "corner", "possession")


def verify_database(path: Path = SQLITE_PATH) -> str:
    """Hash the SQLite file so a silently truncated download fails loudly."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `pitchcast fetch` to download the database.")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != SQLITE_SHA256:
        raise ValueError(f"database.sqlite checksum mismatch: expected {SQLITE_SHA256}, got {got}")
    return got


def _connect(path: Path = SQLITE_PATH) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `pitchcast fetch` to download the database.")
    return sqlite3.connect(path)


def load_matches(path: Path = SQLITE_PATH) -> pd.DataFrame:
    """Return one row per match: fixture, result, lineups, and odds.

    Rows are sorted by kickoff and given a monotonic ``match_index``. Everything
    downstream relies on that ordering to guarantee a feature can only ever see
    matches that had already been played.
    """
    with _connect(path) as conn:
        matches = pd.read_sql("SELECT * FROM Match", conn)
        leagues = pd.read_sql("SELECT id AS league_id, name AS league FROM League", conn)
        teams = pd.read_sql(
            "SELECT team_api_id, team_long_name AS team_name, team_short_name FROM Team", conn
        )

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.merge(leagues, on="league_id", how="left")

    for side in ("home", "away"):
        names = teams.rename(
            columns={
                "team_api_id": f"{side}_team_api_id",
                "team_name": f"{side}_team",
                "team_short_name": f"{side}_team_short",
            }
        )
        matches = matches.merge(names, on=f"{side}_team_api_id", how="left")

    matches["result"] = np.select(
        [
            matches["home_team_goal"] > matches["away_team_goal"],
            matches["home_team_goal"] == matches["away_team_goal"],
        ],
        [0, 1],
        default=2,
    ).astype("int8")
    matches["total_goals"] = matches["home_team_goal"] + matches["away_team_goal"]
    matches["goal_diff"] = matches["home_team_goal"] - matches["away_team_goal"]

    # Ties on date are broken by match_api_id so the ordering is deterministic
    # across runs; without it the Elo walk would depend on SQLite's row order.
    matches = matches.sort_values(["date", "match_api_id"], kind="mergesort").reset_index(drop=True)
    matches["match_index"] = np.arange(len(matches), dtype="int32")

    keep = [
        "match_index",
        "match_api_id",
        "date",
        "season",
        "stage",
        "league_id",
        "league",
        "country_id",
        "home_team_api_id",
        "away_team_api_id",
        "home_team",
        "away_team",
        "home_team_short",
        "away_team_short",
        "home_team_goal",
        "away_team_goal",
        "result",
        "total_goals",
        "goal_diff",
    ]
    keep += [f"{side}_player_{i}" for side in ("home", "away") for i in LINEUP_SLOTS]
    keep += [f"{side}_player_Y{i}" for side in ("home", "away") for i in LINEUP_SLOTS]
    keep += [b + s for b in BOOKMAKERS for s in "HDA"]
    keep += list(EVENT_COLUMNS)
    return matches[keep]


def load_player_attributes(path: Path = SQLITE_PATH) -> pd.DataFrame:
    """FIFA player ratings, one row per player per rating refresh.

    Sorted by (player, date) so an as-of join can take the last rating strictly
    before a given kickoff.
    """
    cols = [
        "player_api_id",
        "date",
        "overall_rating",
        "potential",
        "reactions",
        "finishing",
        "short_passing",
        "ball_control",
        "sprint_speed",
        "stamina",
        "strength",
        "interceptions",
        "positioning",
        "vision",
        "marking",
        "standing_tackle",
        "sliding_tackle",
        "gk_diving",
        "gk_handling",
        "gk_positioning",
        "gk_reflexes",
    ]
    with _connect(path) as conn:
        attrs = pd.read_sql(f"SELECT {', '.join(cols)} FROM Player_Attributes", conn)
    attrs["date"] = pd.to_datetime(attrs["date"])
    attrs = attrs.dropna(subset=["overall_rating"])
    return attrs.sort_values(["player_api_id", "date"], kind="mergesort").reset_index(drop=True)


def load_players(path: Path = SQLITE_PATH) -> pd.DataFrame:
    with _connect(path) as conn:
        players = pd.read_sql("SELECT player_api_id, player_name, birthday, height, weight FROM Player", conn)
    players["birthday"] = pd.to_datetime(players["birthday"])
    return players


def load_team_attributes(path: Path = SQLITE_PATH) -> pd.DataFrame:
    """FIFA team tactical sliders, one row per team per refresh (1,458 rows total).

    ``buildUpPlayDribbling`` is null for every row before 2014 in the source
    data, so it is dropped rather than imputed with a value no season could have.
    """
    with _connect(path) as conn:
        attrs = pd.read_sql("SELECT * FROM Team_Attributes", conn)
    attrs["date"] = pd.to_datetime(attrs["date"])
    numeric = [
        "buildUpPlaySpeed",
        "buildUpPlayPassing",
        "chanceCreationPassing",
        "chanceCreationCrossing",
        "chanceCreationShooting",
        "defencePressure",
        "defenceAggression",
        "defenceTeamWidth",
    ]
    attrs = attrs[["team_api_id", "date", *numeric]]
    return attrs.sort_values(["team_api_id", "date"], kind="mergesort").reset_index(drop=True)


def lineup_columns(side: str) -> list[str]:
    return [f"{side}_player_{i}" for i in LINEUP_SLOTS]


def position_columns(side: str) -> list[str]:
    """Formation Y-coordinates, which encode how far up the pitch a slot starts.

    Verified against the data: Y=1 is the goalkeeper, Y=3 the defensive line,
    Y in 4..8 the midfield band, and Y>=9 the forward line. Reading the line
    off the coordinate rather than the slot number matters because slot 9 is a
    midfielder in some formations and a striker in others.
    """
    return [f"{side}_player_Y{i}" for i in LINEUP_SLOTS]
