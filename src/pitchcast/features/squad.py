"""Squad strength from the FIFA player ratings, joined as-of kickoff.

The database ships 183,978 player-rating snapshots and the starting eleven for
82% of matches, and the original analysis used neither. Together they answer the
question team identity cannot: *this* Real Madrid, on *this* date, with these
eleven players actually on the pitch.

The join is the delicate part. ``Player_Attributes`` is a slowly-changing
dimension with a few refreshes per player per year, so a naive equality join
loses almost everything and a naive nearest join reaches forward in time and
grades a team by ratings published months after the match. Both are handled by
a backward as-of join with exact matches disallowed, so a match sees only the
last rating published strictly before kickoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import LINEUP_SLOTS, lineup_columns, position_columns

# Y-coordinate bands, verified against the formation data.
GK_Y = 1
DEF_Y_MAX = 3
MID_Y_MAX = 8

RATING = "overall_rating"
GK_SKILLS = ["gk_diving", "gk_handling", "gk_positioning", "gk_reflexes"]


def _lineup_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, side, slot) carrying the player and their position."""
    frames = []
    for side in ("home", "away"):
        ids = matches[lineup_columns(side)].to_numpy()
        ys = matches[position_columns(side)].to_numpy()
        n_matches, n_slots = ids.shape
        frames.append(
            pd.DataFrame(
                {
                    "match_index": np.repeat(matches["match_index"].to_numpy(), n_slots),
                    "date": np.repeat(matches["date"].to_numpy(), n_slots),
                    "side": side,
                    "slot": np.tile(np.fromiter(LINEUP_SLOTS, dtype="int8"), n_matches),
                    "player_api_id": ids.ravel(),
                    "pos_y": ys.ravel(),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    return long.dropna(subset=["player_api_id"])


def _assign_line(pos_y: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [pos_y <= GK_Y, pos_y <= DEF_Y_MAX, pos_y <= MID_Y_MAX],
            ["gk", "def", "mid"],
            default="att",
        ),
        index=pos_y.index,
    )


def compute_squad_features(
    matches: pd.DataFrame, player_attributes: pd.DataFrame, players: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate as-of player ratings into per-match squad strength features."""
    long = _lineup_long(matches)
    long["player_api_id"] = long["player_api_id"].astype("int64")
    long = long.sort_values("date", kind="mergesort")

    attrs = player_attributes.sort_values("date", kind="mergesort")
    cols = [RATING, "potential", "reactions", "finishing", "interceptions", "marking", *GK_SKILLS]
    attrs = attrs[["player_api_id", "date", *cols]]

    # allow_exact_matches=False enforces "strictly before kickoff". A rating
    # stamped the same day as the match is dropped rather than trusted, because
    # the database gives no time of day to order them by.
    joined = pd.merge_asof(
        long,
        attrs,
        on="date",
        by="player_api_id",
        direction="backward",
        allow_exact_matches=False,
    )

    if players is not None:
        joined = joined.merge(players[["player_api_id", "birthday"]], on="player_api_id", how="left")
        joined["age"] = (joined["date"] - joined["birthday"]).dt.days / 365.25

    joined["line"] = _assign_line(joined["pos_y"])
    joined["gk_rating"] = joined[GK_SKILLS].mean(axis=1)

    records: list[pd.DataFrame] = []
    grouped = joined.groupby(["match_index", "side"], sort=False)

    agg = grouped.agg(
        squad_rating=(RATING, "mean"),
        squad_rating_max=(RATING, "max"),
        squad_rating_std=(RATING, "std"),
        squad_potential=("potential", "mean"),
        squad_reactions=("reactions", "mean"),
        squad_known=(RATING, "count"),
    )
    if "age" in joined:
        agg["squad_age"] = grouped["age"].mean()
    records.append(agg)

    # Top-4 rated starters: squad quality is not the mean. A side with four
    # world-class players and seven journeymen wins more than its average says.
    top4 = grouped[RATING].apply(lambda s: s.nlargest(4).mean()).rename("squad_top4")
    records.append(top4.to_frame())

    for line in ("gk", "def", "mid", "att"):
        subset = joined[joined["line"] == line]
        by_line = subset.groupby(["match_index", "side"])[RATING].mean().rename(f"line_{line}")
        records.append(by_line.to_frame())

    keeper = joined[joined["line"] == "gk"]
    records.append(keeper.groupby(["match_index", "side"])["gk_rating"].mean().rename("gk_skill").to_frame())

    wide = pd.concat(records, axis=1).reset_index()
    home = wide[wide["side"] == "home"].drop(columns="side").set_index("match_index").add_prefix("home_")
    away = wide[wide["side"] == "away"].drop(columns="side").set_index("match_index").add_prefix("away_")
    out = home.join(away, how="outer").reset_index()

    for col in (c[len("home_") :] for c in home.columns):
        out[f"diff_{col}"] = out[f"home_{col}"] - out[f"away_{col}"]

    # A lineup with only a handful of rated players gives an unreliable mean, so
    # the count is kept as a feature and the caller can gate on it.
    out["squad_min_known"] = out[["home_squad_known", "away_squad_known"]].min(axis=1)
    return out
