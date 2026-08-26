"""Assemble the match-level feature matrix.

Features are grouped into named blocks so the backtest can ablate them and
report what each is actually worth, rather than asserting that more features
help. The blocks are deliberately separable:

``elo``      team strength from results alone
``form``     recent results, scoring, rest and congestion
``squad``    FIFA ratings of the eleven who actually started
``context``  head-to-head history, promotion status, squad churn, Elo momentum
``events``   lagged shot/possession volume from the XML feeds
``market``   the bookmakers' own devigged probabilities

``market`` is kept apart from the rest because including it changes the question
being asked. With market features the model is decoding a forecast that already
exists; without them it is producing an independent one. Both are interesting,
but only the second can be said to *beat* the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DENSE_BOOKMAKERS
from ..data.loader import load_matches, load_player_attributes, load_players
from . import market as market_mod
from .context import compute_context
from .elo import EloParams, compute_elo
from .form import compute_form
from .squad import compute_squad_features

FEATURE_BLOCKS = ("elo", "form", "squad", "context", "events", "market")


def build_features(
    matches: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    elo_params: EloParams | None = None,
    devig_method: str = "shin",
) -> pd.DataFrame:
    """Return matches joined to every feature block, one row per fixture."""
    if matches is None:
        matches = load_matches()

    frame = matches.copy()
    elo = compute_elo(matches, elo_params)
    frame = frame.merge(elo, on="match_index", how="left")
    frame = frame.merge(compute_context(matches, elo), on="match_index", how="left")
    frame = frame.merge(compute_form(matches, events), on="match_index", how="left")
    frame = frame.merge(
        compute_squad_features(matches, load_player_attributes(), load_players()),
        on="match_index",
        how="left",
    )

    consensus = market_mod.consensus_probabilities(matches, books=DENSE_BOOKMAKERS, method=devig_method)
    # Disagreement between books is a genuine signal: a fixture every book
    # prices the same is one the market is confident about. Rows where no book
    # quoted a price are an empty slice, and nanmax warns rather than returning
    # NaN quietly, so they are masked out first.
    per_book = np.stack(
        [
            market_mod.bookmaker_probabilities(matches, book=b, method=devig_method)[:, 0]
            for b in DENSE_BOOKMAKERS
        ]
    )
    quoted = np.isfinite(per_book).any(axis=0)
    spread = np.full(len(frame), np.nan)
    if quoted.any():
        priced = per_book[:, quoted]
        spread[quoted] = np.nanmax(priced, axis=0) - np.nanmin(priced, axis=0)

    stage_max = frame.groupby(["league_id", "season"])["stage"].transform("max")
    # Assigned in one concat: adding these column by column repeatedly copies
    # a 250-column frame and pandas warns about the fragmentation.
    extra = pd.DataFrame(
        {
            "mkt_home": consensus[:, 0],
            "mkt_draw": consensus[:, 1],
            "mkt_away": consensus[:, 2],
            "mkt_overround": market_mod.overround(matches),
            "mkt_home_spread": spread,
            "is_early_season": (frame["stage"] <= 4).astype("int8").to_numpy(),
            "stage_norm": (frame["stage"] / stage_max).to_numpy(),
        },
        index=frame.index,
    )
    return pd.concat([frame, extra], axis=1)


def block_columns(frame: pd.DataFrame, block: str) -> list[str]:
    """Column names belonging to one feature block."""
    if block == "elo":
        # elo_momentum_* belongs to the context block, so it is excluded here to
        # keep the ablation honest: a block must not be credited for another's
        # columns just because they share a prefix.
        return [c for c in frame.columns if c.startswith("elo_") and "momentum" not in c]
    if block == "form":
        return [
            c
            for c in frame.columns
            if (c.startswith(("home_form", "away_form", "diff_form")) and "_ev_" not in c)
            or c.endswith(("season_ppg", "season_matches", "days_rest", "matches_last_14d", "career_matches"))
            or "venue_form" in c
        ]
    if block == "squad":
        keys = ("squad_", "line_", "gk_skill")
        return [c for c in frame.columns if any(k in c for k in keys) and "continuity" not in c]
    if block == "context":
        keys = ("h2h_", "newly_promoted", "seasons_in_league", "squad_continuity", "elo_momentum")
        return [c for c in frame.columns if any(k in c for k in keys)]
    if block == "events":
        return [c for c in frame.columns if "_ev_" in c]
    if block == "market":
        return [c for c in frame.columns if c.startswith("mkt_")]
    raise ValueError(f"unknown feature block {block!r}; choose from {FEATURE_BLOCKS}")


def select_features(frame: pd.DataFrame, blocks: tuple[str, ...]) -> list[str]:
    """Resolve a set of blocks to a deduplicated, ordered column list."""
    cols: list[str] = ["is_early_season", "stage_norm"]
    for block in blocks:
        for col in block_columns(frame, block):
            if col not in cols:
                cols.append(col)
    return [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
