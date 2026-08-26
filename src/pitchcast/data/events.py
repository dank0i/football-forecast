"""Parse the per-match event feeds stored as XML blobs.

``Match`` carries eight XML columns (goals, shots on and off target, fouls,
cards, crosses, corners, possession). They describe what happened *during* a
match, so they can never be features for that same match. Their value is as
history: a team's shot volume over previous fixtures says things about its
underlying strength that the scoreline hides, because goals are a low-count,
high-variance summary of a much larger number of chances.

Two audit findings govern how much weight these deserve, and both argue for
treating the feeds as a measured-not-assumed feature block:

* **Coverage is 32.6%, not the 54.7% the null-counts suggest.** The gap is the
  empty ``<shoton />`` stub documented in :func:`_team_counts`.
* **Signal is weak even where present.** Against a goal-feed control that
  reconciles with the scoreline at r=0.96, shot difference explains goal
  difference at only r=0.13 and corner difference at r=0.07. Possession is the
  one genuinely informative feed at r=0.24.

So these are wired in as an ablatable block rather than assumed useful, and
:mod:`pitchcast.backtest` reports what they are worth.

Everything here is aggregated to per-match, per-side counts. The rolling logic
that turns them into lagged features lives in ``features.form``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

COUNT_COLUMNS = ("shoton", "shotoff", "foulcommit", "corner", "cross")


def _has_blob(blob: object) -> bool:
    """True only for a genuinely present XML string.

    Missing blobs arrive from pandas as NaN, and ``bool(float("nan"))`` is True,
    so a plain truthiness check silently marks every match as having a feed.
    """
    return isinstance(blob, str) and blob.strip() != ""


def _team_counts(blob: str | None, home_id: int, away_id: int) -> tuple[float, float]:
    """Count <value> elements attributed to each side.

    Returns NaN when the feed carries no entries at all. That case is not a
    match in which nothing happened: 40.5% of the non-null ``shoton`` blobs in
    this database are the empty stub ``<shoton />``, which parses without error
    and yields a count of zero. Those ~5,800 fixtures are ordinary matches
    averaging 1.54 home goals, so scoring them as "no shots taken" both biases
    every rolling average toward zero and attenuates the shot-to-goal
    relationship (difference-vs-difference correlation 0.10 counting stubs as
    zero, 0.13 excluding them). Entries with no <team> are dropped rather than
    split between sides.
    """
    if not _has_blob(blob):
        return np.nan, np.nan
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return np.nan, np.nan
    values = root.findall("value")
    if not values:
        return np.nan, np.nan
    home = away = 0
    for value in values:
        team = value.findtext("team")
        if team is None:
            continue
        try:
            team_id = int(team)
        except ValueError:
            continue
        if team_id == home_id:
            home += 1
        elif team_id == away_id:
            away += 1
    return float(home), float(away)


def _final_possession(blob: str | None) -> tuple[float, float]:
    """Take the last recorded possession split in the feed.

    Possession is logged as running snapshots at intervals; the final entry is
    the full-match figure. Entries are occasionally logged with an elapsed time
    beyond 90, so the last element in document order is used rather than a
    time-based filter.
    """
    if not _has_blob(blob):
        return np.nan, np.nan
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return np.nan, np.nan
    home = away = np.nan
    for value in root.findall("value"):
        hp, ap = value.findtext("homepos"), value.findtext("awaypos")
        if hp is None or ap is None:
            continue
        try:
            h, a = float(hp), float(ap)
        except ValueError:
            continue
        if h + a == 0:
            continue
        home, away = h, a
    return home, away


def _card_counts(blob: str | None, home_id: int, away_id: int) -> tuple[float, float, float, float]:
    """Split cards into yellows and reds per side.

    The feed marks colour in <comment>, but that field is missing on a minority
    of entries; those are counted as yellows, which is the overwhelming base
    rate. A card feed with no entries is genuinely ambiguous (a clean match and
    an unrecorded one look identical), so it is treated as missing.
    """
    if not _has_blob(blob):
        return np.nan, np.nan, np.nan, np.nan
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return np.nan, np.nan, np.nan, np.nan
    values = root.findall("value")
    if not values:
        return np.nan, np.nan, np.nan, np.nan
    hy = ay = hr = ar = 0
    for value in values:
        team = value.findtext("team")
        if team is None:
            continue
        try:
            team_id = int(team)
        except ValueError:
            continue
        comment = (value.findtext("comment") or "y").strip().lower()
        is_red = comment.startswith("r")
        if team_id == home_id:
            hr, hy = (hr + 1, hy) if is_red else (hr, hy + 1)
        elif team_id == away_id:
            ar, ay = (ar + 1, ay) if is_red else (ar, ay + 1)
    return float(hy), float(ay), float(hr), float(ar)


def parse_match_events(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the XML feeds into one row of per-side counts per match."""
    records = []
    frame = matches[
        ["match_index", "home_team_api_id", "away_team_api_id", *COUNT_COLUMNS, "card", "possession"]
    ]
    for row in frame.itertuples(index=False):
        home_id, away_id = int(row.home_team_api_id), int(row.away_team_api_id)
        record: dict[str, float] = {"match_index": row.match_index}
        any_feed = False
        for col in COUNT_COLUMNS:
            home, away = _team_counts(getattr(row, col), home_id, away_id)
            record[f"home_{col}"] = home
            record[f"away_{col}"] = away
            any_feed |= bool(np.isfinite(home))
        hy, ay, hr, ar = _card_counts(row.card, home_id, away_id)
        record |= {"home_yellow": hy, "away_yellow": ay, "home_red": hr, "away_red": ar}
        hp, ap = _final_possession(row.possession)
        record["home_possession"] = hp
        record["away_possession"] = ap
        record["has_events"] = any_feed
        records.append(record)

    return pd.DataFrame.from_records(records)
