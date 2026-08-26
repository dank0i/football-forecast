"""Reproduce the original notebook's pipeline and re-score it honestly.

The point is not to dunk on the earlier work but to put a number on each
methodological choice, so the rebuild can be justified by measurement rather
than by assertion. Three claims from the original are checked:

1. **~53% with a Random Forest on one-hot team IDs plus 30 odds columns.**
   Reproduced, then re-run with a chronological split. The one-hot team
   identifiers are what make the random split expensive here: with a dummy per
   team and a shuffled split, the model can memorise how a specific club fared
   across the very seasons it is scored on.

2. **"70% accuracy" after dropping draws.** Reproduced, and then measured
   against the baseline that number has to clear. Dropping draws removes 25% of
   matches and turns a three-way problem into a two-way one, so the majority
   class rises from 46% to ~62%. The comparison that makes 70% look strong is
   against the *three-way* 46%, which is not the same problem.

3. **The odds-normalisation step.** The original overwrote each home column
   before computing the draw and away columns from it, so the second and third
   probabilities were derived from an already-normalised home value rather than
   the original odds. Both versions are computed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ..config import BOOKMAKERS
from ..evaluation.metrics import evaluate

ODDS_COLUMNS = [b + s for b in BOOKMAKERS for s in "HDA"]


def _present_books(frame: pd.DataFrame) -> list[str]:
    """Bookmakers with a complete H/D/A column triple in ``frame``.

    Callers sometimes pass a narrowed frame (a single book, or a test fixture),
    and iterating the full roster would raise on the first absent column.
    """
    return [b for b in BOOKMAKERS if all(b + s in frame.columns for s in "HDA")]


def original_odds_normalisation(matches: pd.DataFrame) -> pd.DataFrame:
    """The original in-place normalisation, reproduced exactly.

    Each triple is written back column by column, so by the time the draw and
    away columns are computed the home column already holds a probability near
    0.45 rather than odds near 2.2. The reciprocal of that value re-enters the
    denominator, and the resulting three "probabilities" do not sum to 1.
    """
    frame = matches.copy()
    for book in _present_books(frame):
        home, draw, away = book + "H", book + "D", book + "A"
        frame[home] = frame.apply(
            lambda r, h=home, d=draw, a=away: (1 / r[h]) / (1 / r[h] + 1 / r[d] + 1 / r[a]), axis=1
        )
        frame[draw] = frame.apply(
            lambda r, h=home, d=draw, a=away: (1 / r[d]) / (1 / r[h] + 1 / r[d] + 1 / r[a]), axis=1
        )
        frame[away] = frame.apply(
            lambda r, h=home, d=draw, a=away: (1 / r[a]) / (1 / r[h] + 1 / r[d] + 1 / r[a]), axis=1
        )
    return frame


def correct_odds_normalisation(matches: pd.DataFrame) -> pd.DataFrame:
    """The same intent, computing all three from the untouched odds."""
    frame = matches.copy()
    for book in _present_books(frame):
        cols = [book + s for s in "HDA"]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 1.0 / frame[cols].to_numpy(dtype=float)
        frame[cols] = raw / raw.sum(axis=1, keepdims=True)
    return frame


def normalisation_audit(matches: pd.DataFrame, book: str = "B365") -> pd.DataFrame:
    """Show the two normalisations side by side on the same fixtures."""
    cols = [book + s for s in "HDA"]
    subset = matches.dropna(subset=cols).head(2000)
    wrong = original_odds_normalisation(subset)[cols]
    right = correct_odds_normalisation(subset)[cols]
    return pd.DataFrame(
        {
            "version": ["original (in-place)", "corrected"],
            "mean_sum": [wrong.sum(axis=1).mean(), right.sum(axis=1).mean()],
            "min_sum": [wrong.sum(axis=1).min(), right.sum(axis=1).min()],
            "max_sum": [wrong.sum(axis=1).max(), right.sum(axis=1).max()],
            f"mean_{book}H": [wrong[book + "H"].mean(), right[book + "H"].mean()],
            f"mean_{book}D": [wrong[book + "D"].mean(), right[book + "D"].mean()],
            f"mean_{book}A": [wrong[book + "A"].mean(), right[book + "A"].mean()],
        }
    )


def _original_design(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot team IDs plus all 30 odds columns, as the original built it."""
    frame = matches.copy()
    frame["home_advantage"] = 1
    features = ["home_advantage", "home_team_api_id", "away_team_api_id", *ODDS_COLUMNS]
    design = pd.get_dummies(
        frame[features], columns=["home_team_api_id", "away_team_api_id"], drop_first=True
    )
    # Global-mean imputation, computed across the whole frame before splitting.
    design = design.fillna(design.mean())
    return design, frame["result"]


def replicate_three_way(matches: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """The original three-class model, scored under both protocols."""
    design, target = _original_design(matches)
    rows = []

    xtr, xte, ytr, yte = train_test_split(design, target, test_size=0.2, random_state=seed)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    model.fit(xtr, ytr)
    rows.append({"protocol": "original (random split)", **evaluate(model.predict_proba(xte), yte.to_numpy())})

    split_at = int(len(design) * 0.8)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    model.fit(design.iloc[:split_at], target.iloc[:split_at])
    probs = model.predict_proba(design.iloc[split_at:])
    honest = evaluate(probs, target.iloc[split_at:].to_numpy())
    rows.append({"protocol": "same model (chronological)", **honest})

    return pd.DataFrame(rows)


def replicate_no_draws(matches: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """The "70% accuracy" result, with the baseline it needs to be read against."""
    decided = matches[matches["result"] != 1].reset_index(drop=True)
    design, _ = _original_design(decided)
    target = (decided["result"] == 0).astype(int)

    xtr, xte, ytr, yte = train_test_split(design, target, test_size=0.2, random_state=seed)
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(xtr, ytr)
    accuracy = float((model.predict(xte) == yte).mean())

    majority = float(max(target.mean(), 1 - target.mean()))
    return pd.DataFrame(
        [
            {
                "metric": "reported accuracy (draws dropped)",
                "value": accuracy,
                "note": "two-class problem on 74.6% of matches",
            },
            {
                "metric": "majority-class baseline on same subset",
                "value": majority,
                "note": "predict home win for every decided match",
            },
            {
                "metric": "lift over the right baseline",
                "value": accuracy - majority,
                "note": "what the model actually adds",
            },
            {
                "metric": "three-way majority baseline",
                "value": float((matches["result"] == 0).mean()),
                "note": "the 46% the 70% was implicitly compared against",
            },
        ]
    )
