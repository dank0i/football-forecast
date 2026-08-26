"""Staking simulation: the economic test of a forecast.

Proper scoring rules say whether a forecast is *good*. Betting returns say
whether it is good *enough to be worth money*, which is a much higher bar,
because the price already embeds a 6% margin. A model can be genuinely skilful,
beat every statistical baseline, and still lose money on every bet it places.

Simulated honestly means:

* Staking only on the model's own out-of-sample forecasts, never on a match
  used to fit the model that priced it.
* Paying the real price, including the margin. No devigged "fair" odds.
* Fractional Kelly rather than flat stakes, since a forecaster with an edge
  should size by the edge, and full Kelly is famously over-aggressive on
  estimated (rather than known) probabilities.
* Reporting a bootstrap interval. Betting returns are extremely heavy-tailed;
  a positive point estimate over a few thousand bets is routinely noise, and
  quoting it without an interval is how backtests lie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DENSE_BOOKMAKERS
from ..features.market import best_available_odds


@dataclass
class BettingResult:
    n_bets: int
    n_opportunities: int
    total_staked: float
    profit: float
    roi: float
    final_bankroll: float
    hit_rate: float
    roi_ci_low: float
    roi_ci_high: float
    bets: pd.DataFrame

    def summary(self) -> dict[str, float]:
        return {
            "n_bets": self.n_bets,
            "total_staked": round(self.total_staked, 2),
            "profit": round(self.profit, 2),
            "roi_pct": round(self.roi * 100, 3),
            "roi_ci_low_pct": round(self.roi_ci_low * 100, 3),
            "roi_ci_high_pct": round(self.roi_ci_high * 100, 3),
            "hit_rate": round(self.hit_rate, 4),
            "final_bankroll": round(self.final_bankroll, 2),
        }


def _bootstrap_roi(profits: np.ndarray, stakes: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile interval for ROI, resampling bets with replacement."""
    if len(profits) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profits), size=(n_boot, len(profits)))
    rois = profits[idx].sum(axis=1) / np.maximum(stakes[idx].sum(axis=1), 1e-9)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def simulate(
    predictions: pd.DataFrame,
    matches: pd.DataFrame,
    prob_prefix: str = "model",
    edge_threshold: float = 0.05,
    kelly_fraction: float = 0.25,
    bankroll: float = 1000.0,
    max_stake_fraction: float = 0.02,
    books: tuple[str, ...] = DENSE_BOOKMAKERS,
    n_boot: int = 2000,
    seed: int = 42,
) -> BettingResult:
    """Bet every outcome whose model edge clears ``edge_threshold``.

    Edge is ``p * odds - 1``: the expected profit per unit staked. A threshold
    above zero is deliberate. At exactly zero the bettor is staking on
    rounding error, and the estimated probability carries far more uncertainty
    than the price does.
    """
    frame = predictions.merge(
        matches[["match_index", *[b + s for b in books for s in "HDA"]]],
        on="match_index",
        how="left",
    )
    probs = frame[[f"{prob_prefix}_{o}" for o in ("home", "draw", "away")]].to_numpy(dtype=float)
    odds = best_available_odds(frame, books=books)
    actual = frame["result"].to_numpy()

    usable = np.isfinite(probs).all(axis=1) & np.isfinite(odds).all(axis=1)
    edges = probs * odds - 1.0

    records = []
    current = bankroll
    for i in np.flatnonzero(usable):
        for outcome in range(3):
            edge = edges[i, outcome]
            if not np.isfinite(edge) or edge <= edge_threshold:
                continue
            price, p = odds[i, outcome], probs[i, outcome]
            # Kelly on decimal odds: edge divided by net odds.
            kelly = edge / (price - 1.0)
            stake = min(max(kelly * kelly_fraction, 0.0) * current, max_stake_fraction * current)
            if stake <= 0:
                continue
            won = actual[i] == outcome
            profit = stake * (price - 1.0) if won else -stake
            current += profit
            records.append(
                {
                    "match_index": frame["match_index"].iloc[i],
                    "season": frame["season"].iloc[i],
                    "outcome": outcome,
                    "model_prob": p,
                    "odds": price,
                    "edge": edge,
                    "stake": stake,
                    "won": bool(won),
                    "profit": profit,
                    "bankroll": current,
                }
            )

    bets = pd.DataFrame(records)
    if bets.empty:
        return BettingResult(0, int(usable.sum()), 0.0, 0.0, np.nan, bankroll, np.nan, np.nan, np.nan, bets)

    staked = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    low, high = _bootstrap_roi(bets["profit"].to_numpy(), bets["stake"].to_numpy(), n_boot, seed)
    return BettingResult(
        n_bets=len(bets),
        n_opportunities=int(usable.sum()),
        total_staked=staked,
        profit=profit,
        roi=profit / staked if staked else np.nan,
        final_bankroll=current,
        hit_rate=float(bets["won"].mean()),
        roi_ci_low=low,
        roi_ci_high=high,
        bets=bets,
    )


def threshold_sweep(
    predictions: pd.DataFrame,
    matches: pd.DataFrame,
    prob_prefix: str = "model",
    thresholds: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    **kwargs,
) -> pd.DataFrame:
    """ROI across edge thresholds.

    Included because a single favourable threshold is the easiest way to
    manufacture a profitable backtest. If the strategy only works at one
    setting, that setting was fitted to the test set.
    """
    rows = []
    for threshold in thresholds:
        result = simulate(predictions, matches, prob_prefix=prob_prefix, edge_threshold=threshold, **kwargs)
        rows.append({"edge_threshold": threshold, **result.summary()})
    return pd.DataFrame(rows)
