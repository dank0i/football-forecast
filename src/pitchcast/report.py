"""Generate the figures that carry the argument.

Four charts, each answering one question the reader will actually ask, in a
consistent style that survives being viewed on a phone or printed in grey.
"""

from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, REPORTS, ROOT
from .evaluation.metrics import calibration_table

INK = "#1b1b1f"
MUTED = "#8a8a94"
ACCENT = "#2f6f9f"
WARN = "#c1553b"
GRID = "#e3e3e8"


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left", fontweight="600")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_calibration(predictions: pd.DataFrame, path) -> None:
    """Reliability diagram: does 60% mean 60%?"""
    actual = predictions["result"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=160)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")

    for source, colour, label in (("model", ACCENT, "pitchcast"), ("market", WARN, "bookmaker")):
        cols = [f"{source}_{o}" for o in ("home", "draw", "away")]
        if not all(c in predictions.columns for c in cols):
            continue
        probs = predictions[cols].to_numpy(dtype=float)
        valid = np.isfinite(probs).all(axis=1)
        table = calibration_table(probs[valid], actual[valid])
        ax.plot(
            table["predicted"],
            table["observed"],
            marker="o",
            markersize=4,
            color=colour,
            linewidth=1.6,
            label=label,
        )

    _style(ax, "Calibration: predicted vs observed frequency", "predicted probability", "observed frequency")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(table: pd.DataFrame, market_rps: float, path) -> None:
    """What each feature block is worth, against the market's line."""
    data = table.sort_values("rps", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    # Substring matching would paint "all non-market" as a market model, so the
    # test excludes that label explicitly.
    uses_market = [("market" in c) and ("non-market" not in c) for c in data["config"]]
    colours = [WARN if flag else ACCENT for flag in uses_market]
    ax.barh(data["config"], data["rps"], color=colours, height=0.65)
    ax.axvline(market_rps, color=INK, linestyle="--", linewidth=1.2)
    ax.text(market_rps, -0.8, f"  bookmaker {market_rps:.4f}", fontsize=8, color=INK, va="top")
    ax.set_xlim(min(data["rps"].min(), market_rps) - 0.004, data["rps"].max() + 0.002)
    _style(ax, "Ranked probability score by feature set (lower is better)", "RPS")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
        plt.Rectangle((0, 0), 1, 1, color=WARN),
    ]
    ax.legend(handles, ["no betting data", "uses betting odds"], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_season_stability(predictions: pd.DataFrame, path) -> None:
    """Does the result hold up season by season, or rest on one lucky fold?"""
    from .backtest import summarise_by_season

    model = summarise_by_season(predictions, "model")
    market = summarise_by_season(predictions, "market")
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=160)
    x = np.arange(len(model))
    ax.plot(x, model["rps"], marker="o", color=ACCENT, linewidth=1.8, label="pitchcast")
    ax.plot(x, market["rps"], marker="s", color=WARN, linewidth=1.8, label="bookmaker")
    ax.set_xticks(x)
    ax.set_xticklabels(model["season"], rotation=30, ha="right")
    _style(ax, "Out-of-sample RPS by held-out season", "", "RPS")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_betting(sweep: pd.DataFrame, no_skill_roi: float, path) -> None:
    """The economic test, with the interval that stops it being oversold."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=160)
    x = sweep["edge_threshold"]
    ax.plot(x, sweep["roi_pct"], marker="o", color=ACCENT, linewidth=1.8, label="model ROI")
    ax.fill_between(
        x,
        sweep["roi_ci_low_pct"],
        sweep["roi_ci_high_pct"],
        color=ACCENT,
        alpha=0.15,
        label="95% bootstrap CI",
    )
    ax.axhline(0, color=INK, linewidth=1)
    ax.axhline(
        no_skill_roi, color=WARN, linestyle="--", linewidth=1.2, label=f"no-skill ROI ({no_skill_roi:.2f}%)"
    )
    _style(ax, "Return on investment by minimum required edge", "minimum edge", "ROI (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


PUBLISHED_FIGURES = ROOT / "docs" / "figures"


def _publish(figure: str) -> None:
    """Copy a generated figure into the published page.

    The page under ``docs/`` is served by GitHub Pages and embeds these by
    relative path. Copying on generation means a refreshed backtest cannot leave
    the published charts showing older numbers than the tables beside them.
    """
    source = REPORTS / figure
    if not source.exists():
        return
    PUBLISHED_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, PUBLISHED_FIGURES / figure)


def generate_report() -> None:
    """Write every figure that has the data behind it available."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions_path = DATA_PROCESSED / "predictions.parquet"
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        plot_calibration(predictions, REPORTS / "calibration.png")
        plot_season_stability(predictions, REPORTS / "season_stability.png")

        ablation_path = REPORTS / "ablation.csv"
        if ablation_path.exists():
            from .backtest import summarise

            market_rps = float(summarise(predictions, sources=("market",)).iloc[0]["rps"])
            plot_ablation(pd.read_csv(ablation_path), market_rps, REPORTS / "ablation.png")

    betting_path = REPORTS / "betting.csv"
    if betting_path.exists():
        plot_betting(pd.read_csv(betting_path), -2.83, REPORTS / "betting.png")

    for figure in ("calibration.png", "season_stability.png", "ablation.png", "betting.png"):
        _publish(figure)
