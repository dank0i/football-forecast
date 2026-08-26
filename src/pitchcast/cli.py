"""Command line entry point.

Every result in the README is reproducible from these commands, which is the
point: a notebook records what happened once, a CLI lets someone else check it.
"""

from __future__ import annotations

import urllib.request
import warnings

import pandas as pd
import typer

from .config import DATA_PROCESSED, REPORTS, SQLITE_PATH, SQLITE_URL
from .features.build import FEATURE_BLOCKS

app = typer.Typer(
    add_completion=False,
    help="Probabilistic football forecasting, benchmarked against the betting market.",
)

FEATURES_CACHE = DATA_PROCESSED / "features.parquet"
EVENTS_CACHE = DATA_PROCESSED / "events.parquet"
DC_CACHE = DATA_PROCESSED / "dixon_coles.parquet"


def _echo(message: str) -> None:
    typer.secho(message, fg=typer.colors.CYAN)


@app.command()
def fetch(force: bool = typer.Option(False, help="Re-download even if the file exists.")) -> None:
    """Download the European Soccer Database (313 MB) and verify its checksum."""
    from .data.loader import verify_database

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SQLITE_PATH.exists() and not force:
        _echo(f"{SQLITE_PATH} already present; verifying")
    else:
        _echo(f"downloading {SQLITE_URL}")
        urllib.request.urlretrieve(SQLITE_URL, SQLITE_PATH)
    digest = verify_database(SQLITE_PATH)
    _echo(f"checksum ok: {digest[:16]}...")


@app.command()
def build(events: bool = typer.Option(True, help="Parse the XML event feeds (slow).")) -> None:
    """Build and cache the feature matrix."""
    from .config import BURN_IN_SEASONS, SEASONS
    from .data.events import parse_match_events
    from .data.loader import load_matches
    from .features.build import build_features
    from .features.elo import tune_elo

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    _echo(f"loaded {len(matches):,} matches")

    event_frame = None
    if events:
        if EVENTS_CACHE.exists():
            event_frame = pd.read_parquet(EVENTS_CACHE)
            _echo("reusing cached event feeds")
        else:
            event_frame = parse_match_events(matches)
            event_frame.to_parquet(EVENTS_CACHE, index=False)
            usable = event_frame["home_shoton"].notna().mean()
            _echo(f"parsed event feeds ({usable:.1%} of matches usable)")

    # Elo is tuned only on the burn-in seasons, which are never scored.
    tuning_window = matches[matches["season"].isin(SEASONS[: BURN_IN_SEASONS + 1])]
    params, _ = tune_elo(tuning_window)
    _echo(
        f"tuned Elo on {tuning_window['season'].nunique()} burn-in seasons: "
        f"k={params.k}, hfa={params.home_advantage}"
    )

    frame = build_features(matches, event_frame, params)
    frame.to_parquet(FEATURES_CACHE, index=False)
    _echo(f"wrote {FEATURES_CACHE} ({frame.shape[0]:,} x {frame.shape[1]})")


@app.command("dixon-coles")
def dixon_coles(xi: float = typer.Option(0.0018, help="Daily time-decay rate.")) -> None:
    """Precompute and cache the Dixon-Coles forecasts."""
    from .backtest import build_dixon_coles_cache

    frame = _load_features()
    cache = build_dixon_coles_cache(frame, xi=xi)
    cache.to_parquet(DC_CACHE, index=False)
    _echo(f"wrote {DC_CACHE} ({len(cache):,} forecasts)")


@app.command()
def backtest(
    blocks: str = typer.Option("elo,form,squad", help=f"Comma-separated from {','.join(FEATURE_BLOCKS)}."),
    dixon_coles: bool = typer.Option(True, "--dixon-coles/--no-dixon-coles"),
) -> None:
    """Run the walk-forward backtest and print the metric table."""
    from .backtest import run_backtest, summarise, summarise_by_season

    frame = _load_features()
    cache = pd.read_parquet(DC_CACHE) if (dixon_coles and DC_CACHE.exists()) else None
    selected = tuple(b.strip() for b in blocks.split(",") if b.strip())

    predictions, _ = run_backtest(frame, blocks=selected, use_dixon_coles=dixon_coles, dc_cache=cache)
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(DATA_PROCESSED / "predictions.parquet", index=False)

    typer.echo("\n" + summarise(predictions).round(4).to_string(index=False))
    typer.echo("\nBy season:")
    typer.echo(summarise_by_season(predictions).round(4).to_string(index=False))


@app.command()
def ablate() -> None:
    """Score every feature-block combination and write reports/ablation.csv."""
    from .backtest import run_backtest, summarise
    from .evaluation.metrics import skill_score

    frame = _load_features()
    cache = pd.read_parquet(DC_CACHE) if DC_CACHE.exists() else None
    configs = [
        ("elo only", ("elo",), False),
        ("form only", ("form",), False),
        ("squad only", ("squad",), False),
        ("Dixon-Coles only", (), True),
        ("elo+form", ("elo", "form"), False),
        ("elo+form+squad", ("elo", "form", "squad"), False),
        ("elo+form+squad+events", ("elo", "form", "squad", "events"), False),
        ("elo+form+squad +DC", ("elo", "form", "squad"), True),
        ("all non-market +DC", ("elo", "form", "squad", "events"), True),
        ("market only", ("market",), False),
        ("all + market +DC", FEATURE_BLOCKS, True),
    ]
    rows = []
    for name, blocks, use_dc in configs:
        predictions, _ = run_backtest(frame, blocks=blocks, use_dixon_coles=use_dc, dc_cache=cache)
        summary = summarise(predictions, sources=("model",)).iloc[0]
        actual = predictions["result"].to_numpy()
        model = predictions[["model_home", "model_draw", "model_away"]].to_numpy()
        market = predictions[["market_home", "market_draw", "market_away"]].to_numpy()
        prior = predictions[["prior_home", "prior_draw", "prior_away"]].to_numpy()
        rows.append(
            {
                "config": name,
                "rps": summary.rps,
                "log_loss": summary.log_loss,
                "accuracy": summary.accuracy,
                "ece": summary.ece,
                "skill_vs_prior": skill_score(model, prior, actual),
                "skill_vs_market": skill_score(model, market, actual),
            }
        )
        _echo(f"  {name:24} rps={summary.rps:.4f}")

    table = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "ablation.csv", index=False)
    typer.echo("\n" + table.round(4).to_string(index=False))


@app.command()
def bet(
    edge: float = typer.Option(0.05, help="Minimum edge required to place a bet."),
    kelly: float = typer.Option(0.25, help="Kelly fraction."),
) -> None:
    """Simulate staking on the model's out-of-sample forecasts."""
    from .data.loader import load_matches
    from .evaluation.betting import threshold_sweep

    predictions = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    sweep = threshold_sweep(predictions, load_matches(), kelly_fraction=kelly)
    REPORTS.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(REPORTS / "betting.csv", index=False)
    typer.echo(sweep.to_string(index=False))


@app.command()
def audit() -> None:
    """Reproduce the original notebook's results and re-score them honestly."""
    from .data.loader import load_matches
    from .evaluation.replication import normalisation_audit, replicate_no_draws, replicate_three_way

    warnings.filterwarnings("ignore")
    matches = load_matches()
    REPORTS.mkdir(parents=True, exist_ok=True)

    typer.echo("\nOdds normalisation:")
    typer.echo(normalisation_audit(matches).round(4).to_string(index=False))
    typer.echo("\nThree-way model under both protocols:")
    typer.echo(replicate_three_way(matches).round(4).to_string(index=False))
    typer.echo("\nThe 'draws dropped' accuracy claim:")
    typer.echo(replicate_no_draws(matches).round(4).to_string(index=False))


@app.command()
def report() -> None:
    """Regenerate the figures in reports/."""
    from .report import generate_report

    generate_report()
    _echo(f"figures written to {REPORTS}")


def _load_features() -> pd.DataFrame:
    if not FEATURES_CACHE.exists():
        raise typer.BadParameter(f"{FEATURES_CACHE} not found. Run `pitchcast build` first.")
    return pd.read_parquet(FEATURES_CACHE)


if __name__ == "__main__":
    app()
