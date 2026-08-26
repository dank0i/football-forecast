"""Dixon-Coles bivariate Poisson goal model.

The standard academic football forecaster (Dixon & Coles, 1997). It models the
two scorelines rather than the three-way result, which buys two things the
original classifier approach could not have:

1. **Draws come out for free, and correctly.** A classifier has to learn the
   draw class against a 25% base rate with no natural structure, which is why
   draw recall collapses in almost every naive attempt. Here a draw is just the
   diagonal of the score matrix, so its probability falls out of the same
   parameters that produce wins and losses.
2. **One model answers every derived question.** Over/under 2.5 goals, both
   teams to score, clean sheets, and the exact 0-0 probability are all sums over
   cells of the same matrix. The original fitted three separate weak classifiers
   for these, each fed only bookmaker 1X2 odds, which barely encode total goals.

Each team gets an attack and a defence parameter; a shared ``gamma`` carries
home advantage. Two refinements matter:

* **Low-score dependence (``rho``).** Independent Poissons underpredict 0-0 and
  1-1 and overpredict 1-0 and 0-1. Dixon-Coles applies a correction to those
  four cells only.
* **Exponential time decay (``xi``).** A match from four years ago says less
  about a team than last month's. Each observation is weighted by
  ``exp(-xi * days_ago)``, which lets the model track squads that improve or
  decline instead of averaging over an era.

Teams are only ever compared within their own league, because the dataset holds
no cross-league fixtures to link the two rating pools.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln

MAX_GOALS = 10


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles correction, applied to the four low-scoring cells only."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class DixonColesFit:
    teams: list[int]
    attack: np.ndarray
    defence: np.ndarray
    gamma: float
    rho: float
    xi: float
    converged: bool
    n_matches: int
    index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {team: i for i, team in enumerate(self.teams)}

    def rates(self, home_team: int, away_team: int) -> tuple[float, float]:
        """Expected goals for each side, or NaN for a team never seen in training."""
        h, a = self.index.get(home_team), self.index.get(away_team)
        if h is None or a is None:
            return np.nan, np.nan
        lam = float(np.exp(self.attack[h] + self.defence[a] + self.gamma))
        mu = float(np.exp(self.attack[a] + self.defence[h]))
        return lam, mu

    def score_matrix(self, home_team: int, away_team: int, max_goals: int = MAX_GOALS) -> np.ndarray:
        """Joint probability of every scoreline up to ``max_goals`` each."""
        lam, mu = self.rates(home_team, away_team)
        if not np.isfinite(lam):
            return np.full((max_goals + 1, max_goals + 1), np.nan)
        goals = np.arange(max_goals + 1)
        home_pmf = np.exp(goals * np.log(lam) - lam - gammaln(goals + 1))
        away_pmf = np.exp(goals * np.log(mu) - mu - gammaln(goals + 1))
        matrix = np.outer(home_pmf, away_pmf)
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        matrix = matrix * _tau(
            hg.ravel(), ag.ravel(), np.full(hg.size, lam), np.full(hg.size, mu), self.rho
        ).reshape(matrix.shape)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def outcome_probabilities(self, home_team: int, away_team: int) -> np.ndarray:
        """[P(home win), P(draw), P(away win)] from the score matrix."""
        matrix = self.score_matrix(home_team, away_team)
        if not np.isfinite(matrix).all():
            return np.full(3, np.nan)
        home = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away = float(np.triu(matrix, 1).sum())
        return np.array([home, draw, away])

    def derived_markets(self, home_team: int, away_team: int) -> dict[str, float]:
        """Goal-market probabilities that fall out of the same score matrix."""
        matrix = self.score_matrix(home_team, away_team)
        if not np.isfinite(matrix).all():
            return dict.fromkeys(("over_2_5", "btts", "home_clean_sheet", "nil_nil"), np.nan)
        goals = np.arange(matrix.shape[0])
        totals = goals[:, None] + goals[None, :]
        return {
            "over_2_5": float(matrix[totals > 2].sum()),
            "btts": float(matrix[1:, 1:].sum()),
            "home_clean_sheet": float(matrix[:, 0].sum()),
            "nil_nil": float(matrix[0, 0]),
        }


def fit_dixon_coles(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float = 0.0018,
    max_iter: int = 400,
    warm_start: np.ndarray | None = None,
) -> DixonColesFit:
    """Fit by weighted maximum likelihood on matches played before ``as_of``.

    ``xi`` is the daily decay rate; ``xi=0`` weights all history equally.
    Identifiability is fixed by constraining mean attack to zero.
    """
    train = matches[matches["date"] < as_of]
    if train.empty:
        raise ValueError(f"no matches before {as_of}")

    teams = sorted(set(train["home_team_api_id"]) | set(train["away_team_api_id"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    home_idx = train["home_team_api_id"].map(index).to_numpy()
    away_idx = train["away_team_api_id"].map(index).to_numpy()
    home_goals = train["home_team_goal"].to_numpy(dtype=float)
    away_goals = train["away_team_goal"].to_numpy(dtype=float)

    days_ago = (as_of - train["date"]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * days_ago)

    log_fact_home = gammaln(home_goals + 1)
    log_fact_away = gammaln(away_goals + 1)

    def negative_log_likelihood(params: np.ndarray) -> float:
        attack = np.empty(n)
        attack[: n - 1] = params[: n - 1]
        # The last attack parameter is pinned so the vector sums to zero;
        # without it attack and defence are only identified up to a constant.
        attack[n - 1] = -attack[: n - 1].sum()
        defence = params[n - 1 : 2 * n - 1]
        gamma, rho = params[-2], params[-1]

        log_lam = attack[home_idx] + defence[away_idx] + gamma
        log_mu = attack[away_idx] + defence[home_idx]
        lam, mu = np.exp(log_lam), np.exp(log_mu)

        tau = _tau(home_goals, away_goals, lam, mu, rho)
        if np.any(tau <= 0):
            return 1e10

        ll = (
            home_goals * log_lam
            - lam
            - log_fact_home
            + away_goals * log_mu
            - mu
            - log_fact_away
            + np.log(tau)
        )
        return float(-np.sum(weights * ll))

    if warm_start is not None and warm_start.size == 2 * n + 1:
        x0 = warm_start.copy()
    else:
        x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25, -0.05]])

    bounds = [(-3.0, 3.0)] * (n - 1) + [(-3.0, 3.0)] * n + [(-1.0, 1.0), (-0.3, 0.3)]
    result = optimize.minimize(
        negative_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "maxfun": max_iter * 40},
    )

    attack = np.empty(n)
    attack[: n - 1] = result.x[: n - 1]
    attack[n - 1] = -attack[: n - 1].sum()
    return DixonColesFit(
        teams=teams,
        attack=attack,
        defence=result.x[n - 1 : 2 * n - 1],
        gamma=float(result.x[-2]),
        rho=float(result.x[-1]),
        xi=xi,
        converged=bool(result.success),
        n_matches=len(train),
    )
