# football-forecast

Probabilistic football match forecasting, benchmarked against the betting market.

**[Read the writeup](https://dank0i.github.io/football-forecast/)**

Predicts home/draw/away probabilities for 25,979 matches across 11 European
leagues (2008-2016), evaluated with strictly chronological walk-forward
validation and scored with proper scoring rules rather than accuracy.

**Using no betting data at all, it recovers 92% of the bookmaker's edge over base
rates. It does not beat the market. The analysis below shows why: the gap is an
information limit, not a modelling one.**

| Task | Baseline | This model | Bookmaker |
|---|---|---|---|
| Three-way (H/D/A) | 45.5% | **52.3%** | 53.0% |
| Home win vs not | 54.5% | **64.2%** | 65.3% |
| Decided matches only | 61.0% | **70.0%** | 70.9% |
| When >80% confident (4% of matches) | - | **85.6%** | - |

Every accuracy figure is quoted against the baseline it has to clear, because a
football accuracy number without one is uninterpretable. The three-way task is
the honest headline; the other rows are easier problems and are labelled as such.

---

## Results

All figures are out-of-sample, averaged over six held-out seasons (16,888
matches), each predicted by a model trained only on the seasons that preceded it.

| Forecast | RPS ↓ | Log loss ↓ | Accuracy ↑ | Calibration error ↓ |
|---|---|---|---|---|
| Always predict a home win | 0.4144 | 18.68 | 45.9% | 0.361 |
| Base rates (46/25/29) | 0.2274 | 1.0655 | 45.6% | 0.008 |
| Dixon-Coles alone | 0.2032 | 0.9960 | 51.5% | 0.008 |
| **This model (no betting data)** | **0.1991** | **0.9817** | **52.3%** | **0.008** |
| This model + market features | 0.1979 | 0.9791 | 52.7% | 0.010 |
| **Bookmaker (Bet365, devigged)** | **0.1967** | **0.9728** | **53.0%** | **0.004** |

Skill scores against the base-rate forecast: the market improves on it by 13.5%,
the model by 12.4%. The model therefore captures **92% of the market's edge
without observing a single price.**

![Calibration](reports/calibration.png)

The model is nearly as well calibrated as the market across the whole
probability range. That is what makes the probabilities usable, not just
rank-ordered.

### What each feature block is worth

![Ablation](reports/ablation.png)

| Feature set | RPS | Accuracy |
|---|---|---|
| Context only (head-to-head, promotion) | 0.2126 | 48.9% |
| Rolling form only | 0.2065 | 50.3% |
| Dixon-Coles only | 0.2032 | 51.5% |
| Elo only | 0.2021 | 51.7% |
| **FIFA squad ratings only** | **0.2011** | **52.3%** |
| Elo + form + squad | 0.1997 | 52.3% |
| + context features | 0.1998 | 52.3% |
| + match events + Dixon-Coles | 0.1996 | 52.4% |
| Team-embedding network (PyTorch) | 0.1994 | 52.2% |
| **Boosting and the network averaged** | **0.1991** | **52.3%** |
| Market probabilities only | 0.1981 | 52.7% |
| Everything including odds | 0.1979 | 52.7% |

Three findings worth stating plainly:

- **Squad ratings are the single strongest block**, beating both Elo and rolling
  form on their own. Who is actually on the pitch matters more than how the club
  has been doing.
- **The match-event features are worth nothing** (0.1999 vs 0.1997), which
  matches what the data audit predicted before they were tried. See below.
- **Feeding the market's own probabilities through a model makes them worse**
  (0.1981 vs the market's 0.1967). The closing line is already an efficient
  forecast; a gradient booster can only add noise to it. This is the clearest
  argument against the common approach of dumping bookmaker odds into a
  classifier and reporting the resulting accuracy as a modelling result.

### Where the model stops improving, and why

Four separate attempts to close the remaining gap to the market produced nothing:

| Change | RPS | Accuracy |
|---|---|---|
| Elo + form + squad + Dixon-Coles | 0.1997 | 52.2% |
| + context features (head-to-head, promotion, squad churn) | 0.1996 | 52.4% |
| + 40-config hyperparameter search | 0.1998 | 52.2% |
| + stacking (GBM ⊕ Dixon-Coles ⊕ Elo) | 0.2000 | 52.3% |
| + calibration (temperature, prior shrinkage) | 0.2013 | 51.7% |

That looks like saturation, but "I tried things and they failed" is not evidence.
So the residual gap was decomposed across every cut available:

| Cut | RPS gap to market |
|---|---|
| Early / mid / late season | +0.0036 / +0.0026 / +0.0044 |
| Full starting XI known vs incomplete | +0.0030 / +0.0031 |
| Promoted team involved vs not | +0.0031 / +0.0030 |
| Market unsure / moderate / confident | +0.0036 / +0.0027 / +0.0028 |
| Across nine leagues | +0.0005 to +0.0041 |

**The gap is flat everywhere**, and that is the informative part. A modelling
deficiency concentrates: a model blind to team news would fall apart late in the
season and on matches with incomplete lineups. This one does not. A uniform
~0.003 penalty across every partition is what a constant information
disadvantage looks like. The missing information is injuries, suspensions,
motivation and money flow. None of it is in this database. In the Eredivisie the gap is +0.0005, statistically
indistinguishable from the bookmaker.

The practical conclusion is that the ceiling here is set by the sport, not the
model. Bookmakers reach 53% on three-way football and 55.8% in the most
predictable league in this dataset; anyone reporting substantially more is
either dropping draws, leaking future information, or comparing against the
wrong baseline.

### Does it make money?

No. The simulation is here because the result is worth showing.

![Betting](reports/betting.png)

Staking fractional Kelly on the best price across six bookmakers, on
out-of-sample forecasts only:

| Minimum edge | Bets | ROI | 95% bootstrap CI |
|---|---|---|---|
| 0% | 21,871 | -1.47% | [-5.21%, +2.56%] |
| 5% | 15,833 | -1.42% | [-5.34%, +2.85%] |
| 10% | 11,326 | -1.59% | [-5.92%, +2.75%] |
| 20% | 5,818 | -3.85% | [-11.22%, +4.06%] |

The no-skill return is **-2.83%**. That is what a bettor loses to the margin
when shopping the best of six books, and a single book charges -5.80%. The model
loses 1.4%, so it converts about half the margin into edge but does not clear it.
The confidence intervals contain zero. The full threshold sweep is shown because
picking one favourable threshold would make almost any strategy look profitable.

---

## Method

### Chronological validation, not random splits

Football data is a time series. A random `train_test_split` across eight seasons
trains on 2016 matches and tests on 2010 ones. That leaks the future three ways.
The model knows how a season ended before predicting its opening fixtures. It has
already met the specific opponents in the test match. And any imputation computed
over the full frame carries test statistics into training.

Every result here uses expanding-window walk-forward validation: for each test
season, train on all prior seasons and predict that season cold. The first two
seasons are burn-in for Elo and rolling form and are never scored.

Every feature is lagged by construction, and tests enforce it rather than assume
it. `tests/test_leakage.py` rewrites the last twenty scorelines and asserts no
earlier Elo rating moves. It rewrites a match's own result and asserts that its
pre-match form does not change. And it checks against the real database that no
FIFA rating used by a match is dated on or after that match.

### Features

| Block | Description |
|---|---|
| `elo` | Margin-aware Elo with home advantage and between-season regression to the mean, one rating pool per league, hyperparameters tuned only on burn-in seasons |
| `form` | Rolling points, goals, clean sheets over 3/5/10 matches; season-to-date PPG; days of rest; fixture congestion; venue-specific form |
| `squad` | As-of FIFA ratings of the eleven who actually started: squad mean, top-4, per-line strength using formation coordinates, goalkeeper skill, squad age |
| `context` | Head-to-head record between the two clubs, promotion status and seasons in the league, starting-XI continuity with the previous match, Elo momentum |
| `events` | Lagged shot, corner and possession volume parsed from the XML match feeds |
| `market` | Devigged bookmaker probabilities, consensus across six books, and inter-book disagreement |

### Models

- **Dixon-Coles bivariate Poisson**, the standard academic football model:
  per-team attack and defence parameters, a shared home-advantage term, the
  low-score dependence correction, and exponential time decay. Refit every 30
  days per league during each test season. It models the *scoreline*, so draws
  fall out of the diagonal instead of having to be learned against a 25% base
  rate, and over/under, both-teams-to-score and clean-sheet probabilities are
  sums over the same matrix.
- **Team-embedding neural network** (PyTorch). Each club gets a learned vector,
  and the network sees both vectors, their difference and their elementwise
  product, which lets it form club-versus-club interactions that a single Elo
  rating cannot express. It ranks slightly worse than boosting (52.2% against
  52.3%) and calibrates twice as well (0.0047 against 0.0100). Averaging the two
  at a fixed equal weight gives the best model here, 0.1991. The weight is fixed
  rather than tuned, because sweeping it on the held-out seasons would be
  choosing a hyperparameter on the test set.
- **LightGBM** over the engineered features, with native NaN handling. Two
  thirds of matches have no event feed and 15% have an incomplete lineup.
  Mean-imputing those would invent a league-average team where the data says
  "unknown". Class priors are left alone, since the target is a proper scoring
  rule.

### Why not accuracy

Accuracy keeps only the argmax, so 34/33/33 and 90/5/5 score the same when the
favourite wins. Look at the "always predict home" row in the results table. It
matches the base-rate forecast on accuracy, 45.9% against 45.6%, but scores a log
loss of 18.7 against 1.07. RPS is the standard in football forecasting because it
also respects the ordering of outcomes. Predicting an away win when the home side
wins costs more than predicting a draw.

---

## Data audit

Three problems in the source data that materially change the results, each found
by checking rather than assuming.

**40.5% of the "present" event feeds are empty stubs.** The `shoton` column is
non-null for 54.7% of matches, but 40.5% of those are `<shoton />`, which parses
successfully and yields a count of zero. Counting them as genuine zeros files
~5,800 ordinary matches (averaging 1.54 home goals) under "no shots taken".
Real usable coverage is 32.6%.

**The event feeds are weak even where they are populated.** Against a control, the `goal` feed, which reconciles with the scoreline at r=0.96, shot difference
explains goal difference at only r=0.13 and corner difference at r=0.07. Only
possession (r=0.24) carries real signal. This is why the events block is wired
in as ablatable rather than assumed useful, and why it contributes nothing.

**Bookmaker coverage is uneven.** Six books cover ~87% of matches; Pinnacle
covers 43%. Using all ten as equal features, as is common, silently weights the
sparse ones through imputation. Only the six dense books form the consensus.

---

## Relationship to the original analysis

This rebuilds a course project (CMSC320, Spring 2025) that reported "~65%
accuracy". `football-forecast audit` reproduces that pipeline and re-scores it. Three
things surfaced:

**A silent bug in the odds normalisation.** The original converted odds to
probabilities by overwriting each column in place:

```python
match_db[home] = ... (1/row[home]) / (1/row[home] + 1/row[draw] + 1/row[away]) ...
match_db[draw] = ... # row[home] is now a probability, not odds
```

By the second line the home column holds ~0.45 rather than ~2.2, and its
reciprocal re-enters the denominator. The resulting triples sum to **0.573 on
average** instead of 1.0, with the away probability collapsed to 0.021 against a
correct 0.281, a 13× understatement across 20 of the 30 odds features feeding
every model in the notebook.

**The "70% accuracy" figure needs its baseline.** Reproduced at 68.97%. But it
was measured after dropping all draws, which removes 25% of matches and turns a
three-way problem into a two-way one where the majority class is **61.5%**, not
the 45.9% it was implicitly compared against. The real lift is 7.5 points.

**Random splitting cost less than expected.** Re-running the original design
chronologically moves accuracy by 0.67 points. Worth measuring rather than
asserting: the large leak was in the *evaluation framing*, not the split.

---

## Reproducing

```bash
uv sync
uv run football-forecast fetch          # 313 MB, checksum-verified
uv run football-forecast build          # parse feeds, tune Elo, build features
uv run football-forecast dixon-coles    # precompute and cache the DC forecasts
uv run football-forecast backtest       # walk-forward evaluation
uv run football-forecast ensemble        # boosting blended with the neural net
uv run football-forecast ablate         # feature-block ablation
uv run football-forecast bet            # staking simulation
uv run football-forecast audit          # reproduce and re-score the original
uv run football-forecast report         # regenerate figures
uv run pytest                   # 44 tests, including the leakage suite
```

## Layout

```
docs/         the published writeup (GitHub Pages serves this)
src/football_forecast/
  data/       loader (tidy frames, checksum), events (XML feeds)
  features/   elo, form, squad, context, market (devigging), build (blocks)
  models/     dixon_coles, gbm, baselines, stacking, tuning
  evaluation/ metrics (RPS, log loss, ECE), betting, leakage, replication
  backtest.py walk-forward protocol
  report.py   figures
tests/        38 tests; test_leakage.py is the important one
```

## Data

[European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)
by Hugo Mathien: 25,979 matches, 11 leagues, 2008-2016, with lineups, FIFA
player and team attributes, ten bookmakers' odds, and XML match-event feeds.
Fetched from a [HuggingFace mirror](https://huggingface.co/datasets/julien-c/kaggle-hugomathien-soccer)
so no Kaggle credentials are needed.
