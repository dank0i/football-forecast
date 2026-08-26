"""Neural forecaster with learned team embeddings.

Every other model here compresses a club into a scalar. Elo gives it one
number, Dixon-Coles gives it two (attack and defence), and the boosted model
sees only those derived quantities. That representation cannot express a
matchup: the idea that a particular side is awkward for another in a way its
overall strength does not predict.

An embedding can. Each club gets a learned vector, and the network sees the
home vector, the away vector, their difference, and their elementwise product.
That last term is the point, since it is the cheapest way to let the model form
interactions between two specific clubs rather than between two ratings.

Three things make this awkward in a walk-forward setting, and each is handled
explicitly rather than papered over:

* **Unseen clubs.** A promoted side has no embedding when the season it is
  promoted into is scored. It is mapped to a reserved unknown-team slot that is
  trained alongside the rest by randomly masking teams during training, so the
  slot means "a club I know nothing about" rather than being dead weight.
* **Missing features.** Two thirds of matches have no event feed. A network
  cannot split on NaN the way a tree can, so missing values are median-imputed
  using training-set medians only, and a companion binary indicator is appended
  for every column that is ever missing.
* **Small data.** 20,000 training rows against a few hundred parameters per
  club overfits readily, so embeddings are small, weight-decayed, and trained
  with early stopping against a chronologically held-out slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

# scikit-learn and PyTorch each ship their own copy of libomp, and on macOS
# loading both into one process segfaults once they actually run parallel work.
# Importing is fine; training is not. Pinning PyTorch to a single thread avoids
# the collision, and costs nothing here because the network is small enough that
# the batches are not the bottleneck.
torch.set_num_threads(1)

UNKNOWN_TEAM = 0


class TeamEmbeddingNet(nn.Module):
    """Embeddings for both clubs, concatenated with match features."""

    def __init__(
        self,
        n_teams: int,
        n_features: int,
        embedding_dim: int = 8,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # +1 for the reserved unknown-team slot at index 0.
        self.embedding = nn.Embedding(n_teams + 1, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.05)

        # home, away, difference, product -> 4 blocks of embedding_dim.
        width = embedding_dim * 4 + n_features
        layers: list[nn.Module] = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, home: torch.Tensor, away: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h, a = self.embedding(home), self.embedding(away)
        x = torch.cat([h, a, h - a, h * a, features], dim=1)
        return self.mlp(x)


@dataclass
class EmbeddingForecaster:
    """Fit/predict wrapper that owns its own preprocessing.

    The imputer, scaler and team index are all fitted on the training frame and
    then frozen, so a test season cannot leak its statistics back through
    normalisation, which is the same mistake the original analysis made with
    ``fillna(X.mean())`` over the full dataset.
    """

    features: list[str]
    embedding_dim: int = 8
    hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.3
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 12
    batch_size: int = 512
    seed: int = 42
    unknown_rate: float = 0.05

    model: TeamEmbeddingNet | None = None
    team_index: dict[int, int] = field(default_factory=dict)
    medians: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    missing_cols: np.ndarray | None = None
    best_epoch: int = 0

    def _device(self) -> torch.device:
        return torch.device("cpu")

    def _encode_teams(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home = frame["home_team_api_id"].map(self.team_index).fillna(UNKNOWN_TEAM).to_numpy("int64")
        away = frame["away_team_api_id"].map(self.team_index).fillna(UNKNOWN_TEAM).to_numpy("int64")
        return home, away

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.features].to_numpy(dtype="float64")
        missing = ~np.isfinite(raw)
        filled = np.where(missing, self.medians, raw)
        scaled = (filled - self.mean) / self.scale
        # Missingness is informative here, so it is a feature rather than noise
        # to be hidden by imputation.
        return np.hstack([scaled, missing[:, self.missing_cols].astype("float64")])

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> EmbeddingForecaster:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        teams = sorted(set(train["home_team_api_id"]) | set(train["away_team_api_id"]))
        self.team_index = {int(t): i + 1 for i, t in enumerate(teams)}

        raw = train[self.features].to_numpy(dtype="float64")
        finite = np.isfinite(raw)
        self.medians = np.where(finite.any(axis=0), np.nanmedian(np.where(finite, raw, np.nan), axis=0), 0.0)
        self.missing_cols = ~finite.all(axis=0)
        filled = np.where(finite, raw, self.medians)
        self.mean = filled.mean(axis=0)
        self.scale = np.where(filled.std(axis=0) > 1e-9, filled.std(axis=0), 1.0)

        x_train = torch.tensor(self._design(train), dtype=torch.float32)
        x_valid = torch.tensor(self._design(valid), dtype=torch.float32)
        h_train, a_train = (torch.tensor(v) for v in self._encode_teams(train))
        h_valid, a_valid = (torch.tensor(v) for v in self._encode_teams(valid))
        y_train = torch.tensor(train["result"].to_numpy("int64"))
        y_valid = torch.tensor(valid["result"].to_numpy("int64"))

        self.model = TeamEmbeddingNet(
            n_teams=len(teams),
            n_features=x_train.shape[1],
            embedding_dim=self.embedding_dim,
            hidden=self.hidden,
            dropout=self.dropout,
        )
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        best_loss, best_state, stale = np.inf, None, 0
        n = len(x_train)
        for epoch in range(self.max_epochs):
            self.model.train()
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs more than one row.
                    continue
                hb, ab = h_train[idx].clone(), a_train[idx].clone()
                # Randomly blank teams so the unknown slot learns a real prior
                # instead of being seen for the first time at prediction.
                mask = torch.tensor(rng.random(len(idx)) < self.unknown_rate)
                hb[mask] = UNKNOWN_TEAM
                mask = torch.tensor(rng.random(len(idx)) < self.unknown_rate)
                ab[mask] = UNKNOWN_TEAM

                optimiser.zero_grad()
                loss = loss_fn(self.model(hb, ab, x_train[idx]), y_train[idx])
                loss.backward()
                optimiser.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(self.model(h_valid, a_valid, x_valid), y_valid))
            if valid_loss < best_loss - 1e-5:
                best_loss, best_state, stale = (
                    valid_loss,
                    {k: v.clone() for k, v in self.model.state_dict().items()},
                    0,
                )
                self.best_epoch = epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict()")
        self.model.eval()
        x = torch.tensor(self._design(frame), dtype=torch.float32)
        home, away = (torch.tensor(v) for v in self._encode_teams(frame))
        with torch.no_grad():
            return torch.softmax(self.model(home, away, x), dim=1).numpy().astype("float64")

    def team_vectors(self) -> pd.DataFrame:
        """Learned embedding per club, for inspecting what the model grouped."""
        if self.model is None:
            raise RuntimeError("fit() must be called before team_vectors()")
        weights = self.model.embedding.weight.detach().numpy()
        rows = [
            {"team_api_id": team, **{f"e{i}": weights[slot, i] for i in range(weights.shape[1])}}
            for team, slot in self.team_index.items()
        ]
        return pd.DataFrame(rows)
