"""
BPR matrix factorization.

Every user u and movie i get a vector of n_factors numbers (p_u, q_i); every movie gets a bias b_i.
  score(u, i) = p_u . q_i + b_i
Training takes triples (u, i, j): u rated movie i and did not rate the random movie j.
It changes the numbers so that score(u, i) gets higher than score(u, j).
"""

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from src.models.base import Recommender


@dataclass
class BPR(Recommender):
    n_factors: int = 64
    regularization: float = 0.001
    learning_rate: float = 0.05
    n_epochs: int = 20
    batch_size: int = 1000
    seed: int = 42
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"bpr_f{self.n_factors}_reg{self.regularization}"

    def fit(self, training_ratings: pd.DataFrame, n_users: int, n_movies: int) -> Self:
        rng = np.random.default_rng(self.seed)
        # small random start values (all zeros would give every movie the same update)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.movie_factors = rng.normal(0, 0.1, (n_movies, self.n_factors)).astype(np.float32)
        self.movie_bias = np.zeros(n_movies, dtype=np.float32)

        # every training rating is a pair (u, i); has_rated[u, j] = True if u rated j
        users = training_ratings["user_index"].to_numpy()
        movies = training_ratings["movie_index"].to_numpy()
        has_rated = np.zeros((n_users, n_movies), dtype=bool)
        has_rated[users, movies] = True

        for _ in range(self.n_epochs):
            order = rng.permutation(len(users))  # every pair once per epoch, in a new order
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                u, i = users[batch], movies[batch]
                # a random movie j for each user, drawn again while the user has rated it
                j = rng.integers(n_movies, size=len(u))
                while has_rated[u, j].any():
                    redraw = has_rated[u, j]
                    j[redraw] = rng.integers(n_movies, size=redraw.sum())
                self._gradient_step(u, i, j)
        return self

    def _gradient_step(self, u, i, j) -> None:
        # read the old values first: a user or movie can appear twice in one batch
        p_u, q_i, q_j = self.user_factors[u], self.movie_factors[i], self.movie_factors[j]
        b_i, b_j = self.movie_bias[i], self.movie_bias[j]
        # how much higher i scores than j
        score_difference = (p_u * (q_i - q_j)).sum(axis=1) + b_i - b_j
        # sigmoid(-difference): big when i is not above j yet, almost 0 when it already is
        weight = (1.0 / (1.0 + np.exp(score_difference)))[:, None]

        learning_rate, regularization = self.learning_rate, self.regularization
        # move u towards i and away from j; raise b_i, lower b_j; shrink everything a little
        # np.add.at instead of "+=": it also adds correctly when a user or movie appears twice
        np.add.at(
            self.user_factors, u, learning_rate * (weight * (q_i - q_j) - regularization * p_u)
        )
        np.add.at(self.movie_factors, i, learning_rate * (weight * p_u - regularization * q_i))
        np.add.at(self.movie_factors, j, learning_rate * (-weight * p_u - regularization * q_j))
        np.add.at(self.movie_bias, i, learning_rate * (weight[:, 0] - regularization * b_i))
        np.add.at(self.movie_bias, j, learning_rate * (-weight[:, 0] - regularization * b_j))

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors[user_indices] @ self.movie_factors.T + self.movie_bias

    def params(self) -> dict[str, Any]:
        return {
            "factors": self.n_factors,
            "reg": self.regularization,
            "lr": self.learning_rate,
            "epochs": self.n_epochs,
            "seed": self.seed,
        }
