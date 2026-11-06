"""
Most Popular baseline
"""
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd

# from src.models.base import Recommender  # Part 2

SECONDS_PER_DAY = 86400


@dataclass
class MostPopular:  # Part 2: MostPopular(Recommender)
    name: str
    window_days: int | None = None
    positives_only: bool = False
    relevant_threshold: int = 4
    rating_counts: np.ndarray = field(init=False, repr=False, compare=False)

    def fit(self, training_ratings: pd.DataFrame, _n_users: int, n_movies: int) -> Self:
        movies = training_ratings["movie_index"].to_numpy()
        is_positive = training_ratings["rating"].to_numpy() >= self.relevant_threshold

        def is_recent(days: int) -> np.ndarray:
            # window ends at the last train interaction, not "now"
            unix_times = training_ratings["unix_time"].to_numpy()
            return unix_times >= unix_times.max() - days * SECONDS_PER_DAY

        match self.window_days, self.positives_only:
            case None, False:
                counted_movies = movies
            case None, True:
                counted_movies = movies[is_positive]
            case days, False:
                counted_movies = movies[is_recent(days)]
            case days, True:
                counted_movies = movies[is_recent(days) & is_positive]

        self.rating_counts = np.bincount(counted_movies, minlength=n_movies).astype(np.float32)
        return self

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        # same scores for everyone, one row per user
        return np.tile(self.rating_counts, (len(user_indices), 1))

    def params(self) -> dict[str, int | bool | None]:
        return {"window_days": self.window_days, "positives_only": self.positives_only}
