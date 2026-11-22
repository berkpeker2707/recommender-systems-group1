"""
Every model has the same three functions, so the evaluation code can treat them all the same way.
"""

import numpy as np
import pandas as pd


class Recommender:
    name = "base"

    # learn from the training ratings (columns user_index, movie_index, rating, unix_time)
    def fit(self, training_ratings: pd.DataFrame, n_users: int, n_movies: int) -> "Recommender":
        raise NotImplementedError

    # one row per user, one column per movie; higher score = recommend earlier
    def score(self, user_indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        """Hyperparameters, written to results.csv."""
        return {}
