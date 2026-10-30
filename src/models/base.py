"""
base.py — common interface so every model is evaluated by the SAME code.

class Recommender:
    name: str
    def fit(self, train, n_users, n_items) -> self      # train = ratings DataFrame with u, i, rating, ts
    def score(self, user_indices) -> np.ndarray   # shape (len(users), n_items), higher = better
The evaluator handles masking of seen items and top-k selection, so models only
produce scores. This keeps 'full catalogue ranking' identical for all models.
Models get the DataFrame (not only a matrix) because some need timestamps
(recency popularity) or ratings (adjusted cosine); item features come via __init__.
"""
import numpy as np
import pandas as pd


class Recommender:
    name = "base"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "Recommender":
        raise NotImplementedError

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        """Hyperparameters, written to results.csv."""
        return {}
