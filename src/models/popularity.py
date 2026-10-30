"""
popularity.py — Most Popular baseline (Part 1, baseline 2). Lecture 1.1 slide 9.

No hyperparameters, but explicit choices (report >= 2 variants, see config.POPULAR_VARIANTS):
- all-time vs recent window (last N days before the train cut)
- count all ratings vs count only positives (rating >= 4)
Same score vector for every user; seen items are masked by the evaluator,
so each user still gets a different list. Ties broken by item id (deterministic).

Expectation (Lecture 1.1): strong on accuracy, very low coverage -> discuss.
"""
import numpy as np

from src.models.base import Recommender


class MostPopular(Recommender):
    def __init__(self, name: str, window_days=None, positives_only=False, threshold=4):
        self.name = name
        self.window_days = window_days
        self.positives_only = positives_only
        self.threshold = threshold

    def fit(self, train, n_users, n_items):
        df = train
        if self.window_days is not None:
            # window ends at the last training interaction = the train cut
            df = df[df["ts"] >= df["ts"].max() - self.window_days * 86400]
        if self.positives_only:
            df = df[df["rating"] >= self.threshold]
        self.counts = np.bincount(df["i"].to_numpy(), minlength=n_items).astype(np.float32)
        return self

    def score(self, user_indices):
        return np.broadcast_to(self.counts, (len(user_indices), len(self.counts))).copy()

    def params(self):
        return {"window_days": self.window_days, "positives_only": self.positives_only}
