"""
Most Popular baseline
"""
import numpy as np

from src.models.base import Recommender


class MostPopular(Recommender):
    def __init__(self, name: str, window_days=None, positives_only=False, threshold=4):
        self.name = name
        self.window_days = window_days
        self.positives_only = positives_only
        self.threshold = threshold

    # here we train the data, count how many ratings each movie has
    def fit(self, train, n_users, n_items):
        df = train
        if self.window_days is not None:
            # window ends at the last training interaction = the train cut
            # in other words, we count recent popular for 30 days, not overall popularity
            df = df[df["ts"] >= df["ts"].max() - self.window_days * 86400]
        if self.positives_only:
            df = df[df["rating"] >= self.threshold]
        self.counts = np.bincount(df["i"].to_numpy(), minlength=n_items).astype(np.float32)
        return self

    def score(self, user_indices):
        return np.broadcast_to(self.counts, (len(user_indices), len(self.counts))).copy()

    def params(self):
        return {"window_days": self.window_days, "positives_only": self.positives_only}
