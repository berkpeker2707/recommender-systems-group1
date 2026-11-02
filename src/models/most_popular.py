"""
Most Popular baseline
"""
import numpy as np

# from src.models.base import Recommender  # Part 2

SECONDS_PER_DAY = 86400


class MostPopular:  # Part 2: MostPopular(Recommender)
    def __init__(self, name: str, window_days=None, positives_only=False, relevant_threshold=4):
        self.name = name
        self.window_days = window_days
        self.positives_only = positives_only
        self.relevant_threshold = relevant_threshold

    # here we train the data, count how many ratings each movie has
    def fit(self, training_ratings, n_users, n_movies):
        counted_ratings = training_ratings
        if self.window_days is not None:
            # window ends at the last training interaction = the train cut
            # in other words, we count recent popularity for window_days days, not overall popularity
            newest_rating_time = counted_ratings["unix_time"].max()
            counted_ratings = counted_ratings[
                counted_ratings["unix_time"] >= newest_rating_time - self.window_days * SECONDS_PER_DAY]
        if self.positives_only:
            counted_ratings = counted_ratings[counted_ratings["rating"] >= self.relevant_threshold]
        self.rating_counts = np.bincount(counted_ratings["movie_index"].to_numpy(), minlength=n_movies).astype(np.float32)
        return self

    def score(self, user_indices):
        return np.broadcast_to(self.rating_counts, (len(user_indices), len(self.rating_counts))).copy()

    def params(self):
        return {"window_days": self.window_days, "positives_only": self.positives_only}
