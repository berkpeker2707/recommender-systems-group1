"""
Content-based: no other users are used, only the genres and release decade of the movies.
  movie vector = 18 genre columns + 7 decade columns (0/1, decades times year_weight)
  user profile = sum of the vectors of the movies the user rated, each with a weight
  score        = cosine similarity between the profile and each movie vector

profile_weighting = the weight of each rated movie:
  "centered": rating - the user's mean rating, so disliked movies count negative
  "liked"   : 1 for 4-5 stars, otherwise 0
  "all"     : 1 for every rated movie
"""

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from src.models.base import Recommender
from src.split_data import build_user_movie_matrix


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Every row divided by its length, so a dot product of two rows is their cosine."""
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


@dataclass
class ContentBased(Recommender):
    genre_matrix: np.ndarray = field(repr=False)
    decade_matrix: np.ndarray = field(repr=False)
    year_weight: float = 1.0
    profile_weighting: str = "centered"
    relevant_threshold: int = 4
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"content_{self.profile_weighting}_year{self.year_weight}"

    def fit(self, training_ratings: pd.DataFrame, n_users: int, n_movies: int) -> Self:
        self.movie_vectors = np.hstack([self.genre_matrix, self.year_weight * self.decade_matrix])

        # user x movie table of weights, filled with the ratings first
        weights = build_user_movie_matrix(training_ratings, n_users, n_movies, values="rating")
        ratings_per_user = np.diff(weights.indptr)
        match self.profile_weighting:
            case "centered":
                # example: mean 2.25, so a rating of 3 counts +0.75 and a rating of 1 counts -1.25
                user_means = np.asarray(weights.sum(axis=1)).ravel() / np.maximum(
                    ratings_per_user, 1
                )
                weights.data -= np.repeat(user_means, ratings_per_user).astype(np.float32)
            case "liked":
                weights.data = (weights.data >= self.relevant_threshold).astype(np.float32)
            case "all":
                weights.data = np.ones_like(weights.data)
            case _:
                raise ValueError(f"unknown profile weighting {self.profile_weighting!r}")
        # profile = sum of weight x movie vector, one row per user
        self.user_profiles = np.asarray(weights @ self.movie_vectors)
        return self

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return _unit_rows(self.user_profiles[user_indices]) @ _unit_rows(self.movie_vectors).T

    def params(self) -> dict[str, Any]:
        return {"year_weight": self.year_weight, "profile": self.profile_weighting}
