"""
User-based kNN: find the users most similar to you and recommend what they rated.
A movie's score = the sum of the similarities of your neighbours who rated it.
"""

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from src.models.base import Recommender
from src.split_data import build_user_movie_matrix


@dataclass
class UserKNN(Recommender):
    n_neighbors: int
    similarity_measure: str = "cosine"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"user_knn_{self.similarity_measure}_k{self.n_neighbors}"

    def fit(self, training_ratings: pd.DataFrame, n_users: int, n_movies: int) -> Self:
        # user x movie table, 1 = rated; and how many movies each user rated
        self.rated_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, "binary")
        self.rating_counts = np.asarray(self.rated_matrix.sum(axis=1)).ravel()
        self.compare_matrix = self.rated_matrix
        if self.similarity_measure == "pearson":
            # pearson = cosine after subtracting each user's mean rating
            ratings = build_user_movie_matrix(training_ratings, n_users, n_movies, "rating")
            ratings_per_user = np.diff(ratings.indptr)
            user_means = np.asarray(ratings.sum(axis=1)).ravel() / np.maximum(ratings_per_user, 1)
            ratings.data -= np.repeat(user_means, ratings_per_user).astype(np.float32)
            self.compare_matrix = ratings
        # length of each user's row, for cosine and pearson
        squared = self.compare_matrix.multiply(self.compare_matrix)
        self.row_lengths = np.sqrt(np.asarray(squared.sum(axis=1)).ravel())
        return self

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        user_indices = np.asarray(user_indices)
        # 1. similarity to every other user; with 0/1 rows the dot product = movies rated by both
        dot_products = (self.compare_matrix[user_indices] @ self.compare_matrix.T).toarray()
        match self.similarity_measure:
            case "cosine" | "pearson":
                lengths = np.outer(self.row_lengths[user_indices], self.row_lengths)
                similarity = dot_products / np.maximum(lengths, 1e-12)
            case "jaccard":
                # rated both / rated at least one
                counts = self.rating_counts
                rated_either = counts[user_indices][:, None] + counts[None, :] - dot_products
                similarity = dot_products / np.maximum(rated_either, 1e-12)
            case _:
                raise ValueError(f"unknown similarity {self.similarity_measure!r}")
        similarity[np.arange(len(user_indices)), user_indices] = 0.0  # not your own neighbour

        # 2. keep the n_neighbors most similar users; every other user gets weight 0
        n_neighbors = min(self.n_neighbors, similarity.shape[1] - 1)
        neighbor_columns = np.argsort(-similarity, axis=1, kind="stable")[:, :n_neighbors]
        neighbor_weights = np.zeros_like(similarity)
        np.put_along_axis(
            neighbor_weights,
            neighbor_columns,
            np.take_along_axis(similarity, neighbor_columns, axis=1),
            axis=1,
        )
        neighbor_weights = neighbor_weights.clip(min=0)  # drop negative similarities

        # 3. movie score = sum of the similarities of the neighbours who rated it
        # example: neighbours A (0.5) and B (0.3) rated the movie, C did not -> 0.8
        return neighbor_weights @ self.rated_matrix

    def params(self) -> dict[str, Any]:
        return {"k": self.n_neighbors, "similarity": self.similarity_measure}
