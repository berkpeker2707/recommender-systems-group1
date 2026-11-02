"""
Item-based kNN baseline.

A movie's score for a user = the sum of its similarity to the movies the user rated.
Each movie keeps only its most similar neighbours; negative similarities are dropped.
"""
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.sparse as sp

# from src.models.base import Recommender  # Part 2
from src.split_data import build_user_movie_matrix


def cosine_from_dot_products(dot_products: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.diag(dot_products))
    return dot_products / np.maximum(np.outer(norm, norm), 1e-12)


def compute_movie_similarity(
    training_ratings: pd.DataFrame, n_users: int, n_movies: int, similarity_measure: str
) -> np.ndarray:
    """Dense movie x movie similarity, zero diagonal."""
    rated_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, values="binary")

    match similarity_measure:
        case "cosine":
            # binary ratings, so the dot product is just the co-rating count
            similarity = cosine_from_dot_products((rated_matrix.T @ rated_matrix).toarray())
        case "jaccard":
            rated_both_counts = (rated_matrix.T @ rated_matrix).toarray()
            rating_counts = np.diag(rated_both_counts)
            rated_either_counts = rating_counts[:, None] + rating_counts[None, :] - rated_both_counts
            similarity = rated_both_counts / np.maximum(rated_either_counts, 1e-12)
        case "adjusted_cosine":
            # center per user so harsh and generous raters are comparable
            rating_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, values="rating")
            ratings_per_user = np.diff(rating_matrix.indptr)
            user_means = np.asarray(rating_matrix.sum(axis=1)).ravel() / np.maximum(ratings_per_user, 1)
            centered_ratings = rating_matrix.copy()
            centered_ratings.data = centered_ratings.data - np.repeat(user_means, ratings_per_user).astype(np.float32)
            similarity = cosine_from_dot_products((centered_ratings.T @ centered_ratings).toarray())
        case _:
            raise ValueError(f"unknown similarity {similarity_measure!r}")

    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32)


def keep_nearest_neighbors(similarity: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    """Row i keeps the n_neighbors largest positive entries, i.e. the neighbours of movie i."""
    n_movies = similarity.shape[0]
    n_neighbors = min(n_neighbors, n_movies - 1)
    neighbor_columns = np.argpartition(-similarity, n_neighbors, axis=1)[:, :n_neighbors]
    neighbor_values = np.take_along_axis(similarity, neighbor_columns, axis=1)
    neighbor_values = np.where(neighbor_values > 0, neighbor_values, 0.0)
    neighbor_rows = np.repeat(np.arange(n_movies), n_neighbors)
    neighbors = sp.csr_matrix((neighbor_values.ravel(), (neighbor_rows, neighbor_columns.ravel())),
                              shape=similarity.shape)
    neighbors.eliminate_zeros()
    return neighbors


@dataclass
class ItemKNN:  # Part 2: ItemKNN(Recommender)
    n_neighbors: int
    similarity_measure: str = "cosine"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"item_knn_{self.similarity_measure}_k{self.n_neighbors}"

    def fit(self, training_ratings: pd.DataFrame, n_users: int, n_movies: int) -> Self:
        similarity = compute_movie_similarity(training_ratings, n_users, n_movies, self.similarity_measure)
        self.neighbor_similarity = keep_nearest_neighbors(similarity, self.n_neighbors)
        self.rated_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, values="binary")
        return self

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        # sum each movie's similarity to everything the user rated
        return np.asarray((self.rated_matrix[user_indices] @ self.neighbor_similarity.T).todense())

    def params(self) -> dict[str, Any]:
        return {"k": self.n_neighbors, "similarity": self.similarity_measure}
