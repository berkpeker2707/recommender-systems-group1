"""
Item-based kNN baseline.

A movie's score for a user = the sum of its similarity to the movies the user rated.
Each movie keeps only its most similar neighbours; negative similarities are dropped.
"""
import numpy as np
import scipy.sparse as sp

# from src.models.base import Recommender  # Part 2
from src.split_data import build_user_movie_matrix


def compute_movie_similarity(training_ratings, n_users, n_movies, similarity_measure: str) -> np.ndarray:
    """Full (dense) movie × movie similarity with zero diagonal."""
    rated_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, values="binary")
    rated_both_counts = (rated_matrix.T @ rated_matrix).toarray()  # [a, b] = #users who rated both
    rating_counts = np.diag(rated_both_counts).copy()

    if similarity_measure == "cosine":
        norm = np.sqrt(rating_counts)
        similarity = rated_both_counts / np.maximum(np.outer(norm, norm), 1e-12)
    elif similarity_measure == "jaccard":
        rated_either_counts = rating_counts[:, None] + rating_counts[None, :] - rated_both_counts
        similarity = rated_both_counts / np.maximum(rated_either_counts, 1e-12)
    elif similarity_measure == "adjusted_cosine":
        rating_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, values="rating")
        ratings_per_user = np.diff(rating_matrix.indptr)
        user_means = np.asarray(rating_matrix.sum(axis=1)).ravel() / np.maximum(ratings_per_user, 1)
        centered_ratings = rating_matrix.copy()
        centered_ratings.data = centered_ratings.data - np.repeat(user_means, ratings_per_user).astype(np.float32)
        dot_products = (centered_ratings.T @ centered_ratings).toarray()
        norm = np.sqrt(np.diag(dot_products))
        similarity = dot_products / np.maximum(np.outer(norm, norm), 1e-12)
    else:
        raise ValueError(f"unknown similarity {similarity_measure!r}")

    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32)


def keep_nearest_neighbors(similarity: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    """Row i keeps its n_neighbors largest positive entries (= the nearest neighbours of movie i)."""
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


class ItemKNN:  # Part 2: ItemKNN(Recommender)
    def __init__(self, n_neighbors: int, similarity_measure: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.similarity_measure = similarity_measure
        self.name = f"item_knn_{similarity_measure}_k{n_neighbors}"

    def fit(self, training_ratings, n_users, n_movies):
        similarity = compute_movie_similarity(training_ratings, n_users, n_movies, self.similarity_measure)
        self.neighbor_similarity = keep_nearest_neighbors(similarity, self.n_neighbors)  # row i = neighbours of movie i
        self.rated_matrix = build_user_movie_matrix(training_ratings, n_users, n_movies, values="binary")
        return self

    def score(self, user_indices):
        # for each movie: add up its similarity to every movie the user rated
        return np.asarray((self.rated_matrix[user_indices] @ self.neighbor_similarity.T).todense())

    def params(self):
        return {"k": self.n_neighbors, "similarity": self.similarity_measure}
