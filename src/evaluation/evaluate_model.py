"""
Evaluates any model the same way: rank all unseen movies, take the top of the list, compute the metrics.
Ties go to the lower movie index, so results are the same on every run.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import catalogue_coverage, ndcg_at_k, precision_at_k, recall_at_k


def recommend_top_movies(model, seen_movies, users, list_size: int):
    scores = np.asarray(model.score(users), dtype=np.float64)
    scores[seen_movies[users].nonzero()] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :list_size]


def evaluate_model(model, seen_movies, relevant_movies: dict, users, list_size: int = 10):
    users = np.asarray(users)
    recommendation_lists = recommend_top_movies(model, seen_movies, users, list_size)
    rows = []
    for user_index, recommended in zip(users, recommendation_lists):
        relevant = relevant_movies[int(user_index)]
        recommended = recommended.tolist()
        rows.append((int(user_index), precision_at_k(recommended, relevant, list_size),
                     recall_at_k(recommended, relevant, list_size), ndcg_at_k(recommended, relevant, list_size)))
    per_user = pd.DataFrame(rows, columns=["user_index", f"precision@{list_size}", f"recall@{list_size}",
                                           f"ndcg@{list_size}"])
    summary = per_user.drop(columns="user_index").mean().to_dict()
    summary[f"coverage@{list_size}"] = catalogue_coverage(recommendation_lists, seen_movies.shape[1])
    summary["n_users"] = len(users)
    return summary, per_user  # Part 3: also return recommendation_lists, re-ranking needs them
