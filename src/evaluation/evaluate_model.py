"""
Evaluates any model the same way: rank all unseen movies, take the top of the list, compute the metrics.
Ties go to the lower movie index, so results are the same on every run.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import catalogue_coverage, ndcg_at_k, precision_at_k, recall_at_k


# returns the top 10 movie_index for each user, one row per user
# example: [[260, 1196, 2571, ...], [...]]
def _recommend_top_movies(model, seen_movies, users, list_size: int):
    # the model gives a score to every movie for every user (users x 3883)
    scores = np.asarray(model.score(users), dtype=np.float64)
    # movies the user already saw in training get -inf, so they can never be in the top 10
    scores[seen_movies[users].nonzero()] = -np.inf
    # sort from highest score to lowest (that's why -scores) and take the first 10
    # stable: if two movies have the same score, the lower movie index comes first
    return np.argsort(-scores, axis=1, kind="stable")[:, :list_size]


# example return: {"precision@10": 0.200, "recall@10": 0.065, "ndcg@10": 0.219, "coverage@10": 0.143, "n_users": 1154}
def evaluate_model(model, seen_movies, relevant_movies: dict, users, list_size: int = 10):
    users = np.asarray(users)
    recommendation_lists = _recommend_top_movies(model, seen_movies, users, list_size)
    rows = []
    # go through users one by one, together with their top 10 list
    for user_index, recommended in zip(users, recommendation_lists):
        # the correct answers of this user (movies rated 4 or 5)
        relevant = relevant_movies[int(user_index)]
        recommended = recommended.tolist()
        rows.append((int(user_index), precision_at_k(recommended, relevant, list_size),
                     recall_at_k(recommended, relevant, list_size), ndcg_at_k(recommended, relevant, list_size)))
    # one row per user with the 3 metrics, used for the confidence interval
    per_user = pd.DataFrame(rows, columns=["user_index", f"precision@{list_size}", f"recall@{list_size}",
                                           f"ndcg@{list_size}"])
    # average over all users
    summary = per_user.drop(columns="user_index").mean().to_dict()
    # coverage looks at all lists together, so it's not per user
    summary[f"coverage@{list_size}"] = catalogue_coverage(recommendation_lists, seen_movies.shape[1])
    summary["n_users"] = len(users)
    return summary, per_user  # Part 3: also return recommendation_lists, re-ranking needs them
