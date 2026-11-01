"""
evaluator.py — full-catalogue evaluation, identical for every model.

evaluate(model, train_csr, ground_truth, users, k=10) -> summary, per-user metric table
1. scores = model.score(users)                     # (n_users, n_items)
2. mask items seen in train (set to -inf)
3. top-k via a stable sort: ties broken by lower item index (deterministic)
4. compute P@k, R@k, NDCG@k per user; coverage over all lists
Candidate set = ALL unseen items (no sampled negatives — Lecture 3.1 slide 5).

Returns per-user results (not only means) so we can compute confidence intervals.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import catalogue_coverage, ndcg_at_k, precision_at_k, recall_at_k


def top_k(model, train_csr, users, k: int):
    scores = np.asarray(model.score(users), dtype=np.float64)
    scores[train_csr[users].nonzero()] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate(model, train_csr, ground_truth: dict, users, k: int = 10):
    users = np.asarray(users)
    lists = top_k(model, train_csr, users, k)
    rows = []
    for u, ranked in zip(users, lists):
        rel = ground_truth[int(u)]
        ranked = ranked.tolist()
        rows.append((int(u), precision_at_k(ranked, rel, k), recall_at_k(ranked, rel, k),
                     ndcg_at_k(ranked, rel, k)))
    per_user = pd.DataFrame(rows, columns=["u", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"])
    summary = per_user.drop(columns="u").mean().to_dict()
    summary[f"coverage@{k}"] = catalogue_coverage(lists, train_csr.shape[1])
    summary["n_users"] = len(users)
    return summary, per_user  # Part 3: also return lists, re-ranking needs them
