"""
split.py — temporal train / validation / test split (Lecture 3.1 slide 4).

Protocol (decisions to justify in the report, 1–2 sentences each):
1. Sort all ratings by timestamp. Cut at the interaction quantiles in
   config.SPLIT_FRACTIONS -> t1 (train|val) and t2 (val|test). Log t1, t2 as dates.
   Why global temporal: mirrors deployment, no future leakage (random split leaks).
2. Validation is used ONLY for tuning; test is touched once per final model.
3. For final test evaluation, retrain on train+validation with the chosen
   hyperparameters (decision to confirm as a team).
4. Users evaluated in val/test must have >= MIN_TRAIN_INTERACTIONS in the training
   data; report how many users and rows this drops. Items are not filtered.
5. Ground truth per user = items in the evaluated split with rating >= POSITIVE_THRESHOLD.
   Users with no relevant item are skipped (report count).

Functions:
- temporal_split(ratings, fractions) -> train, val, test, cut_info
- eval_users(train, target, min_train, threshold) -> array of users + drop statistics
- ground_truth(target, threshold) -> dict user -> set(items)
- to_csr(train, n_users, n_items) -> scipy.sparse user×item matrix
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def temporal_split(ratings: pd.DataFrame, fractions=(0.8, 0.1, 0.1)):
    """Global cut by interaction quantiles. Ties are cut by timestamp, never by position,
    so two ratings in the same second always land in the same split (no leakage)."""
    assert abs(sum(fractions) - 1) < 1e-9
    ts = ratings["ts"].to_numpy()
    sorted_ts = np.sort(ts)
    n = len(sorted_ts)
    t1 = sorted_ts[int(round(fractions[0] * n))]
    t2 = sorted_ts[int(round((fractions[0] + fractions[1]) * n))]
    train = ratings[ts < t1]
    val = ratings[(ts >= t1) & (ts < t2)]
    test = ratings[ts >= t2]
    cut_info = {
        "t1": str(pd.to_datetime(t1, unit="s")),
        "t2": str(pd.to_datetime(t2, unit="s")),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "frac_train": len(train) / n, "frac_val": len(val) / n, "frac_test": len(test) / n,
    }
    return train, val, test, cut_info


def eval_users(train: pd.DataFrame, target: pd.DataFrame, min_train: int, threshold: int):
    """Users evaluable on `target`: >= min_train train interactions AND >= 1 relevant target item.
    Returns (sorted array of u indices, drop statistics for the report)."""
    train_counts = train.groupby("u").size()
    target_users = pd.Index(target["u"].unique())
    rel_users = pd.Index(target.loc[target["rating"] >= threshold, "u"].unique())

    known = target_users[target_users.isin(train_counts.index)]
    warm = known[train_counts.reindex(known).to_numpy() >= min_train]
    keep = np.sort(warm[warm.isin(rel_users)].to_numpy())

    stats = {
        "target_users": len(target_users),
        "dropped_cold_new": int(len(target_users) - len(known)),          # 0 train interactions
        "dropped_cold_few": int(len(known) - len(warm)),                  # 1..min_train-1
        "dropped_no_relevant": int(len(warm) - len(keep)),
        "evaluated_users": int(len(keep)),
        "target_rows": len(target),
        "target_rows_kept": int(target["u"].isin(keep).sum()),
    }
    return keep, stats


def ground_truth(target: pd.DataFrame, threshold: int, users=None) -> dict:
    rel = target[target["rating"] >= threshold]
    if users is not None:
        rel = rel[rel["u"].isin(users)]
    return {int(u): set(g.tolist()) for u, g in rel.groupby("u")["i"]}


def to_csr(df: pd.DataFrame, n_users: int, n_items: int, values: str = "rating") -> sp.csr_matrix:
    """User × item matrix. values='rating' keeps 1–5 ratings, 'binary' stores 1.0."""
    data = df["rating"].to_numpy(np.float32) if values == "rating" else np.ones(len(df), np.float32)
    return sp.csr_matrix((data, (df["u"].to_numpy(), df["i"].to_numpy())), shape=(n_users, n_items))
