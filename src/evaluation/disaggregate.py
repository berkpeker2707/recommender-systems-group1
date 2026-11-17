"""
Splits one model's test result into groups, to see who and what the model serves well.
The top-10 lists stay the same; only the users or the correct answers we look at change.
  user activity  : users with few vs many training ratings (cut at the median)
  item popularity: correct answers among the 20% most rated movies vs the rest (long tail)
  test period    : correct answers from the first vs the second half of the test period
"""

import numpy as np
import pandas as pd

from src.evaluation.confidence_interval import mean_with_confidence_interval
from src.evaluation.evaluate_model import evaluate_lists
from src.split_data import find_relevant_movies


def disaggregate(
    lists,
    users,
    relevant_movies,
    training_ratings,
    test_ratings,
    n_movies,
    relevant_threshold,
    head_share,
):
    """One row per (dimension, group) for one model's test lists."""
    users = np.asarray(users)
    # each group: (dimension, group, definition, correct answers per user, movies counted for coverage)
    groups = []

    # 1. user activity
    user_counts = training_ratings.groupby("user_index").size().reindex(users).to_numpy()
    median = int(np.median(user_counts))
    low = user_counts <= median
    for group, in_group, sign in [("low activity", low, "<="), ("high activity", ~low, ">")]:
        group_relevant = {int(u): relevant_movies[int(u)] for u in users[in_group]}
        groups.append(
            (
                "user activity",
                group,
                f"training ratings {sign} {median} (median)",
                group_relevant,
                None,
            )
        )

    # 2. item popularity: popular = the head_share of movies with the most training ratings
    counts = np.bincount(training_ratings["movie_index"], minlength=n_movies)
    popular = np.argsort(-counts, kind="stable")[: round(head_share * n_movies)]
    is_popular = np.zeros(n_movies, dtype=bool)
    is_popular[popular] = True
    for group, wanted, definition in [
        (
            "popular",
            True,
            f"top {head_share:.0%} of movies by training ratings (>= {counts[popular].min()})",
        ),
        ("long tail", False, f"the other {1 - head_share:.0%} of movies"),
    ]:
        # keep only the correct answers of this group; skip users with none left
        group_relevant = {
            u: {m for m in movies if is_popular[m] == wanted}
            for u, movies in relevant_movies.items()
        }
        group_relevant = {u: movies for u, movies in group_relevant.items() if movies}
        groups.append(
            (
                "item popularity",
                group,
                definition,
                group_relevant,
                np.flatnonzero(is_popular == wanted),
            )
        )

    # 3. test period, split at the median test timestamp
    middle = int(np.median(test_ratings["unix_time"]))
    middle_date = pd.to_datetime(middle, unit="s").date()
    is_early = test_ratings["unix_time"] < middle
    for group, period, sign in [("early", is_early, "before"), ("late", ~is_early, "from")]:
        group_relevant = find_relevant_movies(test_ratings[period], relevant_threshold, users)
        groups.append(
            ("test period", group, f"test ratings {sign} {middle_date}", group_relevant, None)
        )

    rows = []
    for dimension, group, definition, group_relevant, catalogue in groups:
        keep = np.isin(users, list(group_relevant))
        summary, per_user = evaluate_lists(lists[keep], users[keep], group_relevant, n_movies)
        if catalogue is not None:
            # coverage inside the group, e.g. the share of long-tail movies recommended at least once
            recommended = set(lists[keep].ravel().tolist())
            summary["coverage@10"] = len(recommended & set(catalogue.tolist())) / len(catalogue)
        _, interval = mean_with_confidence_interval(per_user["ndcg@10"])
        rows.append(
            {
                "dimension": dimension,
                "group": group,
                "definition": definition,
                **summary,
                "ndcg@10_ci": interval,
            }
        )
    return rows
