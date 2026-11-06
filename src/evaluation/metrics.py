"""
Precision, recall, NDCG (binary relevance, gain 1/log2(rank+1)) and catalogue coverage.
ranked   = [7, 3, 9, 1, 5] # our list, best first
relevant = {3, 5, 8} # movies the user later rated 4 or 5 (the correct answers)
k = 5 # how many from the list we look at

"""
import numpy as np


# of the k movies we recommended, how many are correct answers?
# example: hits are 3 and 5, so 2 / 5 = 0.4 or 40%
def precision_at_k(ranked, relevant, k: int) -> float:
    return sum(1 for x in ranked[:k] if x in relevant) / k

# of the correct answers, how many did we find?
# 2 / 3 = 0.6667 or 66.67%
# Precision vs recall: Both count the same hits.
# Precision divides by list length (5), recall by number of correct answers (3).
def recall_at_k(ranked, relevant, k: int) -> float:
    # no correct answers would mean dividing by 0, so return 0.0
    # doesn't happen in practice, select_users_to_evaluate already drops these users
    if not relevant:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / len(relevant)

# ndcg means normalized discounted cumulative gain. it checks if the hits are near the top of the list.
# each position gets points: 1 / log2(position + 1) -> pos 1 = 1.0, pos 2 = 0.631, pos 3 = 0.5, pos 4 = 0.431, pos 5 = 0.387
# dcg: points of our list, only for the hits -> 3 at pos 2, 5 at pos 5 -> 0.631 + 0.387 = 1.018
# idcg: points of the perfect list, all 3 correct answers at pos 1, 2, 3 -> 1 + 0.631 + 0.5 = 2.131
# ndcg = dcg / idcg = 1.018 / 2.131 = 0.478, always between 0 and 1
def ndcg_at_k(ranked, relevant, k: int) -> float:
    # enumerate starts from 0, position starts from 1, so log2(position + 1) = log2(r + 2)
    dcg = sum(1.0 / np.log2(r + 2) for r, x in enumerate(ranked[:k]) if x in relevant)
    # min: the perfect list can't have more hits than the list length k
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# how many different movies we recommend to all users together, out of all movies
# example: lists [7, 3, 9], [7, 3, 2], [7, 9, 3] -> {7, 3, 9, 2} -> 4 / 10 movies = 0.4
# most popular gives almost the same list to everyone, so its coverage is low (3.6% vs 14.3% for item-kNN)
def catalogue_coverage(all_lists, n_items: int) -> float:
    # set keeps each movie only once
    seen = set()
    for lst in all_lists:
        seen.update(int(x) for x in lst)
    return len(seen) / n_items
