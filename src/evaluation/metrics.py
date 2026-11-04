"""
Precision, recall, NDCG (binary relevance, gain 1/log2(rank+1)) and catalogue coverage.
ranked   = [7, 3, 9, 1, 5] # our list, best first
relevant = {3, 5, 8} # movies the user later rated 4 or 5 (the correct answers)
k = 5 # how many from the list we look at

"""
import numpy as np


# 2 / 5 = 0.4 or 40%
def precision_at_k(ranked, relevant, k: int) -> float:
    return sum(1 for x in ranked[:k] if x in relevant) / k

# 2 / 3 = 0.6667 or 66.67%
# Precision vs recall: Both count the same hits.
# Precision divides by list length (5), recall by number of correct answers (3).
def recall_at_k(ranked, relevant, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / len(relevant)

# always returns 0.0 if there are no relevant items, even if the list is empty
 
# ndcg means normalized discounted cumulative gain. It is a measure of ranking quality.
# example output means that the ranking quality is 61.31% of the ideal ranking quality, given the relevant items and the top k items in the ranked list.
# dcg is the sum of the gains of the relevant items in the ranked list, discounted by their rank. The gain of a relevant item at rank r is 1/log2(r+1). The ideal dcg (idcg) is the sum of the gains of the relevant items in the ideal ranking, which is the ranking that places all relevant items at the top. The ndcg is then dcg/idcg, which normalizes the score to be between 0 and 1.
# idcg is the ideal discounted cumulative gain, which is the maximum possible dcg for a given set of relevant items. It is calculated by summing the gains of the relevant items in the ideal ranking, which is the ranking that places all relevant items at the top. The gain of a relevant item at rank r is 1/log2(r+1). The idcg is used to normalize the dcg score, so that it can be compared across different rankings and sets of relevant items.
def ndcg_at_k(ranked, relevant, k: int) -> float:
    # rank r (1-based) gets gain 1/log2(r+1)
    dcg = sum(1.0 / np.log2(r + 2) for r, x in enumerate(ranked[:k]) if x in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# catalogue coverage is the fraction of items that are recommended to users, out of the total number of items.
# It measures how well the recommender system covers the item space.
# A higher catalogue coverage means that more items are being recommended, which can be beneficial for both users and item providers.
def catalogue_coverage(all_lists, n_items: int) -> float:
    seen = set()
    for lst in all_lists:
        seen.update(int(x) for x in lst)
    return len(seen) / n_items
