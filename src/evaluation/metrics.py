"""
Precision, recall, NDCG (binary relevance, gain 1/log2(rank+1)) and catalogue coverage.
"""
import numpy as np


def precision_at_k(ranked, relevant, k: int) -> float:
    return sum(1 for x in ranked[:k] if x in relevant) / k


def recall_at_k(ranked, relevant, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / len(relevant)


def ndcg_at_k(ranked, relevant, k: int) -> float:
    # rank r (1-based) gets gain 1/log2(r+1)
    dcg = sum(1.0 / np.log2(r + 2) for r, x in enumerate(ranked[:k]) if x in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def catalogue_coverage(all_lists, n_items: int) -> float:
    seen = set()
    for lst in all_lists:
        seen.update(int(x) for x in lst)
    return len(seen) / n_items
