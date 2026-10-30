"""
metrics.py — ranking metrics, Lecture 3.1 slides 7–9. Pure functions, unit-tested.

- precision_at_k(ranked, relevant, k)  = |hits in top k| / k
- recall_at_k(ranked, relevant, k)     = |hits in top k| / |relevant|
- ndcg_at_k(ranked, relevant, k)       = DCG / IDCG, gain 1/log2(rank+1), binary relevance
- catalogue_coverage(all_lists, n_items) = |∪ recommended items| / n_items   (system-level)

Validation (brief allows own metrics if shown to agree with a reference):
tests/test_metrics.py reproduces slide 8 (lists A/B/C -> NDCG@5 = 0.65 / 1.00 / 0.50).
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
