"""
test_metrics.py — prove our metrics agree with the reference (brief: own metrics allowed
if shown to agree on a sample).

Lecture 3.1 slide 8, relevant = {i1, i7}:
  list B [i1,i7,i9,i4,i3] -> NDCG@5 = 1.00
  list A [i4,i1,i9,i7,i3] -> NDCG@5 ≈ 0.65, P@5 = 0.40, R@5 = 1.00
  list C [i9,i3,i4,i1,i7] -> NDCG@5 ≈ 0.50
Run: pytest -q
"""
import pytest

from src.evaluation.metrics import catalogue_coverage, ndcg_at_k, precision_at_k, recall_at_k

REL = {"i1", "i7"}
A = ["i4", "i1", "i9", "i7", "i3"]
B = ["i1", "i7", "i9", "i4", "i3"]
C = ["i9", "i3", "i4", "i1", "i7"]


def test_slide8_ndcg():
    assert ndcg_at_k(B, REL, 5) == pytest.approx(1.00)
    assert ndcg_at_k(A, REL, 5) == pytest.approx(0.65, abs=0.01)
    assert ndcg_at_k(C, REL, 5) == pytest.approx(0.50, abs=0.01)


def test_slide8_precision_recall():
    assert precision_at_k(A, REL, 5) == pytest.approx(0.40)
    assert recall_at_k(A, REL, 5) == pytest.approx(1.00)
    assert recall_at_k(A, REL, 2) == pytest.approx(0.50)


def test_edge_cases():
    assert ndcg_at_k(A, set(), 5) == 0.0
    assert recall_at_k(A, set(), 5) == 0.0
    assert ndcg_at_k(["x"], {"x"}, 10) == pytest.approx(1.0)   # IDCG capped by |relevant|
    assert catalogue_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
