"""
Checks the metrics against a worked example, relevant = {i1, i7}:
  list B [i1,i7,i9,i4,i3] -> NDCG@5 = 1.00
  list A [i4,i1,i9,i7,i3] -> NDCG@5 = 0.65, P@5 = 0.40, R@5 = 1.00
  list C [i9,i3,i4,i1,i7] -> NDCG@5 = 0.50
"""
import pytest

from src.evaluation.metrics import catalogue_coverage, ndcg_at_k, precision_at_k, recall_at_k

# run with: python -m pytest tests -q
# the same 2 correct answers in 3 different orders, to see if ndcg cares about the position
REL = {"i1", "i7"}
A = ["i4", "i1", "i9", "i7", "i3"]
B = ["i1", "i7", "i9", "i4", "i3"]
C = ["i9", "i3", "i4", "i1", "i7"]


def test_ndcg_worked_example():
    # B: hits at 1 and 2, the perfect list
    assert ndcg_at_k(B, REL, 5) == pytest.approx(1.00)
    # A: hits at 2 and 4 -> (0.631 + 0.431) / (1 + 0.631) = 0.65
    assert ndcg_at_k(A, REL, 5) == pytest.approx(0.65, abs=0.01)
    # C: hits at 4 and 5 -> (0.431 + 0.387) / (1 + 0.631) = 0.50
    assert ndcg_at_k(C, REL, 5) == pytest.approx(0.50, abs=0.01)


def test_precision_recall_worked_example():
    # 2 hits out of 5 recommended
    assert precision_at_k(A, REL, 5) == pytest.approx(0.40)
    # found both correct answers
    assert recall_at_k(A, REL, 5) == pytest.approx(1.00)
    # only the first 2 ([i4, i1]): found 1 of 2
    assert recall_at_k(A, REL, 2) == pytest.approx(0.50)


def test_edge_cases():
    # no correct answers -> 0, not a division by 0 error
    assert ndcg_at_k(A, set(), 5) == 0.0
    assert recall_at_k(A, set(), 5) == 0.0
    # 1 correct answer found at position 1 is perfect, even with k = 10
    assert ndcg_at_k(["x"], {"x"}, 10) == pytest.approx(1.0)   # IDCG capped by |relevant|
    # {1, 2, 3} -> 3 different movies out of 10
    assert catalogue_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
