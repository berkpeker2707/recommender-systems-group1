"""
item_knn.py — Item-based kNN (Part 1, baseline 1). Lecture 1.2 slides 5–10.

Top-N version for implicit feedback (own implementation):
1. Item-item similarity from the train matrix (columns = items):
     cosine on binary vectors | Jaccard | adjusted cosine on user-mean-centred ratings.
2. Keep only the k most similar neighbours per item (sparsify). Negative similarities
   (possible with adjusted cosine) are dropped: in a top-N sum they would only push
   items down for reasons unrelated to the user's taste for them.
3. Optional shrinkage: sim * min(n_common, γ)/γ (significance weighting, slide 10).
4. score(u, i) = Σ_{j ∈ history(u)} sim_k(i, j)     (binary history)
Tuning: grid over k × similarity on VALIDATION NDCG@10 (config grids).

Implementation: scipy.sparse; the 3.7k × 3.7k similarity matrix fits easily in memory.
"""
import numpy as np
import scipy.sparse as sp

from src.models.base import Recommender
from src.split import to_csr


def item_similarity(train, n_users, n_items, similarity: str, shrinkage: int = 0) -> np.ndarray:
    """Full (dense) item × item similarity with zero diagonal."""
    X = to_csr(train, n_users, n_items, values="binary")
    co = (X.T @ X).toarray()                       # co[i, j] = #users who rated both
    cnt = np.diag(co).copy()

    if similarity == "cosine":
        norm = np.sqrt(cnt)
        S = co / np.maximum(np.outer(norm, norm), 1e-12)
    elif similarity == "jaccard":
        S = co / np.maximum(cnt[:, None] + cnt[None, :] - co, 1e-12)
    elif similarity == "adjusted_cosine":
        R = to_csr(train, n_users, n_items, values="rating")
        user_mean = np.asarray(R.sum(axis=1)).ravel() / np.maximum(np.diff(R.indptr), 1)
        Rc = R.copy()
        Rc.data = Rc.data - np.repeat(user_mean, np.diff(R.indptr)).astype(np.float32)
        num = (Rc.T @ Rc).toarray()
        norm = np.sqrt(np.diag(num))
        S = num / np.maximum(np.outer(norm, norm), 1e-12)
    else:
        raise ValueError(f"unknown similarity {similarity!r}")

    if shrinkage > 0:
        S = S * np.minimum(co, shrinkage) / shrinkage
    np.fill_diagonal(S, 0.0)
    return S.astype(np.float32)


def keep_top_k(S: np.ndarray, k: int) -> sp.csr_matrix:
    """Row i keeps its k largest positive entries (= the k nearest neighbours of item i)."""
    n = S.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    vals = np.take_along_axis(S, idx, axis=1)
    vals = np.where(vals > 0, vals, 0.0)
    rows = np.repeat(np.arange(n), k)
    out = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=S.shape)
    out.eliminate_zeros()
    return out


class ItemKNN(Recommender):
    def __init__(self, k: int, similarity: str = "cosine", shrinkage: int = 0):
        self.k = k
        self.similarity = similarity
        self.shrinkage = shrinkage
        self.name = f"item_knn_{similarity}_k{k}"

    def fit(self, train, n_users, n_items, sim_full: np.ndarray = None):
        """sim_full: optional precomputed item_similarity(...) to avoid recomputing it for every k."""
        if sim_full is None:
            sim_full = item_similarity(train, n_users, n_items, self.similarity, self.shrinkage)
        self.S_k = keep_top_k(sim_full, self.k)          # row i = neighbours of target item i
        self.X = to_csr(train, n_users, n_items, values="binary")
        return self

    def score(self, user_indices):
        # scores[u, i] = Σ_j X[u, j] * S_k[i, j]
        return np.asarray((self.X[user_indices] @ self.S_k.T).todense())

    def params(self):
        return {"k": self.k, "similarity": self.similarity, "shrinkage": self.shrinkage}
