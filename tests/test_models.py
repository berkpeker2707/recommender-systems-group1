"""
Checks the three self-implemented Part 2 models on tiny examples that can be worked out by hand.
run with: python -m pytest tests -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.bpr import BPR, _add_to_rows
from src.models.content_based import ContentBased
from src.models.user_knn import UserKNN


def _ratings(rows):
    """(user_index, movie_index, rating) tuples -> the ratings table the models expect."""
    table = pd.DataFrame(rows, columns=["user_index", "movie_index", "rating"])
    table["unix_time"] = 0
    return table


def test_content_profile_worked_example():
    # 5 movies with 3 genres: Action, Drama, Comedy
    genres = np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    no_decades = np.zeros((5, 1), dtype=np.float32)
    # Bob (user 0) rated i2=1, i3=2, i4=3, i5=3 -> mean 2.25
    bob = _ratings([(0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 3)])
    model = ContentBased(genres, no_decades, year_weight=0.0, profile_weighting="centered").fit(
        bob, 1, 5
    )
    # expected profile: Action +1.5, Drama +0.5, Comedy -1.5 (last column = decade, weight 0)
    assert model.user_profiles[0, :3] == pytest.approx([1.5, 0.5, -1.5])
    # i1 (Action + Drama) fits Bob's profile better than i2 (Comedy)
    scores = model.score(np.array([0]))[0]
    assert scores[0] > scores[1]


def test_user_knn_worked_example():
    # user 0 rated movies 0, 1; user 1 rated 0, 1, 2; user 2 rated only movie 3
    ratings = _ratings([(0, 0, 5), (0, 1, 4), (1, 0, 5), (1, 1, 4), (1, 2, 5), (2, 3, 5)])
    model = UserKNN(n_neighbors=1, similarity_measure="cosine").fit(ratings, 3, 4)
    scores = model.score(np.array([0]))[0]
    # cosine(user 0, user 1) = 2 shared movies / sqrt(2 * 3) = 0.816; user 2 shares nothing
    # the only neighbour is user 1, so movie 2 gets 0.816 and movie 3 (only user 2 saw it) gets 0
    assert scores[2] == pytest.approx(2 / np.sqrt(6), abs=1e-6)
    assert scores[3] == 0.0


def test_add_to_rows_same_as_numpy_add_at():
    rng = np.random.default_rng(0)
    rows = np.array([4, 7, 4, 0, 7, 7])
    changes = rng.normal(size=(6, 3)).astype(np.float32)
    expected = np.zeros((8, 3), dtype=np.float32)
    np.add.at(expected, rows, changes)
    result = np.zeros((8, 3), dtype=np.float32)
    _add_to_rows(result, rows, changes)
    assert result == pytest.approx(expected, abs=1e-6)


def test_bpr_learns_two_taste_groups():
    # users 0-9 rate movies 0-4, users 10-19 rate movies 5-9; user 0 has not rated movie 4 yet
    rows = [(u, m, 5) for u in range(10) for m in range(5)] + [
        (u, m, 5) for u in range(10, 20) for m in range(5, 10)
    ]
    rows.remove((0, 4, 5))
    model = BPR(
        n_factors=4, regularization=0.001, learning_rate=0.1, n_epochs=200, batch_size=16, seed=0
    )
    scores = model.fit(_ratings(rows), 20, 10).score(np.array([0]))[0]
    # movie 4 belongs to user 0's group, so it must beat every movie of the other group
    assert scores[4] > scores[5:].max()
