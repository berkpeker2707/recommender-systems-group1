"""
constansts and hyperparameters for the experiments
"""

SEED = 42

RELEVANT_RATING_THRESHOLD = 4 # a held-out rating of 4 or 5 counts as "the user liked this movie"

SPLIT_FRACTIONS = (0.8, 0.1, 0.1)   # train / validation / test

MIN_USER_RATINGS_IN_TRAINING = 5 # users with fewer training ratings are not evaluated

SENSITIVITY_MIN_USER_RATINGS = [1, 20] # other filter values, to see if the filter changes the results

RECOMMENDATION_LIST_SIZE = 10 # the 10 in P@10, R@10, NDCG@10

# ---------------------------------------------------------------- Part 1 baselines
MOST_POPULAR_VARIANTS = {
    # name: (time window in days before the train cut, or None = all time; positives only?)
    "pop_alltime_count": (None, False), # all time, count all ratings
    "pop_alltime_positive": (None, True), # all time, count only positives (4 & 5's for example)
    "pop_recent90_count": (90, False), # last 90 days with postive and negative ratings
}
ITEM_KNN_NEIGHBOR_COUNTS = [10, 20, 50, 100, 200]
ITEM_KNN_SIMILARITY_MEASURES = ["cosine", "jaccard", "adjusted_cosine"]
