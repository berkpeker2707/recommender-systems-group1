"""
constansts and hyperparameters for the experiments
"""

SEED = 42

POSITIVE_THRESHOLD = 4

SPLIT_FRACTIONS = (0.8, 0.1, 0.1)   # train / validation / test

MIN_TRAIN_INTERACTIONS = 5

SENSITIVITY_MIN_TRAIN = [1, 20]

TOP_K = 10 # P@10, R@10, NDCG@10 (fixed by the brief)

# ---------------------------------------------------------------- Part 1 baselines
POPULAR_VARIANTS = {
    # name: (time window in days before the train cut, or None = all time; positives only?)
    "pop_alltime_count": (None, False), # all time, count all ratings
    "pop_alltime_positive": (None, True), # all time, count only positives (4 & 5's for example)
    "pop_recent90_count": (90, False), # last 90 days with postive and negative ratings
}
ITEM_KNN_K_GRID = [10, 20, 50, 100, 200]
ITEM_KNN_SIM_GRID = ["cosine", "jaccard", "adjusted_cosine"]
