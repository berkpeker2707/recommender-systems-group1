"""
config.py — the ONLY place for tunable constants.

Rule: the examiner must never need to edit source code. Paths come from the
command line (run_all.py --data-dir); everything else that affects a number in
the report lives here, so it can be quoted in the report and README.
"""

# ---------------------------------------------------------------- randomness
# SEED is used for every random step (there are few: tie-breaking).
SEED = 42

# ---------------------------------------------------------------- feedback
# Decision (Part 1, to justify in report): treat the task as top-N ranking with
# implicit feedback derived from ratings.
#   - TRAIN: every rating is an interaction (the user chose to watch + rate it).
#   - RELEVANT in val/test: rating >= POSITIVE_THRESHOLD.
# Alternative to discuss: use only positives in training too.
POSITIVE_THRESHOLD = 4

# ---------------------------------------------------------------- split
# Global temporal split (Lecture 3.1 slide 4). Cut points are chosen as
# QUANTILES OF INTERACTIONS over time (not calendar dates), because ML-1M
# timestamps are very bursty (most ratings entered in 2000) — a date-based cut
# would give an almost empty test set. Exact cut timestamps are logged.
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)   # train / validation / test

# ---------------------------------------------------------------- filtering
# Evaluate only users with >= MIN_TRAIN_INTERACTIONS in train (cold users cannot
# be served by CF) and >= 1 relevant item in the evaluated split.
# Report how many users/rows each rule removes.
MIN_TRAIN_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 0           # we do NOT drop items: keeps the full catalogue
# Sensitivity check (Part 1, excellent-level): rerun the final baselines with these
# alternative thresholds and compare the change to the gap between baselines.
SENSITIVITY_MIN_TRAIN = [1, 20]

# ---------------------------------------------------------------- evaluation
TOP_K = 10                          # P@10, R@10, NDCG@10 (fixed by the brief)
EXCLUDE_SEEN = True                 # never recommend items already in the user's train history

# ---------------------------------------------------------------- Part 1 baselines
POPULAR_VARIANTS = {
    # name: (time window in days before the train cut, or None = all time; positives only?)
    "pop_alltime_count": (None, False),
    "pop_alltime_positive": (None, True),
    "pop_recent90_count": (90, False),
}
ITEM_KNN_K_GRID = [10, 20, 50, 100, 200]
ITEM_KNN_SIM_GRID = ["cosine", "jaccard", "adjusted_cosine"]
SHRINKAGE = 0                       # significance weighting (Lecture 1.2 slide 10); tune later?
