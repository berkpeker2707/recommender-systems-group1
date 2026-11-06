"""
Mean and 95% confidence interval half-width (t-distribution) of a per-user metric.
"""
import numpy as np
from scipy import stats


# values: the ndcg of each user, ex: 1154 numbers for item-kNN on test
# example return: (0.219, 0.014) -> we write it as 0.219 +- 0.014
# meaning: with other users the average would most likely be between 0.205 and 0.233
def mean_with_confidence_interval(values, confidence: float = 0.95):
    x = np.asarray(values, dtype=float)
    # with 1 user there is no spread to measure
    if len(x) < 2:
        return float(x.mean()), float("nan")
    # t value for 95% (about 1.96 with 1154 users) * std of the users / sqrt(number of users)
    # ddof=1 is the sample std (divides by n - 1)
    # more users -> bigger sqrt -> smaller interval
    half = stats.t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)
