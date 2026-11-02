"""
Mean and 95% confidence interval half-width (t-distribution) of a per-user metric.
"""
import numpy as np
from scipy import stats


def mean_with_confidence_interval(values, confidence: float = 0.95):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)
