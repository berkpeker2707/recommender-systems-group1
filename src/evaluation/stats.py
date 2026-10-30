"""
stats.py — variability (Lecture 3.1 slide 10).

- mean_ci(values) -> mean, 95% CI half-width (t-distribution), computed over users.
Rule from the brief: "a difference reported without an interval will not be accepted".
"""
import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)
