"""Threshold-based classification helpers shared by inference and tests."""

import math


def churn_class_from_probability(probability: float, threshold: float) -> int:
    """Return churn class 1 when probability meets the configured threshold."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    return int(probability >= threshold)