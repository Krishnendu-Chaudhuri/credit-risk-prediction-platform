"""Tests for cost-based threshold."""

from __future__ import annotations

import numpy as np

from src.models.thresholds import optimal_threshold


def test_optimal_threshold_not_always_half() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    ead = np.array([10000, 10000, 10000, 10000, 10000, 10000])
    t = optimal_threshold(y, prob, ead, lgd=0.45)
    assert 0.01 <= t <= 0.99
