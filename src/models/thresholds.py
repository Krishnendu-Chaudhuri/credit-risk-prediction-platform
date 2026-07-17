"""Cost-based classification threshold selection."""

from __future__ import annotations

import os

import numpy as np

FP_COST_RATIO = float(os.getenv("FP_COST_RATIO", "0.05"))


def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, ead: np.ndarray, lgd: float) -> float:
    pred = (y_prob >= threshold).astype(int)
    fn_mask = (y_true == 1) & (pred == 0)
    fp_mask = (y_true == 0) & (pred == 1)
    fn_cost = (ead * lgd)[fn_mask].sum()
    fp_cost = (ead * FP_COST_RATIO)[fp_mask].sum()
    return float(fn_cost + fp_cost)


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ead: np.ndarray,
    lgd: float = 0.45,
    grid: int = 101,
) -> float:
    thresholds = np.linspace(0.01, 0.99, grid)
    costs = [expected_cost(y_true, y_prob, t, ead, lgd) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])
