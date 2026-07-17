"""Cross-validation utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def repeated_stratified_cv_metrics(
    X,
    y,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    n_splits: int = 5,
    n_repeats: int = 3,
    random_states: list[int] | None = None,
) -> dict[str, Any]:
    seeds = random_states or [42, 43, 44]
    all_metrics: list[dict[str, float]] = []
    for seed in seeds[:n_repeats]:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
        for train_idx, test_idx in rskf.split(X, y):
            all_metrics.append(fit_predict_fn(train_idx, test_idx))

    summary: dict[str, Any] = {}
    if not all_metrics:
        return summary
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary
