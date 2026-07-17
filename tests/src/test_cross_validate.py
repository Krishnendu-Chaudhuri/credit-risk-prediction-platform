"""Tests for cross-validation metrics helper."""

from __future__ import annotations

import numpy as np

from src.pipelines.validation.cross_validate import repeated_stratified_cv_metrics


def test_repeated_cv_summary() -> None:
    X = np.arange(60).reshape(-1, 1)
    y = np.array([0, 1] * 30)

    def fit_fn(train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
        return {"roc_auc": float(len(test_idx) / 100)}

    summary = repeated_stratified_cv_metrics(X, y, fit_fn, n_splits=5, n_repeats=2, random_states=[42, 43])
    assert "roc_auc" in summary
    assert "mean" in summary["roc_auc"]
    assert summary["roc_auc"]["mean"] > 0
