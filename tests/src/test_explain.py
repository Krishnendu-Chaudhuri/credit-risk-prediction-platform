"""Tests for SHAP / LR explanation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils.explain import explain_lr, explain_prediction


def test_explain_lr_top_contributions() -> None:
    X = np.array([[1.0, -2.0, 0.5]])
    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[0.5, -1.0, 0.2]])
    model.intercept_ = np.array([0.0])
    names = ["a", "b", "c"]
    result = explain_lr(model, X, names, top_n=2)
    assert len(result) == 2
    assert result[0]["feature"] in names
    assert isinstance(result[0]["contribution"], float)


def test_explain_prediction_lr_path() -> None:
    X = np.array([[1.0, 0.0]])
    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[1.0, -1.0]])
    model.intercept_ = np.array([0.0])
    out = explain_prediction(model, "lr", X, ["x", "y"], top_n=1)
    assert out[0]["feature"] == "x"
