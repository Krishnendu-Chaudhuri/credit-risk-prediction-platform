"""Per-prediction SHAP explanations."""

from __future__ import annotations

from typing import Any

import numpy as np


def explain_lr(model, X_row: np.ndarray, feature_names: list[str], top_n: int = 5) -> list[dict[str, Any]]:
    coefs = model.coef_[0]
    contributions = coefs * X_row.flatten()
    pairs = sorted(zip(feature_names, contributions, strict=False), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": n, "contribution": float(v)} for n, v in pairs[:top_n]]


def explain_xgb(model, X_row: np.ndarray, feature_names: list[str], top_n: int = 5) -> list[dict[str, Any]]:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row.reshape(1, -1))
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]
    except Exception:
        importances = model.feature_importances_[: len(feature_names)]
        values = importances * X_row.flatten()

    pairs = sorted(zip(feature_names, values, strict=False), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": n, "contribution": float(v)} for n, v in pairs[:top_n]]


def explain_prediction(
    model,
    model_name: str,
    X_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if model_name == "lr":
        return explain_lr(model, X_row, feature_names, top_n=top_n)
    return explain_xgb(model, X_row, feature_names, top_n=top_n)
