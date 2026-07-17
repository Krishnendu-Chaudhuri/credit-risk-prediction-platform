"""Satellite regression for macro-driven PD multipliers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass
class MacroRegressionResult:
    coefficients: dict[str, float]
    intercept: float
    r2: float
    spec: str


def fit_macro_multiplier_regression(
    defaults: pd.Series,
    macro_df: pd.DataFrame,
) -> MacroRegressionResult:
    """Regress default indicator ~ macro variables to derive scenario sensitivity."""
    X = macro_df.copy()
    y = defaults.astype(float)
    model = LinearRegression()
    model.fit(X, y)
    r2 = float(r2_score(y, model.predict(X)))
    coefs = {col: float(val) for col, val in zip(X.columns, model.coef_, strict=False)}
    return MacroRegressionResult(
        coefficients=coefs,
        intercept=float(model.intercept_),
        r2=r2,
        spec="default_rate ~ " + " + ".join(X.columns),
    )


def predict_multiplier(
    result: MacroRegressionResult, macro_values: dict[str, float], base_values: dict[str, float]
) -> float:
    delta = sum(result.coefficients.get(k, 0) * (macro_values[k] - base_values[k]) for k in macro_values)
    predicted = result.intercept + delta
    return float(np.clip(1 + predicted, 0.5, 2.0))
