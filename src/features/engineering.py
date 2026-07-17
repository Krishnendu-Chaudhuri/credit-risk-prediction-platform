"""Feature engineering and data cleaning for credit risk PD models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from app.api.logging_config import get_logger

logger = get_logger(__name__)

RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
    "age_bucket",
    "income_bucket",
]

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "log_person_income",
    "dti_ratio",
    "loan_grade_ord",
]

RAW_CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

GRADE_ORDINAL = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

INCOME_BUCKET_LABELS = ["low", "lower_mid", "upper_mid", "high"]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean outliers and invalid values in raw credit data."""
    data = df.copy()

    if "person_age" in data.columns:
        data["person_age"] = data["person_age"].clip(lower=18, upper=100)

    if "person_emp_length" in data.columns:
        data["person_emp_length"] = pd.to_numeric(data["person_emp_length"], errors="coerce")
        data["person_emp_length"] = data["person_emp_length"].clip(lower=0, upper=50)

    if "person_income" in data.columns:
        data["person_income"] = pd.to_numeric(data["person_income"], errors="coerce")
        data.loc[data["person_income"] <= 0, "person_income"] = np.nan

    if "loan_amnt" in data.columns:
        data["loan_amnt"] = pd.to_numeric(data["loan_amnt"], errors="coerce")
        data.loc[data["loan_amnt"] <= 0, "loan_amnt"] = np.nan

    return data


def _age_bucket(age: float) -> str:
    if age < 25:
        return "young"
    if age < 35:
        return "mid_young"
    if age < 50:
        return "mid"
    if age < 65:
        return "senior"
    return "elder"


def _compute_dti_series(data: pd.DataFrame) -> pd.Series:
    return (
        data["loan_percent_income"]
        .fillna(data["loan_amnt"] / data["person_income"].replace(0, np.nan))
        .clip(lower=0, upper=1)
    )


def fit_imputation_stats(raw_train: pd.DataFrame) -> dict[str, Any]:
    """Fit imputation statistics on training raw rows only."""
    cleaned = clean_raw_data(raw_train)
    dti = _compute_dti_series(cleaned)

    stats: dict[str, Any] = {
        "person_income_median": float(cleaned["person_income"].median()),
        "dti_ratio_median": float(dti.median()) if not dti.empty else 0.0,
        "income_quantiles": {
            "q25": float(cleaned["person_income"].quantile(0.25)),
            "q50": float(cleaned["person_income"].quantile(0.50)),
            "q75": float(cleaned["person_income"].quantile(0.75)),
        },
        "categorical_modes": {},
    }

    for col in RAW_CATEGORICAL_COLUMNS:
        if col in cleaned.columns and not cleaned[col].dropna().empty:
            stats["categorical_modes"][col] = str(cleaned[col].mode().iloc[0])

    return stats


def _income_bucket_from_quantiles(income: float, quantiles: dict[str, float]) -> str:
    if income <= quantiles["q25"]:
        return "low"
    if income <= quantiles["q50"]:
        return "lower_mid"
    if income <= quantiles["q75"]:
        return "upper_mid"
    return "high"


def engineer_features(
    df: pd.DataFrame,
    imputation_stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply feature engineering on cleaned data using frozen train statistics."""
    data = clean_raw_data(df)

    if imputation_stats and "person_income_median" in imputation_stats:
        income_median = imputation_stats["person_income_median"]
    else:
        income_median = data["person_income"].median()
        if imputation_stats is not None:
            logger.warning("imputation_stats missing person_income_median; using batch median")

    data["log_person_income"] = np.log1p(data["person_income"].fillna(income_median))

    dti = _compute_dti_series(data)
    if imputation_stats and "dti_ratio_median" in imputation_stats:
        dti_median = imputation_stats["dti_ratio_median"]
    else:
        dti_median = float(dti.median()) if not dti.empty else 0.0
        if imputation_stats is not None:
            logger.warning("imputation_stats missing dti_ratio_median; using batch median")
    data["dti_ratio"] = dti.fillna(dti_median)

    data["age_bucket"] = data["person_age"].apply(_age_bucket)

    quantiles = (imputation_stats or {}).get("income_quantiles")
    if quantiles:
        data["income_bucket"] = data["person_income"].apply(
            lambda x: _income_bucket_from_quantiles(float(x), quantiles) if pd.notna(x) else "lower_mid"
        )
    else:
        logger.warning("income_quantiles missing; falling back to q50-only bucket assignment")
        q50 = income_median
        data["income_bucket"] = data["person_income"].apply(
            lambda x: "low" if pd.notna(x) and x <= q50 else "upper_mid"
        )

    data["loan_grade_ord"] = data["loan_grade"].map(GRADE_ORDINAL).fillna(4)

    modes = (imputation_stats or {}).get("categorical_modes", {})
    for col in RAW_CATEGORICAL_COLUMNS:
        if col in data.columns:
            fill_val = modes.get(col, "unknown")
            data[col] = data[col].astype(str).replace("nan", fill_val).fillna(fill_val)

    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            data[col] = data[col].astype(str).replace("nan", "unknown")

    return data


def build_feature_matrix(
    raw_df: pd.DataFrame,
    imputation_stats: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Transform raw rows into feature matrix X and target y."""
    engineered = engineer_features(raw_df, imputation_stats=imputation_stats)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = engineered[feature_cols].copy()
    y = engineered["loan_status"].astype(int)
    return X, y


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Legacy helper: engineer full frame (prefer split-then-transform in train.py)."""
    stats = fit_imputation_stats(df)
    X, y = build_feature_matrix(df, stats)
    return X, y, NUMERIC_FEATURES + CATEGORICAL_FEATURES


def loan_input_to_frame(loan: dict) -> pd.DataFrame:
    """Convert a single loan dict to a one-row DataFrame."""
    return pd.DataFrame([loan])
