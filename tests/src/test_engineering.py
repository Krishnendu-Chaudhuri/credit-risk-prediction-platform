"""Unit tests for src.features.engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import clean_raw_data, engineer_features


def test_clean_raw_data_clips_age() -> None:
    df = pd.DataFrame({"person_age": [10, 150], "person_income": [50000, 60000]})
    cleaned = clean_raw_data(df)
    assert cleaned["person_age"].min() >= 18
    assert cleaned["person_age"].max() <= 100


def test_engineer_features_uses_training_median_for_missing_income() -> None:
    df = pd.DataFrame(
        {
            "person_age": [30, 40],
            "person_income": [50000, None],
            "person_home_ownership": ["RENT", "OWN"],
            "person_emp_length": [3, 5],
            "loan_intent": ["PERSONAL", "PERSONAL"],
            "loan_grade": ["B", "C"],
            "loan_amnt": [10000, 12000],
            "loan_int_rate": [10.0, 11.0],
            "loan_percent_income": [0.2, 0.25],
            "cb_person_default_on_file": ["N", "N"],
            "cb_person_cred_hist_length": [3, 4],
        }
    )
    training_median = 65000.0
    stats = {
        "person_income_median": training_median,
        "dti_ratio_median": 0.25,
        "income_quantiles": {"q25": 40000.0, "q50": training_median, "q75": 90000.0},
    }
    result = engineer_features(df, imputation_stats=stats)
    missing_row_log = result.loc[1, "log_person_income"]
    expected = np.log1p(training_median)
    assert missing_row_log == pytest.approx(expected)


def test_engineer_features_batch_independent_with_imputation_stats() -> None:
    base_row = {
        "person_age": 35,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 12.0,
        "loan_percent_income": 0.3,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    }
    stats = {
        "person_income_median": 55000.0,
        "dti_ratio_median": 0.3,
        "income_quantiles": {"q25": 40000.0, "q50": 55000.0, "q75": 80000.0},
        "categorical_modes": {"loan_intent": "PERSONAL"},
    }
    alone = engineer_features(pd.DataFrame([base_row]), imputation_stats=stats)
    in_batch = engineer_features(
        pd.DataFrame([base_row, {**base_row, "person_income": 120000}]),
        imputation_stats=stats,
    )
    assert alone.iloc[0]["log_person_income"] == pytest.approx(in_batch.iloc[0]["log_person_income"])
