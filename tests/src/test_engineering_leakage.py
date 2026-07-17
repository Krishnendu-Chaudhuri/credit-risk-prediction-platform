"""Tests for split-before-engineer leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.features.engineering import build_feature_matrix, fit_imputation_stats


def _sample_raw(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "person_age": rng.integers(25, 60, n),
            "person_income": rng.integers(30000, 120000, n),
            "person_home_ownership": ["RENT"] * n,
            "person_emp_length": rng.uniform(1, 10, n),
            "loan_intent": ["PERSONAL"] * n,
            "loan_grade": ["B"] * n,
            "loan_amnt": rng.integers(5000, 20000, n),
            "loan_int_rate": rng.uniform(8, 15, n),
            "loan_status": rng.choice([0, 1], n),
            "loan_percent_income": rng.uniform(0.1, 0.4, n),
            "cb_person_default_on_file": ["N"] * n,
            "cb_person_cred_hist_length": rng.integers(2, 8, n),
        }
    )


def test_test_row_features_invariant_to_extra_train_rows() -> None:
    train = _sample_raw(40)
    test_row = _sample_raw(1)
    stats = fit_imputation_stats(train)
    baseline_x, _ = build_feature_matrix(test_row, stats)

    inflated_train = pd.concat([train, _sample_raw(200)], ignore_index=True)
    inflated_stats = fit_imputation_stats(inflated_train)
    assert inflated_stats["person_income_median"] != stats["person_income_median"]

    same_test_x, _ = build_feature_matrix(test_row, stats)
    pd.testing.assert_frame_equal(baseline_x, same_test_x)
