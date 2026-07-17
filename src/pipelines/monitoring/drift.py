"""Feature drift monitoring via PSI."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

PSI_ALERT = float(os.getenv("PSI_ALERT", "0.25"))


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    expected_perc = np.histogram(expected, bins=breakpoints)[0] / max(len(expected), 1)
    actual_perc = np.histogram(actual, bins=breakpoints)[0] / max(len(actual), 1)
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def compute_psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    cols = features or [
        c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]
    report: dict[str, dict[str, Any]] = {}
    for col in cols:
        psi = _psi(reference[col].astype(float).values, current[col].astype(float).values)
        report[col] = {"psi": psi, "alert": psi >= PSI_ALERT}
    return report
