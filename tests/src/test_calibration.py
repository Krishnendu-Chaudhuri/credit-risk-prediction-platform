"""Tests for PD calibration."""

from __future__ import annotations

import numpy as np

from src.models.calibration import apply_calibrator, fit_calibrator


def test_isotonic_calibration_is_monotonic() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cal = fit_calibrator(y, prob, method="isotonic")
    calibrated = apply_calibrator(cal, prob)
    assert np.all(np.diff(calibrated) >= -1e-9)
