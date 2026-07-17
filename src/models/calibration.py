"""PD calibration and model-risk buffers."""

from __future__ import annotations

import os
from typing import Literal

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibratorType = Literal["platt", "isotonic"]


def fit_calibrator(
    y_cal: np.ndarray,
    prob_cal: np.ndarray,
    method: CalibratorType = "platt",
):
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(prob_cal, y_cal)
        return cal

    lr = LogisticRegression(max_iter=1000)
    lr.fit(prob_cal.reshape(-1, 1), y_cal)
    return lr


def apply_calibrator(calibrator, prob: np.ndarray) -> np.ndarray:
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    return calibrator.predict(prob)


def save_calibrator(path, calibrator) -> None:
    joblib.dump(calibrator, path)


def load_calibrator(path):
    return joblib.load(path)


def apply_pd_buffer(pd_value: float) -> float:
    bps = float(os.getenv("PD_BUFFER_BPS", "25"))
    return min(float(pd_value) + bps / 10_000.0, 0.9999)
