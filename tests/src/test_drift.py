"""Tests for PSI drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.pipelines.monitoring.drift import compute_psi_report


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    shifted = pd.DataFrame({"x": rng.normal(2, 1, 500)})
    report = compute_psi_report(reference, shifted, features=["x"])
    assert report["x"]["psi"] > 0.25
    assert report["x"]["alert"] is True
