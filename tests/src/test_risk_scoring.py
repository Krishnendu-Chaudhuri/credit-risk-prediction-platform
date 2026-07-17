"""Unit tests for src.models.risk_scoring."""

from __future__ import annotations

import pytest

from src.models.risk_scoring import (
    compute_ecl,
    compute_expected_loss,
    enrich_prediction,
    ifrs9_stage,
    pd_to_risk_band,
    pd_to_score,
)


@pytest.mark.parametrize(
    ("pd_value", "expected_stage"),
    [
        (0.01, 1),
        (0.04, 1),
        (0.05, 2),
        (0.19, 2),
        (0.20, 3),
        (0.50, 3),
    ],
)
def test_ifrs9_stage_thresholds(pd_value: float, expected_stage: int) -> None:
    assert ifrs9_stage(pd_value) == expected_stage


@pytest.mark.parametrize(
    ("pd_value", "expected_band"),
    [
        (0.05, "Low"),
        (0.15, "Medium"),
        (0.25, "High"),
        (0.40, "Very High"),
    ],
)
def test_pd_to_risk_band(pd_value: float, expected_band: str) -> None:
    assert pd_to_risk_band(pd_value) == expected_band


def test_pd_to_score_inverse_relationship() -> None:
    low_pd_score = pd_to_score(0.01)
    high_pd_score = pd_to_score(0.50)
    assert low_pd_score > high_pd_score
    assert 300 <= high_pd_score <= 850


def test_compute_expected_loss() -> None:
    assert compute_expected_loss(0.10, 10000, lgd=0.45) == pytest.approx(450.0)


def test_compute_ecl_stage_horizon() -> None:
    stage1 = compute_ecl(0.10, 10000, stage=1, lgd=0.45)
    stage2 = compute_ecl(0.10, 10000, stage=2, lgd=0.45)
    assert stage2 > stage1


def test_enrich_prediction_keys() -> None:
    result = enrich_prediction(0.12, 10000, lgd=0.45)
    assert result["pd"] == pytest.approx(0.12)
    assert result["ifrs9_stage"] == 2
    assert "risk_score" in result
    assert "ecl" in result
