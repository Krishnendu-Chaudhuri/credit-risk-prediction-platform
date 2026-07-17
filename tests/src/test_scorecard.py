"""Tests for PDO scorecard."""

from __future__ import annotations

from src.models.scorecard import pd_to_score


def test_pdo_lower_pd_higher_score() -> None:
    assert pd_to_score(0.01) > pd_to_score(0.30)


def test_pdo_score_in_valid_range() -> None:
    score = pd_to_score(0.15)
    assert 300 <= score <= 850
