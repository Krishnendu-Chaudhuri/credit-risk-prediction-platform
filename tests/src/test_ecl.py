"""Tests for ECL methodology."""

from __future__ import annotations

from src.models.ecl.calculator import compute_ecl_full
from src.models.ecl.discounting import discounted_amount
from src.models.ecl.term_structure import lifetime_pd_from_annual


def test_lifetime_pd_exceeds_annual_for_long_tenor() -> None:
    annual = 0.10
    lifetime = lifetime_pd_from_annual(annual, tenor_months=36)
    assert lifetime > annual


def test_discounted_ecl_less_than_undiscounted() -> None:
    result = compute_ecl_full(0.12, 10000, stage=2, loan_int_rate=10.0)
    undiscounted = result["expected_loss"]
    assert result["ecl"] <= undiscounted


def test_discount_factor_reduces_amount() -> None:
    assert discounted_amount(1000.0, 0.08) < 1000.0
