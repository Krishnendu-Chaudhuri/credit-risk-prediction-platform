"""Discounting for ECL cash flows."""

from __future__ import annotations

import os


def annual_discount_rate(loan_int_rate: float | None = None) -> float:
    # TODO(business-input): confirm discount curve for ECL reporting.
    if loan_int_rate is not None and loan_int_rate > 0:
        return loan_int_rate / 100.0
    return float(os.getenv("ECL_DISCOUNT_RATE", "0.08"))


def discount_factor(rate: float, periods: int = 12) -> float:
    return 1.0 / ((1.0 + rate) ** (periods / 12.0))


def discounted_amount(amount: float, rate: float, periods: int = 12) -> float:
    return amount * discount_factor(rate, periods)
