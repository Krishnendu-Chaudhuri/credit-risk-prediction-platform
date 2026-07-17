"""Lifetime PD term structure from 12-month PD."""

from __future__ import annotations

import os

DEFAULT_LOAN_TENOR_MONTHS = int(os.getenv("DEFAULT_LOAN_TENOR_MONTHS", "36"))


def lifetime_pd_from_annual(pd_12m: float, tenor_months: int | None = None) -> float:
    """Convert 12-month PD to lifetime PD via constant hazard approximation."""
    months = tenor_months or DEFAULT_LOAN_TENOR_MONTHS
    pd_12m = min(max(pd_12m, 1e-6), 0.9999)
    monthly_hazard = 1 - (1 - pd_12m) ** (1 / 12)
    lifetime = 1 - (1 - monthly_hazard) ** months
    return min(float(lifetime), 0.9999)


def horizon_pd(pd_12m: float, stage: int, tenor_months: int | None = None) -> float:
    if stage == 1:
        return min(pd_12m, 0.9999)
    return lifetime_pd_from_annual(pd_12m, tenor_months=tenor_months)
