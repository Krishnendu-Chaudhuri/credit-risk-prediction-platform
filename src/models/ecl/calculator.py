"""Orchestrate ECL with term structure, discounting, and buffers."""

from __future__ import annotations

import os

from src.models.ecl.discounting import annual_discount_rate, discounted_amount
from src.models.ecl.term_structure import horizon_pd
from src.models.lgd import resolve_lgd


def ifrs9_stage(pd_value: float) -> int:
    if pd_value >= 0.20:
        return 3
    if pd_value >= 0.05:
        return 2
    return 1


def apply_pd_buffer(pd_value: float) -> float:
    bps = float(os.getenv("PD_BUFFER_BPS", "25"))
    return min(pd_value + bps / 10_000.0, 0.9999)


def compute_ecl_full(
    pd_value: float,
    ead: float,
    stage: int | None = None,
    lgd: float | None = None,
    loan_int_rate: float | None = None,
    loan_intent: str | None = None,
    loan_grade: str | None = None,
    stressed: bool = False,
    tenor_months: int | None = None,
) -> dict[str, float]:
    stage_val = stage if stage is not None else ifrs9_stage(pd_value)
    pd_buffered = apply_pd_buffer(pd_value)
    horizon = horizon_pd(pd_buffered, stage_val, tenor_months=tenor_months)
    lgd_val = lgd if lgd is not None else resolve_lgd(loan_intent, loan_grade, stressed=stressed)
    el = float(horizon * lgd_val * ead)
    rate = annual_discount_rate(loan_int_rate)
    ecl = discounted_amount(el, rate)
    return {
        "pd": float(pd_value),
        "pd_buffered": pd_buffered,
        "horizon_pd": horizon,
        "lgd": lgd_val,
        "expected_loss": el,
        "ecl": ecl,
        "ifrs9_stage": stage_val,
    }
