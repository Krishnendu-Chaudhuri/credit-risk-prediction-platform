"""Risk scorecards, EL, and IFRS 9 staging."""

from __future__ import annotations

import os

from src.models.ecl.calculator import (
    compute_ecl_full,
    ifrs9_stage,
)
from src.models.scorecard import pd_to_score

__all__ = ["ifrs9_stage", "pd_to_risk_band", "compute_expected_loss", "compute_ecl", "enrich_prediction"]

DEFAULT_LGD = float(os.getenv("DEFAULT_LGD", "0.45"))


def pd_to_risk_band(pd_value: float) -> str:
    if pd_value < 0.10:
        return "Low"
    if pd_value < 0.20:
        return "Medium"
    if pd_value < 0.35:
        return "High"
    return "Very High"


def compute_expected_loss(pd_value: float, ead: float, lgd: float = DEFAULT_LGD) -> float:
    return float(pd_value * lgd * ead)


def compute_ecl(
    pd_value: float,
    ead: float,
    stage: int,
    lgd: float = DEFAULT_LGD,
    loan_int_rate: float | None = None,
    loan_intent: str | None = None,
    loan_grade: str | None = None,
    stressed: bool = False,
) -> float:
    result = compute_ecl_full(
        pd_value,
        ead,
        stage=stage,
        lgd=lgd,
        loan_int_rate=loan_int_rate,
        loan_intent=loan_intent,
        loan_grade=loan_grade,
        stressed=stressed,
    )
    return result["ecl"]


def enrich_prediction(
    pd_value: float,
    loan_amnt: float,
    lgd: float = DEFAULT_LGD,
    loan_int_rate: float | None = None,
    loan_intent: str | None = None,
    loan_grade: str | None = None,
    threshold: float = 0.5,
    calibrated_pd: float | None = None,
) -> dict:
    pd_used = calibrated_pd if calibrated_pd is not None else pd_value
    ecl_result = compute_ecl_full(
        pd_used,
        loan_amnt,
        lgd=lgd,
        loan_int_rate=loan_int_rate,
        loan_intent=loan_intent,
        loan_grade=loan_grade,
    )
    return {
        "pd": float(pd_value),
        "calibrated_pd": float(pd_used),
        "risk_score": pd_to_score(pd_used),
        "risk_band": pd_to_risk_band(pd_used),
        "predicted_class": int(pd_used >= threshold),
        "expected_loss": ecl_result["expected_loss"],
        "ifrs9_stage": ecl_result["ifrs9_stage"],
        "ecl": ecl_result["ecl"],
    }
