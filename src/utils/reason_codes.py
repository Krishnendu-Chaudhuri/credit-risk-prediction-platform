"""Map feature contributions to adverse-action reason codes."""

from __future__ import annotations

from typing import Any

from src.config.loader import load_json_config

_REASON_CODES: list[dict[str, str]] | None = None

FEATURE_TO_CODE = {
    "person_income": "INSUFFICIENT_INCOME",
    "log_person_income": "INSUFFICIENT_INCOME",
    "loan_amnt": "EXCESSIVE_OBLIGATIONS",
    "loan_int_rate": "HIGH_INTEREST_RATE",
    "loan_percent_income": "HIGH_DEBT_TO_INCOME",
    "dti_ratio": "HIGH_DEBT_TO_INCOME",
    "person_emp_length": "LIMITED_EMPLOYMENT_HISTORY",
    "cb_person_cred_hist_length": "LIMITED_CREDIT_HISTORY",
    "loan_grade": "POOR_CREDIT_RATING",
    "cb_person_default_on_file": "PRIOR_DEFAULT",
}


def _load_reason_codes() -> list[dict[str, str]]:
    global _REASON_CODES  # noqa: PLW0603
    if _REASON_CODES is not None:
        return _REASON_CODES
    try:
        data: list[dict[str, str]] = load_json_config("reason_codes.json")  # type: ignore[assignment]
        _REASON_CODES = data
    except FileNotFoundError:
        _REASON_CODES = [
            {"code": code, "description": code.replace("_", " ").title()} for code in FEATURE_TO_CODE.values()
        ]
    return _REASON_CODES


def map_reason_codes(contributions: list[dict[str, Any]], limit: int = 4) -> list[str]:
    codes: list[str] = []
    for item in sorted(contributions, key=lambda x: x["contribution"]):
        feature = item["feature"].split("__")[-1].replace("num__", "").replace("cat__", "")
        for key, code in FEATURE_TO_CODE.items():
            if key in feature and code not in codes:
                codes.append(code)
                break
        if len(codes) >= limit:
            break
    return codes or ["GENERAL_CREDIT_RISK"]
