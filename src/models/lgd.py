"""Segment LGD lookup."""

from __future__ import annotations

import json
import os

from src.utils import PROJECT_ROOT

DEFAULT_LGD = float(os.getenv("DEFAULT_LGD", "0.45"))
LGD_BUFFER_PCT = float(os.getenv("LGD_BUFFER_PCT", "0.05"))
DOWNTURN_FACTOR = float(os.getenv("LGD_DOWNTURN_FACTOR", "1.15"))

_LGD_TABLE: dict[str, float] | None = None


def _load_lgd_table() -> dict[str, float]:
    global _LGD_TABLE  # noqa: PLW0603
    if _LGD_TABLE is not None:
        return _LGD_TABLE
    path = PROJECT_ROOT / "configs" / "lgd_lookup.json"
    if path.exists():
        _LGD_TABLE = json.loads(path.read_text(encoding="utf-8"))
    else:
        # TODO(business-input): replace with collateral/seniority-specific LGD table.
        _LGD_TABLE = {
            "PERSONAL": 0.50,
            "EDUCATION": 0.55,
            "MEDICAL": 0.48,
            "VENTURE": 0.60,
            "HOMEIMPROVEMENT": 0.40,
            "DEBTCONSOLIDATION": 0.52,
            "default": DEFAULT_LGD,
        }
    return _LGD_TABLE


def resolve_lgd(
    loan_intent: str | None,
    loan_grade: str | None = None,
    stressed: bool = False,
    base_override: float | None = None,
) -> float:
    if base_override is not None:
        base = base_override
    else:
        table = _load_lgd_table()
        base = table.get(loan_intent or "", table.get("default", DEFAULT_LGD))
        if loan_grade in {"E", "F", "G"}:
            base = min(base + 0.05, 0.99)
    buffered = base * (1 + LGD_BUFFER_PCT)
    if stressed:
        buffered *= DOWNTURN_FACTOR
    return min(float(buffered), 0.99)
