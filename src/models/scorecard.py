"""PDO (points-to-double-odds) scorecard mapping."""

from __future__ import annotations

import os

import numpy as np

PDO = int(os.getenv("SCORECARD_PDO", "20"))
BASE_SCORE = int(os.getenv("SCORECARD_BASE_SCORE", "600"))
BASE_ODDS = float(os.getenv("SCORECARD_BASE_ODDS", "50"))  # 50:1 good:bad
MIN_SCORE = int(os.getenv("SCORECARD_MIN", "300"))
MAX_SCORE = int(os.getenv("SCORECARD_MAX", "850"))


def pd_to_score(pd_value: float) -> int:
    """Map PD to score using PDO / log-odds scaling."""
    clipped = float(np.clip(pd_value, 1e-6, 0.9999))
    odds = (1 - clipped) / clipped
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    score = offset + factor * np.log(odds)
    return int(round(np.clip(score, MIN_SCORE, MAX_SCORE)))
