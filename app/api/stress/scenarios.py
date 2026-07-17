"""Scenario PD adjustment logic."""

from __future__ import annotations

import numpy as np

from app.api.stress.config_loader import load_json_config
from src.models.schemas import StressScenario

DEFAULT_SEGMENT_MULTIPLIERS = {"default": 1.0}

_SEGMENT_MULTIPLIERS: dict[str, float] | None = None


def _load_segment_multipliers() -> dict[str, float]:
    global _SEGMENT_MULTIPLIERS  # noqa: PLW0603
    if _SEGMENT_MULTIPLIERS is not None:
        return _SEGMENT_MULTIPLIERS
    _SEGMENT_MULTIPLIERS = load_json_config(
        "segment_multipliers.json",
        DEFAULT_SEGMENT_MULTIPLIERS,
        "segment PD multipliers",
    )
    return _SEGMENT_MULTIPLIERS


def reset_segment_multipliers_cache() -> None:
    """Clear cached segment multipliers (for tests)."""
    global _SEGMENT_MULTIPLIERS  # noqa: PLW0603
    _SEGMENT_MULTIPLIERS = None


def apply_pd_shock(
    base_pd: float | np.ndarray,
    scenario: StressScenario,
    loan_intents: list[str] | None = None,
) -> float | np.ndarray:
    base = np.asarray(base_pd, dtype=float)
    portfolio_mult = scenario.portfolio_pd_multiplier
    if loan_intents is None:
        shocked = base * portfolio_mult
        return np.clip(shocked, 0, 0.9999)

    seg_table = _load_segment_multipliers()
    mults = np.array([seg_table.get(intent, seg_table.get("default", 1.0)) for intent in loan_intents])
    shocked = base * portfolio_mult * mults
    return np.clip(shocked, 0, 0.9999)
