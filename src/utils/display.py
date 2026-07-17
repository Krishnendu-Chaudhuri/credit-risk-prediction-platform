"""Display formatting from shared contract."""

from __future__ import annotations

from typing import Any

from src.config.loader import load_json_config


def load_display_contract() -> dict[str, Any]:
    return load_json_config("display_contract.json")


def band_style(risk_band: str) -> dict[str, str]:
    contract = load_display_contract()
    return contract["risk_bands"].get(risk_band, {"color": "#666666", "label": risk_band})


def format_pd(pd_value: float) -> str:
    return f"{pd_value:.2%}"


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"
