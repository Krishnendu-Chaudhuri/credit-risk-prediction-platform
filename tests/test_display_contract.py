"""Tests for shared display contract."""

from __future__ import annotations

from src.utils.display import band_style, format_currency, format_pd, load_display_contract


def test_display_contract_loads() -> None:
    contract = load_display_contract()
    assert "Low" in contract["risk_bands"]
    assert contract["format"]["currency"] == "USD"


def test_band_style_and_formatters() -> None:
    style = band_style("Low")
    assert style["color"].startswith("#")
    assert format_pd(0.1234) == "12.34%"
    assert format_currency(1000).startswith("$")
