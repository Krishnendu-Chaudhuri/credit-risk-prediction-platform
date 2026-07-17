"""Tests for stress config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.api.stress.config_loader import load_json_config
from app.api.stress.portfolio import DEFAULT_SCENARIO_WEIGHTS, _load_scenario_weights
from app.api.stress.scenarios import (
    DEFAULT_SEGMENT_MULTIPLIERS,
    _load_segment_multipliers,
    reset_segment_multipliers_cache,
)


def test_load_scenario_weights_from_file() -> None:
    weights = _load_scenario_weights()
    assert weights["Normal"] == 0.6
    assert weights["Recession"] == 0.25


def test_load_json_config_uses_default_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setattr("app.api.stress.config_loader.CONFIGS_DIR", config_dir)
    loaded = load_json_config("missing.json", DEFAULT_SCENARIO_WEIGHTS, "scenario probability weights")
    assert loaded == DEFAULT_SCENARIO_WEIGHTS


def test_load_segment_multipliers_from_file() -> None:
    reset_segment_multipliers_cache()
    multipliers = _load_segment_multipliers()
    assert multipliers.get("PERSONAL") == 1.0
    assert multipliers.get("default") == 1.0


def test_segment_multipliers_default_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_segment_multipliers_cache()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setattr("app.api.stress.config_loader.CONFIGS_DIR", config_dir)
    reset_segment_multipliers_cache()
    multipliers = _load_segment_multipliers()
    assert multipliers == DEFAULT_SEGMENT_MULTIPLIERS
