"""Unified JSON config loader with caching."""

from __future__ import annotations

import json
from typing import Any

from src.config.paths import CONFIGS_DIR

_CACHE: dict[str, dict[str, Any]] = {}


def load_json_config(
    filename: str,
    default: dict[str, Any] | None = None,
    *,
    config_label: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load a JSON config from configs/; return default when missing."""
    if use_cache and filename in _CACHE:
        return _CACHE[filename]

    path = CONFIGS_DIR / filename
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    elif default is not None:
        data = default
    else:
        raise FileNotFoundError(f"Config not found: {path}")

    if use_cache:
        _CACHE[filename] = data
    return data
