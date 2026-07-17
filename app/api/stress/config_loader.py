"""Load stress-testing JSON config with documented defaults."""

from __future__ import annotations

from typing import Any

from src.config.loader import load_json_config as _load_config
from src.config.paths import CONFIGS_DIR

from app.api.logging_config import get_logger

logger = get_logger(__name__)


def load_json_config(filename: str, default: dict[str, Any], config_label: str) -> dict[str, Any]:
    path = CONFIGS_DIR / filename
    if path.exists():
        return _load_config(filename, use_cache=False)
    logger.warning(
        "Using default %s because %s is missing; TODO(business-input): confirm with risk committee.",
        config_label,
        path,
    )
    return default
