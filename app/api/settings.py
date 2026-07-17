"""Centralized environment configuration — loads .env once at import."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.config.paths import PROJECT_ROOT
_ENV_LOADED = False
_DEV_VALUES = frozenset({"development", "dev", "local"})


def load_env() -> None:
    """Load .env from the project root (idempotent)."""
    global _ENV_LOADED  # noqa: PLW0603
    if _ENV_LOADED:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        example_path = PROJECT_ROOT / ".env.example"
        if example_path.exists():
            load_dotenv(example_path, override=False)
    _ENV_LOADED = True


def get_environment() -> str:
    """Return ENVIRONMENT or ENV; default to development when unset."""
    load_env()
    raw = os.getenv("ENVIRONMENT") or os.getenv("ENV") or os.getenv("APP_ENV")
    if raw is None or not str(raw).strip():
        return "development"
    return str(raw).strip()


def get_api_key() -> str | None:
    load_env()
    value = os.getenv("API_KEY")
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def set_api_key(key: str) -> None:
    """Persist API key to process environment (used for dev auto-generation)."""
    os.environ["API_KEY"] = key


def is_dev_mode() -> bool:
    load_env()
    return get_environment().strip().lower() in _DEV_VALUES


# Load .env before any other API module reads os.environ.
load_env()
