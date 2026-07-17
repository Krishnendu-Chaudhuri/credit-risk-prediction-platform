"""Project root and environment path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from src.config.paths import PROJECT_ROOT, resolve_path

__all__ = [
    "PROJECT_ROOT",
    "get_env_path",
    "get_project_root",
    "to_relative_path",
]


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_env_path(key: str, default: str | Path) -> Path:
    value = os.getenv(key)
    if value is None:
        return resolve_path(None, Path(default))
    return resolve_path(value, Path(default))


def to_relative_path(path: Path | str, root: Path | None = None) -> str:
    """Return path relative to project root when possible, else unchanged."""
    resolved_root = root or PROJECT_ROOT
    candidate = Path(path)
    try:
        if candidate.is_absolute():
            return candidate.relative_to(resolved_root.resolve()).as_posix()
        return candidate.as_posix()
    except ValueError:
        return str(path)
