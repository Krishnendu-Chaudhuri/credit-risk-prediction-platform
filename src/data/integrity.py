"""Data file checksum manifest for training/serving consistency."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.api.logging_config import get_logger

from src.utils import load_json, save_json

logger = get_logger(__name__)


def file_sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def compute_manifest(paths: list[Path | str]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for raw in paths:
        path = Path(raw)
        if path.exists():
            manifest[path.as_posix()] = file_sha256(path)
    return manifest


def verify_manifest(
    manifest: dict[str, Any],
    paths: list[Path | str] | None = None,
) -> list[str]:
    """Return list of drift warnings (empty if all match)."""
    warnings: list[str] = []
    check_paths = [Path(p) for p in (paths or manifest.keys())]
    for path in check_paths:
        key = path.as_posix()
        if key not in manifest:
            warnings.append(f"No manifest entry for {key}")
            continue
        if not path.exists():
            warnings.append(f"Missing data file: {key}")
            continue
        current = file_sha256(path)
        expected = manifest[key]
        if current["sha256"] != expected["sha256"]:
            warnings.append(f"Data drift detected for {key}: checksum mismatch")
    for msg in warnings:
        logger.warning(msg)
    return warnings


def save_data_manifest(model_dir: Path, manifest: dict[str, Any]) -> None:
    save_json(model_dir / "data_manifest.json", manifest)


def load_data_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "data_manifest.json"
    if not path.exists():
        return {}
    return load_json(path)
