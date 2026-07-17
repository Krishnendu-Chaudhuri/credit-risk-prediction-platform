"""JSON and model artifact I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_artifacts(model_dir: Path, artifacts: dict[str, Any]) -> None:
    from src.models.storage import get_model_storage, save_artifact_bundle

    storage = get_model_storage(model_dir)
    save_artifact_bundle(storage, "", artifacts)


def load_artifacts(model_dir: Path) -> dict[str, Any]:
    from src.models.storage import get_model_storage, load_artifact_bundle

    storage = get_model_storage(model_dir)
    try:
        return load_artifact_bundle(storage, "")
    except (FileNotFoundError, OSError):
        return {
            "preprocessor": joblib.load(model_dir / "preprocessor.joblib"),
            "lr_model": joblib.load(model_dir / "lr_model.joblib"),
            "xgb_model": joblib.load(model_dir / "xgb_model.joblib"),
            "metadata": load_json(model_dir / "training_metadata.json"),
        }


def models_exist(model_dir: Path) -> bool:
    resolved = model_dir
    try:
        from src.models.registry import resolve_model_dir

        resolved = resolve_model_dir(model_dir)
    except Exception:
        resolved = model_dir
    required = ["preprocessor.joblib", "lr_model.joblib", "xgb_model.joblib", "training_metadata.json"]
    from src.models.storage import get_model_storage

    storage = get_model_storage(resolved)
    return all(storage.exists(name) for name in required)
