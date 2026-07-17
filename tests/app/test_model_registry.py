"""Tests for versioned model registry."""

from __future__ import annotations

from pathlib import Path

from src.models.registry import (
    ModelStatus,
    get_current_version,
    list_versions,
    new_version_id,
    promote_version,
    prune_registry,
    save_version_artifacts,
    version_dir,
)
from src.utils import load_json, models_exist, save_json


def _save_stub_version(model_dir: Path, version_id: str, status: ModelStatus) -> None:
    save_version_artifacts(
        version_id,
        {
            "preprocessor.joblib": {"stub": True},
            "lr_model.joblib": {"stub": True},
            "xgb_model.joblib": {"stub": True},
            "training_metadata.json": {"best_model": "xgb"},
        },
        model_dir,
    )
    save_json(version_dir(version_id, model_dir) / "version.json", {"version_id": version_id, "status": status})


def test_registry_staged_and_promote(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    version_id = new_version_id()
    save_version_artifacts(
        version_id,
        {
            "preprocessor.joblib": {"stub": True},
            "lr_model.joblib": {"stub": True},
            "xgb_model.joblib": {"stub": True},
            "training_metadata.json": {"best_model": "xgb"},
        },
        model_dir,
    )
    versions = list_versions(model_dir)
    assert any(v["version_id"] == version_id and v["status"] == ModelStatus.staged for v in versions)
    assert get_current_version(model_dir) is None

    promote_version(version_id, model_dir)
    assert get_current_version(model_dir) == version_id
    meta = load_json(model_dir / "registry" / version_id / "version.json")
    assert meta["status"] == ModelStatus.current
    assert models_exist(model_dir)


def test_prune_registry_keeps_current_and_staged(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    archived_ids = [f"2025010{i}T12000{i}Z" for i in range(1, 8)]
    for version_id in archived_ids:
        _save_stub_version(model_dir, version_id, ModelStatus.archived)

    current_id = "20250110T120010Z"
    staged_id = "20250111T120011Z"
    _save_stub_version(model_dir, current_id, ModelStatus.current)
    _save_stub_version(model_dir, staged_id, ModelStatus.staged)
    save_json(model_dir / "current.json", {"version_id": current_id, "status": ModelStatus.current})

    pruned = prune_registry(keep_last_n=5, model_dir=model_dir)
    assert len(pruned) == 2
    assert set(pruned) == {archived_ids[0], archived_ids[1]}

    remaining = {version["version_id"] for version in list_versions(model_dir)}
    assert current_id in remaining
    assert staged_id in remaining
    assert archived_ids[0] not in remaining
    assert archived_ids[1] not in remaining
    assert archived_ids[-1] in remaining
