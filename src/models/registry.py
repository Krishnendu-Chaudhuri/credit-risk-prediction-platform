"""Model registry with versioned artifact directories."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.config.paths import MODEL_DIR as DEFAULT_MODEL_DIR
from src.models.promotion import validate_promotion
from src.utils import get_env_path, load_json, save_json


class ModelStatus(StrEnum):
    staged = "staged"
    current = "current"
    archived = "archived"


def registry_root(model_dir: Path | None = None) -> Path:
    base = model_dir or get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    reg = base / "registry"
    reg.mkdir(parents=True, exist_ok=True)
    return reg


def current_pointer_path(model_dir: Path | None = None) -> Path:
    base = model_dir or get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    return base / "current.json"


def new_version_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def version_dir(version_id: str, model_dir: Path | None = None) -> Path:
    return registry_root(model_dir) / version_id


def save_version_artifacts(version_id: str, artifacts: dict[str, Any], model_dir: Path | None = None) -> Path:
    base = model_dir or get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    target = version_dir(version_id, base)
    target.mkdir(parents=True, exist_ok=True)
    from src.models.storage import get_model_storage, save_artifact_bundle

    prefix = f"registry/{version_id}"
    storage = get_model_storage(base)
    save_artifact_bundle(storage, prefix, artifacts)
    save_json(target / "version.json", {"version_id": version_id, "status": ModelStatus.staged})
    return target


def list_versions(model_dir: Path | None = None) -> list[dict[str, Any]]:
    reg = registry_root(model_dir)
    versions = []
    for path in sorted(reg.iterdir()):
        if path.is_dir() and (path / "version.json").exists():
            versions.append(load_json(path / "version.json"))
    return versions


def get_current_version(model_dir: Path | None = None) -> str | None:
    ptr = current_pointer_path(model_dir)
    if not ptr.exists():
        return None
    return load_json(ptr).get("version_id")


def resolve_model_dir(model_dir: Path | None = None) -> Path:
    base = model_dir or get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    current_id = get_current_version(base)
    if current_id:
        candidate = version_dir(current_id, base)
        if candidate.exists():
            return candidate
    return base


def promote_version(version_id: str, model_dir: Path | None = None, *, force: bool = False) -> None:
    base = model_dir or get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    target = version_dir(version_id, base)
    if not target.exists():
        raise FileNotFoundError(f"Version {version_id} not found")

    current_id = get_current_version(base)
    champion_dir = version_dir(current_id, base) if current_id else None
    validate_promotion(target, champion_dir, force=force)

    if current_id:
        old_path = version_dir(current_id, base) / "version.json"
        if old_path.exists():
            meta = load_json(old_path)
            meta["status"] = ModelStatus.archived
            save_json(old_path, meta)

    meta = load_json(target / "version.json")
    meta["status"] = ModelStatus.current
    save_json(target / "version.json", meta)
    save_json(current_pointer_path(base), {"version_id": version_id, "status": ModelStatus.current})

    from src.models.storage import get_model_storage

    storage = get_model_storage(base)
    version_prefix = f"registry/{version_id}"
    for key in storage.list_keys(version_prefix):
        filename = key.split("/")[-1]
        storage.copy_file(key, filename)

    keep_last_n = int(os.getenv("REGISTRY_KEEP_LAST_N", "5"))
    prune_registry(keep_last_n=keep_last_n, model_dir=base)


def prune_registry(keep_last_n: int = 5, model_dir: Path | None = None) -> list[str]:
    """Remove archived versions beyond the most recent N; never delete current/staged."""
    base = model_dir or get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)
    protected = {get_current_version(base)}
    protected.update(
        version["version_id"]
        for version in list_versions(base)
        if version.get("status") in {ModelStatus.current, ModelStatus.staged}
    )

    archived = [
        version
        for version in list_versions(base)
        if version.get("status") == ModelStatus.archived and version.get("version_id") not in protected
    ]
    archived.sort(key=lambda item: item.get("version_id", ""), reverse=True)
    to_remove = archived[keep_last_n:]

    pruned: list[str] = []
    for version in to_remove:
        version_id = version.get("version_id")
        if not version_id:
            continue
        target = version_dir(version_id, base)
        from src.models.storage import get_model_storage

        storage = get_model_storage(base)
        if target.exists() or storage.list_keys(f"registry/{version_id}"):
            storage.delete_prefix(f"registry/{version_id}")
            if target.exists():
                shutil.rmtree(target)
            pruned.append(version_id)
    return pruned


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model registry utilities")
    parser.add_argument("command", choices=["prune"])
    parser.add_argument("--keep-last-n", type=int, default=int(os.getenv("REGISTRY_KEEP_LAST_N", "5")))
    args = parser.parse_args()
    if args.command == "prune":
        removed = prune_registry(keep_last_n=args.keep_last_n)
        print(f"Pruned {len(removed)} archived version(s): {', '.join(removed) if removed else 'none'}")


if __name__ == "__main__":
    _cli()
