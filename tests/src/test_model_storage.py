"""Tests for model artifact storage backends."""

from __future__ import annotations

from pathlib import Path

import pytest
from moto import mock_aws
from src.models.storage import LocalModelStorage, S3ModelStorage, get_model_storage, save_artifact_bundle


def test_local_storage_round_trip(tmp_path: Path) -> None:
    storage = LocalModelStorage(tmp_path / "models")
    save_artifact_bundle(
        storage,
        "registry/v1",
        {
            "preprocessor.joblib": {"stub": True},
            "training_metadata.json": {"best_model": "xgb"},
        },
    )
    assert storage.exists("registry/v1/preprocessor.joblib")
    assert storage.read_bytes("registry/v1/training_metadata.json")


@mock_aws
def test_s3_storage_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    monkeypatch.setenv("MODEL_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET", "test-models")
    monkeypatch.setenv("S3_PREFIX", "backend/models")

    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-models")
    storage = get_model_storage(Path("backend/models"))
    assert isinstance(storage, S3ModelStorage)

    save_artifact_bundle(
        storage,
        "registry/v1",
        {
            "preprocessor.joblib": {"stub": True},
            "training_metadata.json": {"best_model": "xgb"},
        },
    )
    assert storage.exists("registry/v1/preprocessor.joblib")
    keys = storage.list_keys("registry/v1")
    assert any(k.endswith("preprocessor.joblib") for k in keys)
