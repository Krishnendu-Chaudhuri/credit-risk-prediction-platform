"""Model artifact storage backends (local filesystem and S3-compatible)."""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

import joblib


class ModelStorage(Protocol):
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def read_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def copy_file(self, src_key: str, dest_key: str) -> None: ...


class LocalModelStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base_dir / key

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        root = self._path(prefix)
        if not root.exists():
            return []
        if root.is_file():
            return [prefix]
        keys: list[str] = []
        for path in root.rglob("*"):
            if path.is_file():
                keys.append(str(path.relative_to(self.base_dir)).replace("\\", "/"))
        return keys

    def delete_prefix(self, prefix: str) -> None:
        target = self._path(prefix)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    def copy_file(self, src_key: str, dest_key: str) -> None:
        src = self._path(src_key)
        dest = self._path(dest_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


class S3ModelStorage:
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None) -> None:
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        client_kwargs: dict[str, Any] = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        self.client = boto3.client("s3", **client_kwargs)

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def write_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def read_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        body = response["Body"].read()
        return body if isinstance(body, bytes) else bytes(body)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix).rstrip("/") + "/"
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                raw = obj["Key"]
                rel = raw[len(self.prefix) + 1 :] if self.prefix else raw
                keys.append(rel)
        return keys

    def delete_prefix(self, prefix: str) -> None:
        keys = self.list_keys(prefix)
        if not keys:
            return
        self.client.delete_objects(
            Bucket=self.bucket,
            Delete={"Objects": [{"Key": self._full_key(key)} for key in keys]},
        )

    def copy_file(self, src_key: str, dest_key: str) -> None:
        self.client.copy_object(
            Bucket=self.bucket,
            Key=self._full_key(dest_key),
            CopySource={"Bucket": self.bucket, "Key": self._full_key(src_key)},
        )


def get_model_storage(model_dir: Path | None = None) -> ModelStorage:
    backend = os.getenv("MODEL_STORAGE_BACKEND", "local").strip().lower()
    base = model_dir or Path(os.getenv("MODEL_DIR", "backend/models"))
    if backend == "s3":
        bucket = os.getenv("S3_BUCKET", "")
        if not bucket:
            raise RuntimeError("S3_BUCKET is required when MODEL_STORAGE_BACKEND=s3")
        prefix = os.getenv("S3_PREFIX", str(base).replace("\\", "/")).strip("/")
        endpoint = os.getenv("S3_ENDPOINT_URL") or None
        return S3ModelStorage(bucket=bucket, prefix=prefix, endpoint_url=endpoint)
    return LocalModelStorage(base)


def _storage_key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def save_artifact_bundle(storage: ModelStorage, prefix: str, artifacts: dict[str, Any]) -> None:
    for name, obj in artifacts.items():
        key = _storage_key(prefix, name)
        if name.endswith(".json"):
            if isinstance(obj, (bytes, str)):
                data = obj if isinstance(obj, bytes) else obj.encode()
            else:
                data = json.dumps(obj, indent=2).encode()
        else:
            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            data = buffer.getvalue()
        storage.write_bytes(key, data)


def load_artifact_bundle(storage: ModelStorage, prefix: str) -> dict[str, Any]:
    keys = storage.list_keys(prefix)
    json_keys = {k.split("/")[-1] for k in keys if k.endswith(".json")}
    result: dict[str, Any] = {}
    if any(k.endswith("preprocessor.joblib") for k in keys):
        result["preprocessor"] = joblib.load(io.BytesIO(storage.read_bytes(_storage_key(prefix, "preprocessor.joblib"))))
        result["lr_model"] = joblib.load(io.BytesIO(storage.read_bytes(_storage_key(prefix, "lr_model.joblib"))))
        result["xgb_model"] = joblib.load(io.BytesIO(storage.read_bytes(_storage_key(prefix, "xgb_model.joblib"))))
    if "training_metadata.json" in json_keys:
        result["metadata"] = json.loads(storage.read_bytes(_storage_key(prefix, "training_metadata.json")))
    return result
