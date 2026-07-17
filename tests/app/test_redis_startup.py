"""Tests for Redis/worker startup validation."""

from __future__ import annotations

import pytest
from app.api.worker_config import get_worker_count, validate_redis_for_workers


def test_get_worker_count_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("GUNICORN_WORKERS", raising=False)
    assert get_worker_count() == 1


def test_get_worker_count_reads_uvicorn_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    assert get_worker_count() == 4


def test_validate_redis_warns_single_worker_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    validate_redis_for_workers()


def test_validate_redis_raises_multiworker_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    monkeypatch.setenv("REQUIRE_REDIS_FOR_MULTIWORKER", "true")
    with pytest.raises(RuntimeError, match="Multi-worker deployment"):
        validate_redis_for_workers()


def test_validate_redis_ok_when_ping_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    import app.api.redis_store as redis_store

    monkeypatch.setattr(redis_store, "ping_redis", lambda: True)
    validate_redis_for_workers()
