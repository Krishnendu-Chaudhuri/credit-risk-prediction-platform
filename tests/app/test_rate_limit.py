"""Tests for predict rate limiting."""

from __future__ import annotations

import pytest
from app.api.rate_limit import clear_rate_limits, enforce_rate_limit
from fastapi import HTTPException
from starlette.requests import Request


def _fake_request(api_key: str = "burst-key") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [(b"x-api-key", api_key.encode())],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def _assert_rate_limit_blocks_burst() -> None:
    for _ in range(3):
        enforce_rate_limit(_fake_request())
    with pytest.raises(HTTPException) as exc:
        enforce_rate_limit(_fake_request())
    assert exc.value.status_code == 429


def test_rate_limit_blocks_burst_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    from app.api import redis_store

    redis_store.reset_redis_client()
    monkeypatch.setenv("PREDICT_RATE_LIMIT", "3")
    import app.api.rate_limit as rl

    monkeypatch.setattr(rl, "PREDICT_RATE_LIMIT", 3)
    clear_rate_limits()
    _assert_rate_limit_blocks_burst()


def test_rate_limit_blocks_burst_redis(redis_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREDICT_RATE_LIMIT", "3")
    import app.api.rate_limit as rl

    monkeypatch.setattr(rl, "PREDICT_RATE_LIMIT", 3)
    redis_client.flushall()
    _assert_rate_limit_blocks_burst()
