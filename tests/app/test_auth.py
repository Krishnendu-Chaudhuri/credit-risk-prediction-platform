"""Integration tests for API key authentication."""

from __future__ import annotations

import os

import pytest
from app.api import auth
from fastapi.testclient import TestClient


def test_validate_auth_config_raises_without_key_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    original_key = auth.API_KEY
    original_warned = auth._AUTH_WARNED
    try:
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "production")
        auth.API_KEY = None
        auth._AUTH_WARNED = False
        with pytest.raises(RuntimeError, match="API_KEY is required"):
            auth.validate_auth_config()
    finally:
        auth.API_KEY = original_key
        auth._AUTH_WARNED = original_warned


def test_validate_auth_config_generates_temp_key_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    original_key = auth.API_KEY
    original_warned = auth._AUTH_WARNED
    original_dev_logged = auth._DEV_KEY_LOGGED
    try:
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")
        auth.API_KEY = None
        auth._AUTH_WARNED = False
        auth._DEV_KEY_LOGGED = False
        auth.validate_auth_config()
        assert auth.API_KEY is not None
        assert len(auth.API_KEY) == 64
    finally:
        auth.API_KEY = original_key
        auth._AUTH_WARNED = original_warned
        auth._DEV_KEY_LOGGED = original_dev_logged


def test_protected_route_requires_api_key(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 401


def test_protected_route_rejects_invalid_key(client: TestClient) -> None:
    response = client.get("/metrics", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_protected_route_accepts_valid_key(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/metrics", headers=auth_headers)
    if response.status_code == 404:
        pytest.skip("Models not trained yet")
    assert response.status_code == 200


def test_health_remains_open_without_api_key(client: TestClient) -> None:
    assert os.environ.get("API_KEY") == "test-key"
    response = client.get("/health")
    assert response.status_code == 200
