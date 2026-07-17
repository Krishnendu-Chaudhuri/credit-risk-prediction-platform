"""Tests for OAuth2/JWT authentication."""

from __future__ import annotations

import time

import jwt
import pytest

from app.api.oauth2 import create_access_token, verify_access_token

JWT_ISSUER = "credit-risk-pd-engine"
JWT_AUDIENCE = "credit-risk-api"


def test_create_and_verify_access_token() -> None:
    token = create_access_token("user-123")
    payload = verify_access_token(token)
    assert payload["sub"] == "user-123"


def test_verify_rejects_expired_token(monkeypatch: pytest.MonkeyPatch) -> None:
    token = create_access_token("user-123", ttl_seconds=-10)
    with pytest.raises(ValueError, match="expired"):
        verify_access_token(token)


def test_verify_rejects_malformed_token() -> None:
    with pytest.raises(ValueError, match="Invalid token"):
        verify_access_token("not-a-jwt")


def test_verify_rejects_wrong_signature() -> None:
    bad = jwt.encode(
        {"sub": "user-123", "iat": int(time.time()), "exp": int(time.time()) + 3600, "iss": JWT_ISSUER, "aud": JWT_AUDIENCE},
        "wrong-secret",
        algorithm="HS256",
    )
    with pytest.raises(ValueError, match="Invalid token"):
        verify_access_token(bad)


def test_oauth2_token_allows_protected_route(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_MODE", "oauth2")
    monkeypatch.setenv("API_KEY", "test-key")
    import app.api.auth as auth_module

    monkeypatch.setattr(auth_module, "API_KEY", "test-key")

    token_resp = client.post(
        "/auth/token",
        json={"username": "analyst", "password": "test-key"},
    )
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]
    response = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code in {200, 404}


def test_oauth2_rejects_invalid_credentials(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_MODE", "oauth2")
    monkeypatch.setenv("API_KEY", "test-key")
    import app.api.auth as auth_module

    monkeypatch.setattr(auth_module, "API_KEY", "test-key")

    response = client.post(
        "/auth/token",
        json={"username": "analyst", "password": "wrong"},
    )
    assert response.status_code == 401
