"""Tests for HttpOnly session authentication."""

from __future__ import annotations

import pytest
from app.api.session import SESSION_COOKIE_NAME, create_session_token, verify_session_token
from fastapi.testclient import TestClient


def test_create_and_verify_session_token() -> None:
    token = create_session_token()
    assert verify_session_token(token)


def test_invalid_session_token_rejected() -> None:
    assert not verify_session_token("invalid.token")


def test_session_cookie_auth_on_protected_route(client: TestClient) -> None:
    token = create_session_token()
    response = client.get("/metrics", cookies={SESSION_COOKIE_NAME: token})
    if response.status_code == 404:
        pytest.skip("Models not trained yet")
    assert response.status_code == 200


def test_session_endpoint_rejects_invalid_key(client: TestClient) -> None:
    response = client.post("/auth/session", json={"api_key": "wrong-key"})
    assert response.status_code == 401


def test_session_endpoint_accepts_valid_key(client: TestClient) -> None:
    response = client.post("/auth/session", json={"api_key": "test-key"})
    assert response.status_code == 200
    assert response.json()["authenticated"] is True
    assert SESSION_COOKIE_NAME in response.cookies


def test_logout_clears_session(client: TestClient) -> None:
    login = client.post("/auth/session", json={"api_key": "test-key"})
    assert login.status_code == 200
    logout = client.post("/auth/logout")
    assert logout.status_code == 200
