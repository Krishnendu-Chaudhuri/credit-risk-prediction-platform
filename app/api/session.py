"""HttpOnly cookie session tokens for browser clients."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Any

from fastapi import Request, Response

from app.api.settings import is_dev_mode

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
_DEV_SECRET_WARNED = False


def _session_secret() -> str:
    global _DEV_SECRET_WARNED  # noqa: PLW0603

    secret = os.getenv("SESSION_SECRET")
    if secret:
        return secret
    if is_dev_mode():
        if not _DEV_SECRET_WARNED:
            logger.warning("SESSION_SECRET is not set; using ephemeral dev secret")
            _DEV_SECRET_WARNED = True
        return "dev-session-secret"
    raise RuntimeError("SESSION_SECRET is required outside development/local mode")


def validate_session_config() -> None:
    """Ensure session signing secret is configured outside dev mode."""
    if is_dev_mode():
        return
    if not os.getenv("SESSION_SECRET"):
        raise RuntimeError("SESSION_SECRET is required outside development/local mode")


def create_session_token() -> str:
    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
        "nonce": secrets.token_urlsafe(8),
    }
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(_session_secret().encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_session_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    body, sig = token.rsplit(".", 1)
    expected = hmac.new(_session_secret().encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False
    try:
        payload: dict[str, Any] = json.loads(base64.urlsafe_b64decode(body.encode()))
    except (json.JSONDecodeError, ValueError):
        return False
    return int(payload.get("exp", 0)) >= int(time.time())


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
        secure=not is_dev_mode(),
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def get_session_from_request(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE_NAME)
