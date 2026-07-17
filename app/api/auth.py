"""API key authentication for protected endpoints.

Supports shared-secret auth via X-API-Key header or HttpOnly session cookie,
and optional OAuth2/JWT mode via AUTH_MODE=oauth2.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from app.api.logging_config import get_logger
from app.api.oauth2 import extract_bearer_token, get_auth_mode, verify_access_token
from app.api.session import get_session_from_request, verify_session_token
from app.api.settings import get_api_key, is_dev_mode, load_env, set_api_key

logger = get_logger(__name__)

# Mutable runtime key — initialized by validate_auth_config() during startup.
API_KEY: str | None = None
_AUTH_WARNED = False
_DEV_KEY_LOGGED = False


def _is_dev_mode() -> bool:
    return is_dev_mode()


def _ensure_api_key_loaded() -> str | None:
    global API_KEY  # noqa: PLW0603
    if API_KEY is None:
        API_KEY = get_api_key()
    return API_KEY


def validate_auth_config() -> None:
    """Configure API key for the process; fail closed outside development/local."""
    global API_KEY, _AUTH_WARNED, _DEV_KEY_LOGGED  # noqa: PLW0603

    load_env()

    if get_auth_mode() == "oauth2":
        from app.api.oauth2 import validate_oauth2_config

        validate_oauth2_config()
        API_KEY = get_api_key()
        return

    API_KEY = get_api_key()
    if API_KEY:
        return

    if _is_dev_mode():
        API_KEY = secrets.token_hex(32)
        set_api_key(API_KEY)
        if not _DEV_KEY_LOGGED:
            logger.warning("Development mode detected. Using temporary API key.")
            _DEV_KEY_LOGGED = True
        if not _AUTH_WARNED:
            logger.warning("API_KEY was not set; generated ephemeral dev key (development mode only)")
            _AUTH_WARNED = True
        return

    raise RuntimeError("API_KEY is required outside development/local mode")


def _jwt_user_id(request: Request) -> str | None:
    token = extract_bearer_token(request.headers.get("Authorization"))
    if not token:
        return None
    try:
        payload = verify_access_token(token)
    except ValueError:
        return None
    sub = payload.get("sub")
    return str(sub) if sub else None


def resolve_authenticated_user(request: Request) -> str | None:
    """Return authenticated user id (JWT sub) when available."""
    user_id = _jwt_user_id(request)
    if user_id:
        return user_id
    api_key = request.headers.get("X-API-Key")
    current_key = _ensure_api_key_loaded()
    if current_key and api_key and secrets.compare_digest(api_key, current_key):
        return "api-key-client"
    session = get_session_from_request(request)
    if session and verify_session_token(session):
        return f"session:{session[:16]}"
    return None


def resolve_client_key(request: Request) -> str:
    """Stable client identifier for rate limiting and abuse monitoring."""
    user_id = resolve_authenticated_user(request)
    if user_id:
        return user_id
    return request.client.host if request.client else "anonymous"


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    global _AUTH_WARNED  # noqa: PLW0603

    current_key = _ensure_api_key_loaded()

    if get_auth_mode() == "oauth2":
        user_id = _jwt_user_id(request)
        if user_id:
            return
        if current_key and x_api_key is not None and secrets.compare_digest(x_api_key, current_key):
            return
        session = get_session_from_request(request)
        if session and verify_session_token(session):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )

    if not current_key:
        if _is_dev_mode():
            if not _AUTH_WARNED:
                logger.warning("API_KEY is not set; protected endpoints are open (dev mode only)")
                _AUTH_WARNED = True
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    if x_api_key is not None and secrets.compare_digest(x_api_key, current_key):
        return

    session = get_session_from_request(request)
    if session and verify_session_token(session):
        return

    bearer_user = _jwt_user_id(request)
    if bearer_user:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
