"""OAuth2/JWT authentication helpers (demo token issuance)."""

from __future__ import annotations

import os
import time
from typing import Any

import jwt

from app.api.logging_config import get_logger
from app.api.settings import load_env

logger = get_logger(__name__)


def get_auth_mode() -> str:
    load_env()
    mode = os.getenv("AUTH_MODE", "api_key").strip().lower()
    return mode if mode in {"api_key", "oauth2"} else "api_key"


def _jwt_secret() -> str:
    load_env()
    return os.getenv("JWT_SECRET") or os.getenv("SESSION_SECRET") or "dev-jwt-secret"


def _jwt_algorithm() -> str:
    load_env()
    return os.getenv("JWT_ALGORITHM", "HS256")


def _jwt_issuer() -> str:
    load_env()
    return os.getenv("JWT_ISSUER", "credit-risk-pd-engine")


def _jwt_audience() -> str:
    load_env()
    return os.getenv("JWT_AUDIENCE", "credit-risk-api")


def jwt_ttl_seconds() -> int:
    load_env()
    return int(os.getenv("JWT_TTL_SECONDS", "3600"))


# Backward-compatible module constant (resolved after load_env).
JWT_TTL_SECONDS = jwt_ttl_seconds()


def validate_oauth2_config() -> None:
    if get_auth_mode() != "oauth2":
        return
    secret = _jwt_secret()
    if not secret or secret == "dev-jwt-secret":
        from app.api.settings import is_dev_mode

        if not is_dev_mode():
            raise RuntimeError("JWT_SECRET (or SESSION_SECRET) is required when AUTH_MODE=oauth2")


def create_access_token(subject: str, *, ttl_seconds: int | None = None) -> str:
    now = int(time.time())
    ttl = ttl_seconds or jwt_ttl_seconds()
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + ttl,
        "iss": _jwt_issuer(),
        "aud": _jwt_audience(),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_jwt_algorithm())


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[_jwt_algorithm()],
            issuer=_jwt_issuer(),
            audience=_jwt_audience(),
        )
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError("Invalid token") from exc


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()
