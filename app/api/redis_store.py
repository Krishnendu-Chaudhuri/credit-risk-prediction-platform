"""Redis client factory with in-memory dev fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_CLIENT: Any | None = None
_REDIS_UNAVAILABLE_LOGGED = False


def is_redis_enabled() -> bool:
    return bool(os.getenv("REDIS_URL", "").strip())


def ping_redis() -> bool:
    """Return True when REDIS_URL is set and the server responds to PING."""
    if not is_redis_enabled():
        return False
    try:
        import redis

        client = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        return bool(client.ping())
    except Exception as exc:
        logger.error("Redis ping failed: %s", exc)
        return False


def get_redis() -> Any | None:
    global _REDIS_CLIENT, _REDIS_UNAVAILABLE_LOGGED  # noqa: PLW0603

    if not is_redis_enabled():
        if not _REDIS_UNAVAILABLE_LOGGED:
            logger.warning("Redis unavailable; using in-memory single-worker dev mode")
            _REDIS_UNAVAILABLE_LOGGED = True
        return None

    if _REDIS_CLIENT is None:
        import redis

        _REDIS_CLIENT = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _REDIS_CLIENT


def reset_redis_client() -> None:
    """Reset cached client (for tests)."""
    global _REDIS_CLIENT, _REDIS_UNAVAILABLE_LOGGED  # noqa: PLW0603

    _REDIS_CLIENT = None
    _REDIS_UNAVAILABLE_LOGGED = False
