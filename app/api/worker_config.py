"""Worker count detection and Redis requirements for multi-worker deployments."""

from __future__ import annotations

import os

from app.api.logging_config import get_logger
from app.api.redis_store import is_redis_enabled, ping_redis

logger = get_logger(__name__)


def get_worker_count() -> int:
    """Return configured Uvicorn/Gunicorn worker count (default 1)."""
    for env_name in ("UVICORN_WORKERS", "WEB_CONCURRENCY", "GUNICORN_WORKERS"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            try:
                count = int(raw)
            except ValueError:
                logger.warning("Invalid %s=%r; defaulting worker count to 1", env_name, raw)
                return 1
            return max(1, count)
    return 1


def validate_redis_for_workers() -> None:
    """Warn or fail when multi-worker deployment lacks reachable Redis."""
    worker_count = get_worker_count()
    redis_ok = ping_redis()
    strict = os.getenv("REQUIRE_REDIS_FOR_MULTIWORKER", "").strip().lower() in {"1", "true", "yes"}

    if redis_ok:
        logger.info("Redis connected; shared state enabled for %s worker(s)", worker_count)
        return

    if worker_count > 1:
        message = (
            f"Multi-worker deployment detected ({worker_count} workers) but Redis is unavailable. "
            "Rate limiting, abuse monitoring, and job state will be inconsistent across workers."
        )
        if strict:
            logger.critical(message)
            raise RuntimeError(message)
        logger.error(message)
        return

    if is_redis_enabled():
        logger.warning("REDIS_URL is set but Redis is unreachable; using in-memory single-worker fallback")
    else:
        logger.warning(
            "Redis is not connected; using in-memory single-worker fallback "
            "(acceptable for UVICORN_WORKERS=1 only)"
        )
