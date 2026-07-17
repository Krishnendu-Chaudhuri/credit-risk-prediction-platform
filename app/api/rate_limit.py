"""Rate limiting for API endpoints."""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

from app.api import redis_store
from app.api.auth import resolve_client_key

PREDICT_RATE_LIMIT = int(os.getenv("PREDICT_RATE_LIMIT", "100"))
WINDOW_SECONDS = 60

_lock = Lock()
_requests: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    return resolve_client_key(request)


def _enforce_rate_limit_redis(key: str, max_requests: int, now: float) -> None:
    redis = redis_store.get_redis()
    if redis is None:
        return
    redis_key = f"ratelimit:{key}"
    window_start = now - WINDOW_SECONDS
    redis.zremrangebyscore(redis_key, 0, window_start)
    count = redis.zcard(redis_key)
    if count >= max_requests:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    member = secrets.token_hex(8)
    redis.zadd(redis_key, {member: now})
    redis.expire(redis_key, WINDOW_SECONDS + 1)


def _enforce_rate_limit_memory(key: str, max_requests: int, now: float) -> None:
    with _lock:
        bucket = _requests[key]
        while bucket and now - bucket[0] > WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= max_requests:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        bucket.append(now)


def enforce_rate_limit(request: Request, limit: int | None = None) -> None:
    max_requests = limit or PREDICT_RATE_LIMIT
    key = _client_key(request)
    now = time.time()
    if redis_store.get_redis() is not None:
        _enforce_rate_limit_redis(key, max_requests, now)
    else:
        _enforce_rate_limit_memory(key, max_requests, now)


def clear_rate_limits() -> None:
    """Clear in-memory counters (for tests)."""
    with _lock:
        _requests.clear()
