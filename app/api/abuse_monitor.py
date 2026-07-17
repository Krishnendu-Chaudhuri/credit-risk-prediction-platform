"""Heuristic abuse monitoring for /predict."""

from __future__ import annotations

import os
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from app.api import redis_store

BOUNDARY = 0.5
BOUNDARY_TOL = float(os.getenv("ABUSE_BOUNDARY_TOL", "0.01"))
BOUNDARY_RATIO = float(os.getenv("ABUSE_BOUNDARY_RATIO", "0.8"))
FREQ_THRESHOLD = int(os.getenv("ABUSE_FREQ_THRESHOLD", "50"))
MAX_WINDOW = 200

_lock = Lock()
_recent_pds: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=MAX_WINDOW))
_flags: dict[str, list[str]] = defaultdict(list)


def _pds_key(api_key: str) -> str:
    return f"abuse:pds:{api_key}"


def _flags_key(api_key: str) -> str:
    return f"abuse:flags:{api_key}"


def _evaluate_window(window: list[float]) -> str | None:
    if len(window) < FREQ_THRESHOLD:
        return None
    near_boundary = sum(1 for p in window if abs(p - BOUNDARY) <= BOUNDARY_TOL)
    if near_boundary / len(window) >= BOUNDARY_RATIO:
        return "High-frequency near-boundary PD queries detected"
    return None


def record_predict(api_key: str, pd_value: float) -> None:
    redis = redis_store.get_redis()
    if redis is not None:
        pds_key = _pds_key(api_key)
        redis.lpush(pds_key, str(pd_value))
        redis.ltrim(pds_key, 0, MAX_WINDOW - 1)
        raw_window = redis.lrange(pds_key, 0, MAX_WINDOW - 1)
        window = [float(value) for value in reversed(raw_window)]
        msg = _evaluate_window(window)
        if msg and msg not in redis.smembers(_flags_key(api_key)):
            redis.sadd(_flags_key(api_key), msg)
        return

    with _lock:
        _recent_pds[api_key].append(pd_value)
        window = list(_recent_pds[api_key])
        msg = _evaluate_window(window)
        if msg and msg not in _flags[api_key]:
            _flags[api_key].append(msg)


def get_abuse_flags(api_key: str | None = None) -> dict[str, Any]:
    redis = redis_store.get_redis()
    if redis is not None:
        if api_key:
            return {"client": api_key, "flags": sorted(redis.smembers(_flags_key(api_key)))}
        keys = redis.keys("abuse:flags:*")
        return {key.removeprefix("abuse:flags:"): sorted(redis.smembers(key)) for key in keys}

    with _lock:
        if api_key:
            return {"client": api_key, "flags": list(_flags.get(api_key, []))}
        return {k: list(v) for k, v in _flags.items()}


def clear_abuse_state() -> None:
    """Clear in-memory abuse state (for tests)."""
    with _lock:
        _recent_pds.clear()
        _flags.clear()
