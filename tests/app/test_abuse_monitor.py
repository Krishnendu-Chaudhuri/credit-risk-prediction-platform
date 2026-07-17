"""Tests for abuse monitoring heuristics."""

from __future__ import annotations

from app.api.abuse_monitor import clear_abuse_state, get_abuse_flags, record_predict


def test_abuse_flags_near_boundary_memory(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    from app.api import redis_store

    redis_store.reset_redis_client()
    clear_abuse_state()
    monkeypatch.setenv("ABUSE_FREQ_THRESHOLD", "10")
    import app.api.abuse_monitor as abuse

    monkeypatch.setattr(abuse, "FREQ_THRESHOLD", 10)

    client_key = "abuse-test-client"
    for _ in range(10):
        record_predict(client_key, 0.5)

    flags = get_abuse_flags(client_key)
    assert flags["flags"]


def test_abuse_flags_near_boundary_redis(redis_client, monkeypatch) -> None:
    redis_client.flushall()
    clear_abuse_state()
    monkeypatch.setenv("ABUSE_FREQ_THRESHOLD", "10")
    import app.api.abuse_monitor as abuse

    monkeypatch.setattr(abuse, "FREQ_THRESHOLD", 10)

    client_key = "abuse-redis-client"
    for _ in range(10):
        record_predict(client_key, 0.5005)

    flags = get_abuse_flags(client_key)
    assert flags["flags"]
