# Scaling Guide

This document describes how to scale the Credit Risk PD & Stress Testing Engine beyond a single-node deployment.

## Current architecture (single instance)

| Component | Technology | Notes |
|-----------|------------|-------|
| Prediction audit | SQLite (`app/api/data/audit.db`) | WAL mode enabled for better write concurrency |
| Rate limiting / abuse monitoring | Redis (optional) or in-memory | In-memory is correct only when `UVICORN_WORKERS=1` |
| Model artifacts | Local filesystem (`backend/models/`) | Versioned under `registry/` |
| Training jobs | Redis or in-memory | Same multi-worker constraint as rate limiting |

## SQLite audit store

### Schema

```sql
CREATE TABLE predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    shap_json TEXT,
    reason_codes_json TEXT
);
```

### WAL mode

`app/api/audit_store.py` sets `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` on each connection. This improves concurrent read/write performance on a single host but does **not** support multiple application instances writing to the same file.

## Multi-instance requirements

When running more than one backend replica:

1. **Redis** — Set `REDIS_URL` and `UVICORN_WORKERS=1` per replica (or use a process manager with shared Redis). Set `REQUIRE_REDIS_FOR_MULTIWORKER=true` to fail fast if Redis is missing.
2. **Audit store** — Migrate from SQLite to PostgreSQL (or another shared database). SQLite must not be used across instances.
3. **Model artifacts** — Use shared object storage (see `MODEL_STORAGE_BACKEND=s3`) or a shared volume with consistent promotion semantics.

## PostgreSQL migration path (audit store)

### Recommended approach

1. Introduce an `AuditStore` protocol in `app/api/audit_store.py`:

   ```python
   class AuditStore(Protocol):
       def log_prediction(...) -> str: ...
       def get_prediction_audit(prediction_id: str) -> dict | None: ...
       def recent_inputs(limit: int = 500) -> list[dict]: ...
   ```

2. Keep the current SQLite functions as `SQLiteAuditStore` (default when `AUDIT_DB_PATH` ends in `.db`).

3. Add `PostgresAuditStore` using SQLAlchemy or `asyncpg`:

   ```sql
   CREATE TABLE predictions (
       prediction_id TEXT PRIMARY KEY,
       created_at TIMESTAMPTZ NOT NULL,
       inputs_json JSONB NOT NULL,
       outputs_json JSONB NOT NULL,
       shap_json JSONB,
       reason_codes_json JSONB,
       user_id TEXT
   );
   CREATE INDEX idx_predictions_created_at ON predictions (created_at DESC);
   ```

4. Select implementation via `AUDIT_STORE_BACKEND=sqlite|postgres` and `DATABASE_URL` for Postgres.

### Files to change

| File | Change |
|------|--------|
| `app/api/audit_store.py` | Extract protocol; add Postgres backend |
| `src/pipelines/predict.py` | Accept optional `user_id` for attribution |
| `app/api/main.py` | Wire store factory at startup |
| `.env.example` | Document `DATABASE_URL`, `AUDIT_STORE_BACKEND` |

## Related configuration

| Variable | Purpose |
|----------|---------|
| `AUDIT_DB_PATH` | SQLite file path (default `app/api/data/audit.db`) |
| `UVICORN_WORKERS` | Worker count; checked against Redis at startup |
| `REQUIRE_REDIS_FOR_MULTIWORKER` | Fail startup when workers > 1 without Redis |
| `REDIS_URL` | Shared state for rate limits, abuse monitoring, jobs |

See the README **Environment Variables** section for Redis and worker configuration. See [AUTH.md](AUTH.md) for authentication modes.
