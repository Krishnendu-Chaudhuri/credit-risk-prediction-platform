"""Training job store backed by Redis or in-memory fallback."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from threading import Lock

from src.models.schemas import TrainConfig, TrainJobStatus, TrainResponse
from src.pipelines.train import train_models

from app.api import redis_store

_lock = Lock()
_jobs: dict[str, JobRecord] = {}


@dataclass
class JobRecord:
    job_id: str
    status: TrainJobStatus
    result: TrainResponse | None = None
    error: str | None = None


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _serialize_job(record: JobRecord) -> str:
    payload = {
        "job_id": record.job_id,
        "status": record.status.value,
        "result": record.result.model_dump() if record.result else None,
        "error": record.error,
    }
    return json.dumps(payload)


def _deserialize_job(raw: str) -> JobRecord:
    payload = json.loads(raw)
    result = TrainResponse.model_validate(payload["result"]) if payload.get("result") else None
    return JobRecord(
        job_id=payload["job_id"],
        status=TrainJobStatus(payload["status"]),
        result=result,
        error=payload.get("error"),
    )


def _save_job(record: JobRecord) -> None:
    redis = redis_store.get_redis()
    if redis is not None:
        redis.set(_job_key(record.job_id), _serialize_job(record))
        return
    with _lock:
        _jobs[record.job_id] = record


def _load_job(job_id: str) -> JobRecord | None:
    redis = redis_store.get_redis()
    if redis is not None:
        raw = redis.get(_job_key(job_id))
        return _deserialize_job(raw) if raw else None
    with _lock:
        return _jobs.get(job_id)


def create_job() -> JobRecord:
    job_id = str(uuid.uuid4())
    record = JobRecord(job_id=job_id, status=TrainJobStatus.pending)
    _save_job(record)
    return record


def get_job(job_id: str) -> JobRecord | None:
    return _load_job(job_id)


def _set_job_status(job_id: str, status: TrainJobStatus) -> None:
    record = _load_job(job_id)
    if record is None:
        return
    record.status = status
    _save_job(record)


def run_training_job(job_id: str, config: TrainConfig) -> None:
    _set_job_status(job_id, TrainJobStatus.running)
    try:
        result = train_models(config)
    except Exception as exc:
        record = _load_job(job_id)
        if record is not None:
            record.status = TrainJobStatus.failed
            record.error = str(exc)
            _save_job(record)
        return

    record = _load_job(job_id)
    if record is not None:
        record.status = TrainJobStatus.completed
        record.result = result
        _save_job(record)


def clear_jobs() -> None:
    """Clear in-memory jobs (for tests)."""
    with _lock:
        _jobs.clear()
