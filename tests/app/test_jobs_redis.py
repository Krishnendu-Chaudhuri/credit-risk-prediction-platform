"""Tests for Redis-backed training jobs."""

from __future__ import annotations

from app.api.jobs import clear_jobs, create_job, get_job, run_training_job
from src.models.schemas import TrainConfig, TrainJobStatus


def test_job_lifecycle_in_redis(redis_client, monkeypatch) -> None:
    clear_jobs()
    redis_client.flushall()

    captured: dict[str, TrainConfig] = {}

    def fake_train(config: TrainConfig):
        captured["config"] = config
        from src.models.schemas import ModelMetrics, TrainResponse

        metrics = ModelMetrics(
            model_name="xgb",
            roc_auc=0.8,
            f1=0.7,
            precision=0.7,
            recall=0.7,
            accuracy=0.8,
            ks=0.4,
            gini=0.6,
        )
        return TrainResponse(
            best_model="xgb",
            metrics={"xgb": metrics},
            feature_importance={"loan_amnt": 0.5},
            training_metadata={"trained_at": "now"},
        )

    monkeypatch.setattr("app.api.jobs.train_models", fake_train)

    job = create_job()
    assert job.status == TrainJobStatus.pending

    run_training_job(job.job_id, TrainConfig())
    stored = get_job(job.job_id)
    assert stored is not None
    assert stored.status == TrainJobStatus.completed
    assert stored.result is not None
    assert stored.result.best_model == "xgb"
