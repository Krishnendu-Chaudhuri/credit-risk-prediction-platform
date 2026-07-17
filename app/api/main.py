"""FastAPI backend for Credit Risk PD & Stress Testing Engine."""

from __future__ import annotations

import os
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.settings import load_env

load_env()

from src.config.constants import MAX_BATCH_LOANS
from src.config.paths import CREDIT_CSV_FILE as CREDIT_CSV_PATH
from src.config.paths import MODEL_DIR as DEFAULT_MODEL_DIR
from src.config.paths import require_file
from src.features.encoding_monitor import get_unknown_category_count
from src.models.promotion import PromotionBlockedError
from src.models.registry import get_current_version, list_versions, promote_version, resolve_model_dir
from src.models.schemas import (
    LoanInput,
    PredictionOutput,
    StressTestRequest,
    StressTestResponse,
    TrainConfig,
    TrainJobResponse,
    TrainJobStatus,
    TrainJobStatusResponse,
)
from src.pipelines.monitoring.drift import compute_psi_report
from src.pipelines.predict import predict_pd
from src.utils import get_env_path, load_json, models_exist

from app.api.abuse_monitor import get_abuse_flags, record_predict
from app.api.audit_store import get_prediction_audit, recent_inputs
from app.api.auth import API_KEY, resolve_authenticated_user, resolve_client_key, validate_auth_config, verify_api_key
from app.api.jobs import create_job, get_job, run_training_job
from app.api.logging_config import configure_logging
from app.api.oauth2 import create_access_token, get_auth_mode
from app.api.rate_limit import enforce_rate_limit
from app.api.request_middleware import RequestIdMiddleware
from app.api.session import (
    SESSION_TTL_SECONDS,
    clear_session_cookie,
    create_session_token,
    set_session_cookie,
    validate_session_config,
)
from app.api.settings import get_api_key
from app.api.stress.portfolio import run_portfolio_stress_test
from app.api.worker_config import validate_redis_for_workers

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    validate_auth_config()
    validate_session_config()
    validate_redis_for_workers()
    yield


app = FastAPI(
    title="Credit Risk PD & Stress Testing Engine",
    description="PD modeling, stress testing, and regulatory ECL calculations",
    version="1.0.0",
    lifespan=lifespan,
)


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = _parse_cors_origins()

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = get_env_path("MODEL_DIR", DEFAULT_MODEL_DIR)


def _active_model_dir() -> Path:
    return resolve_model_dir(MODEL_DIR)


class PredictRequest(BaseModel):
    loan: LoanInput | None = None
    loans: list[LoanInput] | None = None
    model_name: Literal["xgb", "lr"] = "xgb"
    lgd: float = Field(default=0.45, ge=0, le=1)

    @model_validator(mode="after")
    def validate_loans_batch_size(self) -> PredictRequest:
        if self.loans is not None and len(self.loans) > MAX_BATCH_LOANS:
            raise ValueError(f"loans batch size cannot exceed {MAX_BATCH_LOANS}")
        return self


class PredictResponse(BaseModel):
    predictions: list[PredictionOutput]


class SessionRequest(BaseModel):
    api_key: str | None = None


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@app.post("/auth/token", response_model=TokenResponse)
def auth_token(body: TokenRequest) -> TokenResponse:
    """Demo OAuth2 token endpoint (password grant stub). Not a full OAuth2 provider."""
    if get_auth_mode() != "oauth2":
        raise HTTPException(status_code=400, detail="AUTH_MODE must be oauth2 to use /auth/token")
    current_key = API_KEY or get_api_key()
    if not current_key or not secrets.compare_digest(body.password, current_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    from app.api.oauth2 import jwt_ttl_seconds

    token = create_access_token(body.username)
    return TokenResponse(access_token=token, expires_in=jwt_ttl_seconds())


@app.post("/auth/session")
def auth_session(body: SessionRequest, request: Request, response: Response) -> dict:
    current_key = API_KEY or get_api_key()
    if not current_key:
        token = create_session_token()
        set_session_cookie(response, token)
        return {"authenticated": True, "expires_in": SESSION_TTL_SECONDS}

    provided_key = body.api_key or request.headers.get("X-API-Key")
    if provided_key is None or not secrets.compare_digest(provided_key, current_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    token = create_session_token()
    set_session_cookie(response, token)
    return {"authenticated": True, "expires_in": SESSION_TTL_SECONDS}


@app.post("/auth/logout")
def auth_logout(response: Response) -> dict:
    clear_session_cookie(response)
    return {"authenticated": False}


@app.get("/health")
def health() -> dict:
    active = _active_model_dir()
    loaded = models_exist(MODEL_DIR)
    metadata = {}
    if loaded:
        try:
            metadata = load_json(active / "training_metadata.json")
        except Exception:
            metadata = {}
    staged = [v for v in list_versions(MODEL_DIR) if v.get("status") == "staged"]
    return {
        "status": "ok",
        "model_loaded": loaded,
        "best_model": metadata.get("best_model"),
        "trained_at": metadata.get("trained_at"),
        "current_version_id": get_current_version(MODEL_DIR),
        "staged_versions": [v.get("version_id") for v in staged],
    }


@app.post("/train", response_model=TrainJobResponse, status_code=status.HTTP_202_ACCEPTED)
def train(
    background_tasks: BackgroundTasks,
    config: TrainConfig | None = None,
    _: None = Depends(verify_api_key),
) -> TrainJobResponse:
    job = create_job()
    background_tasks.add_task(run_training_job, job.job_id, config or TrainConfig())
    return TrainJobResponse(job_id=job.job_id, status=TrainJobStatus.pending)


@app.get("/train/status/{job_id}", response_model=TrainJobStatusResponse)
def train_status(job_id: str, _: None = Depends(verify_api_key)) -> TrainJobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )


@app.post("/models/promote/{version_id}")
def promote_model(version_id: str, force: bool = False, _: None = Depends(verify_api_key)) -> dict:
    try:
        promote_version(version_id, MODEL_DIR, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PromotionBlockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "promoted", "version_id": version_id}


@app.get("/monitoring/encoding")
def encoding_monitor(_: None = Depends(verify_api_key)) -> dict:
    return {"encoding_unknown_total": get_unknown_category_count()}


@app.get("/monitoring/drift")
def drift_monitor(_: None = Depends(verify_api_key)) -> dict:
    active = _active_model_dir()
    if not models_exist(MODEL_DIR):
        raise HTTPException(status_code=404, detail="No trained models found")
    metadata = load_json(active / "training_metadata.json")
    csv_rel = metadata.get("csv_path", CREDIT_CSV_PATH)
    csv_path = get_env_path("DATA_CSV_PATH", csv_rel)
    try:
        require_file(csv_path, label="Training reference CSV")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reference = pd.read_csv(csv_path)
    recent = recent_inputs(limit=500)
    if not recent:
        return {"features": {}, "message": "No recent predictions in audit store"}
    current = pd.DataFrame(recent)
    report = compute_psi_report(reference, current)
    return {"features": report}


@app.get("/monitoring/abuse")
def abuse_monitor(request: Request, _: None = Depends(verify_api_key)) -> dict:
    return get_abuse_flags(resolve_client_key(request))


@app.get("/predict/audit/{prediction_id}")
def predict_audit(prediction_id: str, _: None = Depends(verify_api_key)) -> dict:
    record = get_prediction_audit(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction audit record not found")
    return record


@app.get("/metrics")
def metrics(_: None = Depends(verify_api_key)) -> dict:
    active = _active_model_dir()
    metrics_path = active / "metrics.json"
    importance_path = active / "feature_importance.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    payload = {"metrics": load_json(metrics_path)}
    if importance_path.exists():
        payload["feature_importance"] = load_json(importance_path)
    return payload


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    http_request: Request,
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    enforce_rate_limit(http_request)
    if not models_exist(MODEL_DIR):
        raise HTTPException(status_code=400, detail="Models not trained. Call POST /train first.")

    user_id = resolve_authenticated_user(http_request)

    try:
        if request.loans:
            preds = predict_pd(
                request.loans,
                model_name=request.model_name,
                lgd=request.lgd,
                user_id=user_id,
            )
            if not isinstance(preds, list):
                preds = [preds]
        elif request.loan:
            pred = predict_pd(
                request.loan,
                model_name=request.model_name,
                lgd=request.lgd,
                user_id=user_id,
            )
            preds = [pred] if isinstance(pred, PredictionOutput) else pred
        else:
            raise HTTPException(status_code=422, detail="Provide 'loan' or 'loans'")

        client_key = resolve_client_key(http_request)
        for item in preds:
            record_predict(client_key, item.pd)

        return PredictResponse(predictions=preds)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stress_test", response_model=StressTestResponse)
def stress_test(request: StressTestRequest, _: None = Depends(verify_api_key)) -> StressTestResponse:
    from app.api.logging_config import get_logger

    stress_logger = get_logger(__name__)
    stress_logger.info(
        "stress_test incoming payload: sample_size=%s scenarios=%s model_name=%s loan_count=%s",
        request.sample_size,
        request.scenarios,
        request.model_name,
        len(request.loans) if request.loans else None,
    )
    if not models_exist(MODEL_DIR):
        raise HTTPException(status_code=400, detail="Models not trained. Call POST /train first.")
    try:
        response = run_portfolio_stress_test(request)
        stress_logger.info(
            "stress_test completed: loan_count=%s scenarios=%s",
            response.loan_count,
            [r.scenario for r in response.results],
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        stress_logger.exception("stress_test failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
