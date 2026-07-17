from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.config.constants import MAX_BATCH_LOANS


class LoanInput(BaseModel):
    person_age: float = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(..., ge=0, le=50)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float = Field(..., ge=0)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class PredictionOutput(BaseModel):
    pd: float
    risk_score: int
    risk_band: str
    predicted_class: int
    expected_loss: float
    ifrs9_stage: int
    ecl: float
    model_name: str
    calibrated_pd: float | None = None
    optimal_threshold_class: int | None = None
    prediction_id: str | None = None
    shap_contributions: list[FeatureContribution] | None = None
    reason_codes: list[str] | None = None


class TrainConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    primary_model: Literal["xgb", "lr"] = "xgb"


class ModelMetrics(BaseModel):
    model_name: str
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    ks: float
    gini: float


class TrainResponse(BaseModel):
    best_model: str
    metrics: dict[str, ModelMetrics]
    feature_importance: dict[str, float]
    training_metadata: dict[str, Any]


class TrainJobStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class TrainJobResponse(BaseModel):
    job_id: str
    status: TrainJobStatus


class TrainJobStatusResponse(BaseModel):
    job_id: str
    status: TrainJobStatus
    result: TrainResponse | None = None
    error: str | None = None


class MacroVariable(BaseModel):
    variable: str
    base_value: float
    stressed_value: float
    pd_multiplier: float


class StressScenario(BaseModel):
    name: str
    variables: list[MacroVariable]
    portfolio_pd_multiplier: float


class StressTestRequest(BaseModel):
    loans: list[LoanInput] | None = None
    sample_size: int | None = Field(default=500, ge=1, le=MAX_BATCH_LOANS)
    scenarios: list[str] = Field(default_factory=lambda: ["Normal", "Boom", "Recession"])
    model_name: Literal["xgb", "lr"] = "xgb"
    lgd: float = Field(default=0.45, ge=0, le=1)

    @model_validator(mode="after")
    def validate_loans_batch_size(self) -> StressTestRequest:
        if self.loans is not None and len(self.loans) > MAX_BATCH_LOANS:
            raise ValueError(f"loans batch size cannot exceed {MAX_BATCH_LOANS}")
        return self


class PortfolioStressResult(BaseModel):
    scenario: str
    avg_pd: float
    total_el: float
    total_ecl: float
    stage_1_count: int
    stage_2_count: int
    stage_3_count: int
    stage_1_ecl: float
    stage_2_ecl: float
    stage_3_ecl: float
    pd_multiplier: float


class ScenarioComparison(BaseModel):
    scenario: str
    avg_pd: float
    total_ecl: float
    delta_ecl_vs_normal: float | None = None


class StressTestResponse(BaseModel):
    results: list[PortfolioStressResult]
    comparison: list[ScenarioComparison]
    loan_count: int
    probability_weighted_ecl: float | None = None
