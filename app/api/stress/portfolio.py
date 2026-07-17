"""Portfolio-level stress testing and ECL aggregation."""

from __future__ import annotations

import numpy as np
from src.config.constants import MAX_BATCH_LOANS
from src.data.credit_loader import sample_portfolio
from src.data.macro_loader import load_macro_scenarios
from src.models.risk_scoring import compute_ecl, ifrs9_stage
from src.models.schemas import (
    LoanInput,
    PortfolioStressResult,
    ScenarioComparison,
    StressTestRequest,
    StressTestResponse,
)
from src.pipelines.predict import predict_pd

from app.api.stress.config_loader import load_json_config
from app.api.stress.scenarios import apply_pd_shock

# TODO(business-input): confirm scenario probability weights with risk committee.
DEFAULT_SCENARIO_WEIGHTS = {"Normal": 0.6, "Boom": 0.15, "Recession": 0.25}


def _load_scenario_weights() -> dict[str, float]:
    return load_json_config(
        "scenario_weights.json",
        DEFAULT_SCENARIO_WEIGHTS,
        "scenario probability weights",
    )


def _loans_to_inputs(loans_df) -> list[LoanInput]:
    drop_cols = [c for c in ["loan_status"] if c in loans_df.columns]
    records = loans_df.drop(columns=drop_cols).to_dict(orient="records")
    return [LoanInput(**r) for r in records]


def run_portfolio_stress_test(request: StressTestRequest) -> StressTestResponse:
    scenarios_map = load_macro_scenarios()
    selected = [s for s in request.scenarios if s in scenarios_map]
    if not selected:
        raise ValueError(f"No valid scenarios found. Available: {list(scenarios_map)}")

    if request.loans:
        if len(request.loans) > MAX_BATCH_LOANS:
            raise ValueError(f"loans batch size cannot exceed {MAX_BATCH_LOANS}")
        loans_df = __import__("pandas").DataFrame([loan.model_dump() for loan in request.loans])
    else:
        loans_df = sample_portfolio(n=request.sample_size or 500)

    loan_inputs = _loans_to_inputs(loans_df)
    raw_predictions = predict_pd(loan_inputs, model_name=request.model_name, lgd=request.lgd)
    predictions = raw_predictions if isinstance(raw_predictions, list) else [raw_predictions]
    base_pds = np.array([p.pd for p in predictions])
    eads = loans_df["loan_amnt"].values.astype(float)
    intents = loans_df["loan_intent"].astype(str).tolist()
    grades = loans_df["loan_grade"].astype(str).tolist()
    rates = loans_df["loan_int_rate"].values.astype(float)

    results: list[PortfolioStressResult] = []
    normal_ecl: float | None = None
    scenario_weights = _load_scenario_weights()

    for scenario_name in selected:
        scenario = scenarios_map[scenario_name]
        stressed_pds = np.asarray(apply_pd_shock(base_pds, scenario, loan_intents=intents), dtype=float)

        stages = [ifrs9_stage(float(pd)) for pd in stressed_pds]
        ecls = [
            compute_ecl(
                float(pd),
                float(ead),
                stage,
                request.lgd,
                loan_int_rate=float(rate),
                loan_intent=intent,
                loan_grade=grade,
                stressed=scenario_name != "Normal",
            )
            for pd, ead, stage, rate, intent, grade in zip(
                stressed_pds, eads, stages, rates, intents, grades, strict=False
            )
        ]
        els = stressed_pds * request.lgd * eads

        stage_counts = {1: 0, 2: 0, 3: 0}
        stage_ecl = {1: 0.0, 2: 0.0, 3: 0.0}
        for stage, ecl in zip(stages, ecls, strict=False):
            stage_counts[stage] += 1
            stage_ecl[stage] += ecl

        total_ecl = float(sum(ecls))
        if scenario_name == "Normal":
            normal_ecl = total_ecl

        results.append(
            PortfolioStressResult(
                scenario=scenario_name,
                avg_pd=float(np.mean(stressed_pds)),
                total_el=float(np.sum(els)),
                total_ecl=total_ecl,
                stage_1_count=stage_counts[1],
                stage_2_count=stage_counts[2],
                stage_3_count=stage_counts[3],
                stage_1_ecl=stage_ecl[1],
                stage_2_ecl=stage_ecl[2],
                stage_3_ecl=stage_ecl[3],
                pd_multiplier=scenario.portfolio_pd_multiplier,
            )
        )

    weighted_ecl = sum(r.total_ecl * scenario_weights.get(r.scenario, 0.0) for r in results) / max(
        sum(scenario_weights.get(r.scenario, 0.0) for r in results), 1e-9
    )

    comparison = [
        ScenarioComparison(
            scenario=r.scenario,
            avg_pd=r.avg_pd,
            total_ecl=r.total_ecl,
            delta_ecl_vs_normal=(
                (r.total_ecl - normal_ecl)
                if normal_ecl is not None and r.scenario != "Normal"
                else 0.0
                if r.scenario == "Normal"
                else None
            ),
        )
        for r in results
    ]

    return StressTestResponse(
        results=results,
        comparison=comparison,
        loan_count=len(loan_inputs),
        probability_weighted_ecl=weighted_ecl,
    )
