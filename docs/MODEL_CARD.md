# Model Card — Credit Risk PD Engine

## Model Overview

Dual-model probability-of-default (PD) engine with Logistic Regression and XGBoost classifiers, shared feature engineering, Platt/isotonic calibration, and IFRS 9 ECL outputs.

| Property | Value |
|----------|-------|
| Target | `loan_status` (0 = performing, 1 = default) |
| Models | Logistic Regression, XGBoost |
| Best model selection | Holdout ROC AUC |
| Calibration | Platt (LR), isotonic (XGB) on 15% calibration split |

## Training Data

- **Source:** `assets/data/credit_risk_dataset_new.csv` (public benchmark-style loan dataset)
- **Macro overlay:** `assets/data/US_Macro_Economic_Stress_Test_Data.xlsx` (synthetic stress scenarios)
- **No origination date** — temporal out-of-time (OOT) validation is **not** possible on this CSV without business data enrichment.

## Validation

- **Holdout:** stratified 80/20 split on raw rows before feature engineering (prevents median/quantile leakage).
- **Repeated stratified k-fold CV:** 5 folds × 3 repeats when train size ≥ 50; reported as mean ± std in `metrics.json`.
- **OOT proxy (optional):** set `OOT_VALIDATION=row_order_proxy` to hold out the last 20% of CSV rows as a temporal proxy. This is **not** a substitute for true origination-date OOT — see `# TODO(business-input)` in code.

## Limitations

1. **Benchmark data** — not representative of a live lending portfolio; expect operational degradation in production.
2. **No loan tenor field** — lifetime PD uses configurable `DEFAULT_LOAN_TENOR_MONTHS=36`.
3. **Simplified IFRS 9 staging** — educational MVP, not audit-ready.
4. **Governance buffers** — PD (+25 bps) and LGD (+5%) buffers applied post-calibration; tune via `PD_BUFFER_BPS` / `LGD_BUFFER_PCT`.
5. **Macro multipliers** — segment/scenario shocks include regression-assisted estimates; validate with business stress committees.

## Intended Use

- PD estimation for individual loans and portfolio stress scenarios
- Internal risk dashboards and methodology demonstrations

## Out of Scope

- Regulatory submission without independent model validation
- Adverse-action compliance without legal review of reason-code mappings

## Monitoring

- **Encoding monitor:** unseen categorical values (`GET /monitoring/encoding`)
- **PSI drift:** feature drift vs training reference (`GET /monitoring/drift`)
- **Abuse heuristics:** rate limits and boundary-probing detection (`GET /monitoring/abuse`)
- **Audit trail:** per-prediction SHAP + reason codes (`GET /predict/audit/{prediction_id}`)

## Versioning

Models are stored under `backend/models/registry/{version_id}/`. New training creates a **staged** version; promote via `POST /models/promote/{version_id}`.
