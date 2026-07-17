"""Streamlit UI for Credit Risk PD & Stress Testing Engine."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from src.utils.display import band_style, format_currency, format_pd

import streamlit as st
from app.api.settings import get_api_key, load_env

logger = logging.getLogger(__name__)

load_env()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

INTERNAL_WARNING = (
    "Internal tool only: Streamlit routes through FastAPI but must run on a trusted "
    "internal network. Never expose publicly."
)


def _resolve_api_key() -> str:
    """Read API_KEY from the same .env the backend uses."""
    return get_api_key() or ""


@st.cache_resource
def _get_authenticated_session() -> tuple[requests.Session, str]:
    """Create a persistent requests.Session with backend session cookie + API key."""
    api_key = _resolve_api_key()
    session = requests.Session()

    if not api_key:
        logger.warning("API_KEY not found in .env; protected backend routes will return 401")
        return session, api_key

    headers = {"X-API-Key": api_key}
    resp = session.post(
        f"{API_URL}/auth/session",
        json={"api_key": api_key},
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        logger.error("Session establishment failed: %s %s", resp.status_code, resp.text)
        resp.raise_for_status()

    logger.info("Authenticated with backend via /auth/session")
    return session, api_key


def _api_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _api_post(path: str, payload: dict | None = None):
    session, api_key = _get_authenticated_session()
    url = f"{API_URL}{path}"
    resp = session.post(url, json=payload or {}, headers=_api_headers(api_key), timeout=120)
    if resp.status_code == 401:
        raise RuntimeError(
            "401 Unauthorized — verify API_KEY in .env matches the backend "
            f"(current key length: {len(api_key)}). Restart Streamlit after changing .env."
        )
    resp.raise_for_status()
    return resp.json()


def _api_get(path: str):
    session, api_key = _get_authenticated_session()
    url = f"{API_URL}{path}"
    resp = session.get(url, headers=_api_headers(api_key), timeout=30)
    if resp.status_code == 401:
        raise RuntimeError(
            f"401 Unauthorized — verify API_KEY in .env matches the backend (current key length: {len(api_key)})."
        )
    resp.raise_for_status()
    return resp.json()


def _poll_train_result() -> dict:
    start = _api_post("/train", {})
    job_id = start["job_id"]
    while True:
        status = _api_get(f"/train/status/{job_id}")
        if status["status"] == "completed":
            return status["result"]
        if status["status"] == "failed":
            raise RuntimeError(status.get("error") or "Training failed")
        time.sleep(1)


def _stress_test_post(payload: dict) -> dict:
    """POST /stress_test with extended timeout and debug logging (stress dashboard only)."""
    session, api_key = _get_authenticated_session()
    url = f"{API_URL}/stress_test"
    logger.info("[stress_test] POST %s", url)
    logger.info("[stress_test] request payload: %s", json.dumps(payload))
    resp = session.post(
        url,
        json=payload,
        headers=_api_headers(api_key),
        timeout=600,
    )
    logger.info("[stress_test] response status=%s", resp.status_code)
    logger.info("[stress_test] response body (first 2000 chars): %s", resp.text[:2000])
    if resp.status_code == 401:
        raise RuntimeError(
            "401 Unauthorized — verify API_KEY in .env matches the backend "
            f"(current key length: {len(api_key)}). Restart Streamlit after changing .env."
        )
    if not resp.ok:
        detail = resp.text
        try:
            parsed = resp.json()
            detail = parsed.get("detail", detail)
        except (json.JSONDecodeError, ValueError, AttributeError):
            logger.exception("[stress_test] failed to parse error response JSON")
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
    result = resp.json()
    logger.info(
        "[stress_test] loan_count=%s result_scenarios=%s",
        result.get("loan_count"),
        [row.get("scenario") for row in result.get("results", [])],
    )
    return result


def _show_stress_test_error(exc: BaseException) -> None:
    """Display stress test failure with full traceback (stress dashboard only)."""
    logger.exception("[stress_test] dashboard run failed: %s", exc)
    st.error(f"Stress test failed: {type(exc).__name__}: {exc}")
    with st.expander("Debug traceback", expanded=True):
        st.code(traceback.format_exc())


logger.warning(INTERNAL_WARNING)

st.set_page_config(page_title="Credit Risk PD Engine", layout="wide")
st.warning(INTERNAL_WARNING)

_api_key = _resolve_api_key()
if not _api_key:
    st.error(
        "API_KEY is not configured. Copy `.env.example` to `.env` and set "
        "`API_KEY=dev-local-api-key` (must match the backend), then restart Streamlit."
    )
else:
    try:
        _get_authenticated_session()
    except Exception as exc:
        st.error(f"Backend authentication failed: {exc}")

st.title("Credit Risk PD & Stress Testing Engine")
st.caption("Streamlit alternative UI — connects to FastAPI backend")

page = st.sidebar.radio("Navigation", ["Train", "Single Predict", "Batch Predict", "Stress Dashboard"])


if page == "Train":
    st.header("Train Models")
    if st.button("Run Training", type="primary"):
        with st.spinner("Training LR and XGBoost..."):
            try:
                result = _poll_train_result()
                st.success(f"Best model: {result['best_model']}")
                for name, m in result["metrics"].items():
                    st.subheader(name.upper())
                    st.json(m)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

elif page == "Single Predict":
    st.header("Loan PD Calculator")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Person Age", 18, 100, 35)
        person_income = st.number_input("Person Income", 1000, 1000000, 60000)
        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
        emp_length = st.number_input("Employment Length (years)", 0.0, 50.0, 5.0)
        loan_intent = st.selectbox(
            "Loan Intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
    with col2:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Loan Amount", 500, 500000, 10000)
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 12.0)
        loan_pct_income = st.number_input("Loan % Income", 0.0, 1.0, 0.3)
        default_file = st.selectbox("Prior Default on File", ["Y", "N"])
        cred_hist = st.number_input("Credit History Length", 0, 30, 3)

    model_name = st.selectbox("Model", ["xgb", "lr"])
    if st.button("Calculate PD", type="primary"):
        payload = {
            "loan": {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": home,
                "person_emp_length": emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_pct_income,
                "cb_person_default_on_file": default_file,
                "cb_person_cred_hist_length": cred_hist,
            },
            "model_name": model_name,
        }
        try:
            result = _api_post("/predict", payload)
            pred = result["predictions"][0]
            style = band_style(pred["risk_band"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PD", format_pd(pred["pd"]))
            c2.metric("Risk Score", pred["risk_score"])
            c3.metric("Risk Band", style["label"])
            c4.metric("ECL", format_currency(pred["ecl"]))
            st.markdown(
                f"<span style='color:{style['color']};font-weight:bold'>{style['label']} risk band</span>",
                unsafe_allow_html=True,
            )
            if pred.get("reason_codes"):
                st.write("Reason codes:", ", ".join(pred["reason_codes"]))
            st.json(pred)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Batch Predict":
    st.header("Batch Predict")
    uploaded = st.file_uploader("Upload CSV with loan columns", type=["csv"])
    model_name = st.selectbox("Model", ["xgb", "lr"], key="batch_model")
    if uploaded and st.button("Run Batch Predict", type="primary"):
        df = pd.read_csv(uploaded)
        loans = df.to_dict(orient="records")
        try:
            result = _api_post("/predict", {"loans": loans, "model_name": model_name})
            out = pd.DataFrame(result["predictions"])
            st.dataframe(out)
            st.download_button("Download Results", out.to_csv(index=False), "predictions.csv")
        except Exception as exc:
            st.error(f"Batch predict failed: {exc}")

else:
    st.header("Stress Test Dashboard")
    sample_size = st.slider("Portfolio Sample Size", 100, 2000, 500)
    scenarios = st.multiselect("Scenarios", ["Normal", "Boom", "Recession"], default=["Normal", "Boom", "Recession"])
    model_name = st.selectbox("Model", ["xgb", "lr"], key="stress_model")

    if st.button("Run Stress Test", type="primary"):
        if not scenarios:
            st.error("Select at least one scenario (Normal, Boom, or Recession).")
        else:
            payload = {"sample_size": sample_size, "scenarios": scenarios, "model_name": model_name}
            try:
                with st.spinner("Running portfolio stress test..."):
                    result = _stress_test_post(payload)

                if not result.get("results"):
                    st.warning("Stress test returned no scenario results.")
                else:
                    st.metric("Loans Tested", result["loan_count"])

                    comp_df = pd.DataFrame(result["comparison"])
                    st.subheader("Scenario Comparison")
                    st.dataframe(comp_df)

                    results_df = pd.DataFrame(result["results"])
                    st.subheader("Portfolio Results")
                    st.bar_chart(results_df.set_index("scenario")[["avg_pd", "total_ecl"]])

                    st.subheader("IFRS 9 Stage Breakdown")
                    stage_cols = ["stage_1_count", "stage_2_count", "stage_3_count"]
                    st.bar_chart(results_df.set_index("scenario")[stage_cols])
            except Exception as exc:
                _show_stress_test_error(exc)
