"""Streamlit dashboard for the FraudOps transaction risk platform."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLEAN_TRANSACTIONS_FILE, MODEL_METRICS_PATH
from src.database import fetch_scored_transactions
from src.evaluate_model import evaluate_saved_model
from src.features import add_amount_bucket


SETUP_COMMANDS = """
```powershell
python -m src.ingestion
python -m src.train_model
python -m src.evaluate_model
python -m src.seed_scores
uvicorn api.main:app --reload
```
"""


def format_currency(value: float) -> str:
    """Format a numeric value as a dollar amount."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""
    return f"{value * 100:.1f}%"


@st.cache_data
def load_clean_transactions() -> pd.DataFrame:
    """Load cleaned transaction data from the ingestion pipeline."""
    if not CLEAN_TRANSACTIONS_FILE.exists():
        return pd.DataFrame()

    transactions = pd.read_csv(CLEAN_TRANSACTIONS_FILE)
    if "transaction_time" in transactions.columns:
        transactions["transaction_time"] = pd.to_datetime(
            transactions["transaction_time"],
            errors="coerce",
        )
    if "amount" in transactions.columns:
        transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0)
    if "is_fraud" in transactions.columns:
        transactions["is_fraud"] = transactions["is_fraud"].astype(int)
    return transactions


@st.cache_data
def load_model_metrics() -> dict[str, Any]:
    """Load saved model metrics from JSON when available."""
    if not MODEL_METRICS_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data
def load_scored_transaction_data() -> pd.DataFrame:
    """Load scored transactions from SQLite."""
    try:
        scored = fetch_scored_transactions()
    except Exception:
        return pd.DataFrame()

    for col in ["scored_at", "transaction_time"]:
        if col in scored.columns:
            scored[col] = pd.to_datetime(scored[col], errors="coerce")
    if "fraud_probability" in scored.columns:
        scored["fraud_probability"] = pd.to_numeric(
            scored["fraud_probability"],
            errors="coerce",
        )
    if "amount" in scored.columns:
        scored["amount"] = pd.to_numeric(scored["amount"], errors="coerce")
    return scored


@st.cache_data
def load_threshold_table() -> pd.DataFrame:
    """Recreate threshold comparison data from saved model artifacts."""
    try:
        _summary, threshold_table = evaluate_saved_model()
    except Exception:
        return pd.DataFrame()
    return threshold_table


def show_setup_instructions() -> None:
    """Show setup commands when dashboard inputs are missing."""
    st.info("Run the project pipeline first, then refresh this dashboard.")
    st.markdown(SETUP_COMMANDS)


def metric_grid(transactions: pd.DataFrame, scored: pd.DataFrame) -> None:
    """Render top-level KPI metrics."""
    total_transactions = len(transactions)
    fraud_cases = int(transactions["is_fraud"].sum()) if "is_fraud" in transactions else 0
    fraud_rate = fraud_cases / max(total_transactions, 1)
    total_volume = float(transactions["amount"].sum()) if "amount" in transactions else 0.0
    average_amount = float(transactions["amount"].mean()) if "amount" in transactions else 0.0

    high_risk_scored = 0
    if not scored.empty and "risk_tier" in scored:
        high_risk_scored = int((scored["risk_tier"] == "High").sum())

    first_row = st.columns(5)
    first_row[0].metric("Total transactions", f"{total_transactions:,}")
    first_row[1].metric("Fraud rate", format_percent(fraud_rate))
    first_row[2].metric("Fraud cases", f"{fraud_cases:,}")
    first_row[3].metric("Transaction volume", format_currency(total_volume))
    first_row[4].metric("Average amount", format_currency(average_amount))

    second_row = st.columns(2)
    second_row[0].metric("Scored transactions", f"{len(scored):,}")
    second_row[1].metric("High-risk scored", f"{high_risk_scored:,}")


def render_overview(transactions: pd.DataFrame, scored: pd.DataFrame) -> None:
    """Render the dashboard overview tab."""
    st.subheader("Portfolio Health Snapshot")
    if transactions.empty:
        st.warning(f"Clean transaction data was not found at `{CLEAN_TRANSACTIONS_FILE}`.")
        show_setup_instructions()
        return

    metric_grid(transactions, scored)
    st.markdown(
        """
        This view summarizes transaction activity, observed fraud labels, and scored risk
        decisions. It is meant to help a risk or analytics team quickly understand how
        much activity is flowing through the pipeline, where suspicious activity appears,
        and whether the scoring layer is producing review-ready decisions.
        """
    )

    if not scored.empty:
        risk_counts = (
            scored["risk_tier"].value_counts().rename_axis("risk_tier").reset_index(name="count")
        )
        fig = px.bar(
            risk_counts,
            x="risk_tier",
            y="count",
            color="risk_tier",
            title="Scored Transactions by Risk Tier",
            color_discrete_map={"Low": "#2ca02c", "Medium": "#ffbf00", "High": "#d62728"},
        )
        st.plotly_chart(fig, width="stretch")


def filter_scored_transactions(scored: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar-style filters for scored transactions."""
    filtered = scored.copy()
    with st.container(border=True):
        filter_cols = st.columns(3)
        risk_options = sorted(filtered["risk_tier"].dropna().unique().tolist())
        decision_options = sorted(filtered["decision"].dropna().unique().tolist())

        selected_risks = filter_cols[0].multiselect(
            "Risk tier",
            risk_options,
            default=risk_options,
        )
        selected_decisions = filter_cols[1].multiselect(
            "Decision",
            decision_options,
            default=decision_options,
        )

        if "scored_at" in filtered and filtered["scored_at"].notna().any():
            min_date = filtered["scored_at"].min().date()
            max_date = filtered["scored_at"].max().date()
            selected_dates = filter_cols[2].date_input(
                "Scored date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
                scored_dates = filtered["scored_at"].dt.date
                filtered = filtered[(scored_dates >= start_date) & (scored_dates <= end_date)]

        filtered = filtered[filtered["risk_tier"].isin(selected_risks)]
        filtered = filtered[filtered["decision"].isin(selected_decisions)]
    return filtered


def render_risk_monitoring(scored: pd.DataFrame) -> None:
    """Render scored-transaction monitoring."""
    st.subheader("Scored Transaction Queue")
    if scored.empty:
        st.info("No scored transactions are available yet.")
        st.markdown(
            """
            Generate scored rows by running the seed command, or run the API and
            call `/score_transaction` or `/score_batch`.

            ```powershell
            python -m src.seed_scores
            uvicorn api.main:app --reload
            ```
            """
        )
        return

    filtered = filter_scored_transactions(scored)
    high_risk = filtered[
        (filtered["risk_tier"] == "High") | (filtered["decision"] == "Manual Review")
    ]

    st.metric("Filtered scored transactions", f"{len(filtered):,}")
    if not high_risk.empty:
        st.error(f"{len(high_risk):,} filtered transaction(s) require manual review.")
        st.dataframe(
            high_risk[
                [
                    "transaction_id",
                    "fraud_probability",
                    "risk_tier",
                    "decision",
                    "scored_at",
                    "amount",
                    "merchant_category",
                    "location",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.dataframe(filtered, width="stretch", hide_index=True)


def fraud_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate transaction count and fraud rate by a grouping column."""
    return (
        df.groupby(group_col, dropna=False)
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_rate=("is_fraud", "mean"),
        )
        .reset_index()
    )


def render_fraud_pattern_analysis(transactions: pd.DataFrame) -> None:
    """Render charts describing fraud patterns in cleaned data."""
    st.subheader("Fraud Pattern Analysis")
    if transactions.empty:
        st.warning("Clean transaction data is missing.")
        show_setup_instructions()
        return

    analysis = transactions.copy()
    analysis["transaction_hour"] = analysis["transaction_time"].dt.hour
    analysis = add_amount_bucket(analysis)

    hourly_fraud = fraud_rate_by_group(analysis, "transaction_hour")
    hourly_count = analysis.groupby("transaction_hour").size().reset_index(name="transaction_count")
    amount_bucket_fraud = fraud_rate_by_group(analysis, "amount_bucket")

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(
        px.line(
            hourly_fraud,
            x="transaction_hour",
            y="fraud_rate",
            markers=True,
            title="Fraud Rate by Hour of Day",
        ),
        width="stretch",
    )
    chart_cols[1].plotly_chart(
        px.bar(
            hourly_count,
            x="transaction_hour",
            y="transaction_count",
            title="Transaction Count by Hour",
        ),
        width="stretch",
    )
    st.caption(
        "Hourly views help risk teams spot suspicious timing patterns, such as unusual late-night activity."
    )

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(
        px.bar(
            amount_bucket_fraud,
            x="amount_bucket",
            y="fraud_rate",
            title="Fraud Rate by Amount Bucket",
        ),
        width="stretch",
    )

    if "merchant_category" in analysis:
        merchant_fraud = fraud_rate_by_group(analysis, "merchant_category")
        merchant_fraud = merchant_fraud.sort_values(
            ["fraud_rate", "transaction_count"],
            ascending=[False, False],
        ).head(10)
        chart_cols[1].plotly_chart(
            px.bar(
                merchant_fraud,
                x="merchant_category",
                y="fraud_rate",
                color="transaction_count",
                title="Top Merchant Categories by Fraud Rate",
            ),
            width="stretch",
        )

    st.caption(
        "Amount and merchant views make it easier to separate routine activity from categories that merit tighter monitoring."
    )

    fig = px.histogram(
        analysis,
        x="amount",
        color=analysis["is_fraud"].map({0: "Not fraud", 1: "Fraud"}),
        nbins=20,
        title="Amount Distribution by Fraud Status",
        labels={"color": "Fraud status"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Distribution charts are useful for spotting whether fraud appears in micro-charges, larger purchases, or both."
    )


def best_model_metrics(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the best model name and its metric dictionary."""
    best_model = metrics.get("best_model", "")
    model_metrics = metrics.get("models", {}).get(best_model, {})
    return best_model, model_metrics


def render_model_performance(metrics: dict[str, Any]) -> None:
    """Render model metrics and threshold comparison details."""
    st.subheader("Model Performance")
    st.warning(
        "Current metrics are based on starter/sample data and are intended to validate "
        "the pipeline, not claim production-grade performance."
    )

    if not metrics:
        st.info(f"Model metrics were not found at `{MODEL_METRICS_PATH}`.")
        show_setup_instructions()
        return

    best_model, model_metrics = best_model_metrics(metrics)
    st.metric("Best model", best_model or "Unknown")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Precision", f"{model_metrics.get('precision', 0):.4f}")
    metric_cols[1].metric("Recall", f"{model_metrics.get('recall', 0):.4f}")
    metric_cols[2].metric("F1", f"{model_metrics.get('f1', 0):.4f}")
    metric_cols[3].metric("ROC-AUC", f"{model_metrics.get('roc_auc', 0):.4f}")
    metric_cols[4].metric("PR-AUC", f"{model_metrics.get('pr_auc', 0):.4f}")

    confusion = model_metrics.get("confusion_matrix", {})
    if confusion:
        st.markdown("**Confusion matrix at evaluation threshold**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "true_negatives": confusion.get("true_negatives", 0),
                        "false_positives": confusion.get("false_positives", 0),
                        "false_negatives": confusion.get("false_negatives", 0),
                        "true_positives": confusion.get("true_positives", 0),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        """
        In fraud detection, **precision** answers: when the model flags a transaction,
        how often is it truly fraud? **Recall** answers: of all fraud cases, how many
        did the model catch? A risk team often accepts more false positives when the
        cost of missing fraud is high, but the right threshold depends on review capacity.
        """
    )

    threshold_table = load_threshold_table()
    if not threshold_table.empty:
        st.markdown("**Threshold comparison**")
        st.dataframe(threshold_table, width="stretch", hide_index=True)
    else:
        st.info("Threshold comparison is unavailable until model artifacts can be loaded.")

    with st.expander("Raw model metrics JSON"):
        st.json(metrics)


def render_about() -> None:
    """Render project purpose, architecture, and next improvements."""
    st.subheader("About FraudOps")
    st.markdown(
        """
        FraudOps is a production-style portfolio project for data analyst, risk analytics,
        data engineering, analytics engineering, and ML-adjacent internship applications.
        It shows an end-to-end transaction monitoring workflow rather than a notebook-only model.

        **Tech stack**
        Python, pandas, scikit-learn, XGBoost, SQLite, FastAPI, Streamlit, Plotly, pytest.

        **Architecture**
        CSV ingestion -> validation -> SQLite -> feature engineering -> model training
        -> FastAPI scoring -> Streamlit dashboard.

        **Suggested next improvements**
        - PostgreSQL
        - Docker Compose service for API + dashboard
        - larger public dataset
        - model drift monitoring
        - authentication
        - cloud deployment
        """
    )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="FraudOps — Transaction Risk Monitoring Platform",
        page_icon="FO",
        layout="wide",
    )

    st.title("FraudOps — Transaction Risk Monitoring Platform")
    st.caption(
        "Monitor fraud patterns, scored transaction risk, model performance, and business impact."
    )

    transactions = load_clean_transactions()
    metrics = load_model_metrics()
    scored = load_scored_transaction_data()

    tabs = st.tabs(
        [
            "Overview",
            "Risk Monitoring",
            "Fraud Pattern Analysis",
            "Model Performance",
            "About",
        ]
    )
    with tabs[0]:
        render_overview(transactions, scored)
    with tabs[1]:
        render_risk_monitoring(scored)
    with tabs[2]:
        render_fraud_pattern_analysis(transactions)
    with tabs[3]:
        render_model_performance(metrics)
    with tabs[4]:
        render_about()


if __name__ == "__main__":
    main()
