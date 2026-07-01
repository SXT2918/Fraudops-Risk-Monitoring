import json

import pandas as pd

from dashboard import app as dashboard_app


def clear_dashboard_caches() -> None:
    for loader in [
        dashboard_app.load_clean_transactions,
        dashboard_app.load_model_metrics,
        dashboard_app.load_scored_transaction_data,
    ]:
        if hasattr(loader, "clear"):
            loader.clear()


def test_dashboard_load_clean_transactions_missing_file(monkeypatch, tmp_path):
    clear_dashboard_caches()
    monkeypatch.setattr(dashboard_app, "CLEAN_TRANSACTIONS_FILE", tmp_path / "missing.csv")

    transactions = dashboard_app.load_clean_transactions()

    assert transactions.empty


def test_dashboard_load_model_metrics_missing_file(monkeypatch, tmp_path):
    clear_dashboard_caches()
    monkeypatch.setattr(dashboard_app, "MODEL_METRICS_PATH", tmp_path / "missing_metrics.json")

    metrics = dashboard_app.load_model_metrics()

    assert metrics == {}


def test_dashboard_load_model_metrics_valid_file(monkeypatch, tmp_path):
    clear_dashboard_caches()
    metrics_path = tmp_path / "model_metrics.json"
    metrics_path.write_text(json.dumps({"best_model": "logistic_regression"}), encoding="utf-8")
    monkeypatch.setattr(dashboard_app, "MODEL_METRICS_PATH", metrics_path)

    metrics = dashboard_app.load_model_metrics()

    assert metrics["best_model"] == "logistic_regression"


def test_dashboard_load_scored_transaction_data(monkeypatch):
    clear_dashboard_caches()
    scored = pd.DataFrame(
        {
            "transaction_id": ["txn_1"],
            "fraud_probability": ["0.91"],
            "risk_tier": ["High"],
            "decision": ["Manual Review"],
            "scored_at": ["2026-01-01T12:00:00+00:00"],
            "transaction_time": ["2026-01-01 03:00:00"],
            "amount": ["0.75"],
        }
    )
    monkeypatch.setattr(dashboard_app, "fetch_scored_transactions", lambda: scored)

    loaded = dashboard_app.load_scored_transaction_data()

    assert len(loaded) == 1
    assert loaded.loc[0, "fraud_probability"] == 0.91
    assert pd.api.types.is_datetime64_any_dtype(loaded["transaction_time"])
