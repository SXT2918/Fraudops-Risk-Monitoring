import json

import numpy as np
import pandas as pd

from src.score import (
    align_feature_columns,
    load_feature_columns,
    load_model,
    probability_to_risk_tier,
    risk_tier_to_decision,
    score_transactions,
)
from src.train_model import train_models


class FakeFraudModel:
    def __init__(self) -> None:
        self.seen_columns: list[str] = []

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.seen_columns = list(features.columns)
        probabilities = np.array([0.80, 0.20])
        return np.column_stack([1 - probabilities, probabilities])


def test_probability_to_risk_tier():
    assert probability_to_risk_tier(0.10) == "Low"
    assert probability_to_risk_tier(0.30) == "Medium"
    assert probability_to_risk_tier(0.50) == "Medium"
    assert probability_to_risk_tier(0.70) == "High"
    assert probability_to_risk_tier(0.90) == "High"


def test_risk_tier_to_decision():
    assert risk_tier_to_decision("Low") == "Approve"
    assert risk_tier_to_decision("Medium") == "Monitor"
    assert risk_tier_to_decision("High") == "Manual Review"


def test_align_feature_columns_fills_missing_and_drops_extra():
    features = pd.DataFrame(
        {
            "amount": [10.0],
            "merchant_category_online": [1],
            "extra_new_category": [1],
        }
    )

    aligned = align_feature_columns(
        features,
        ["amount", "merchant_category_online", "missing_training_column"],
    )

    assert list(aligned.columns) == [
        "amount",
        "merchant_category_online",
        "missing_training_column",
    ]
    assert aligned.loc[0, "missing_training_column"] == 0


def test_score_transactions_uses_saved_feature_column_order(monkeypatch, tmp_path):
    fake_model = FakeFraudModel()
    monkeypatch.setattr("src.score.load_model", lambda _path: fake_model)

    feature_columns_path = tmp_path / "feature_columns.json"
    feature_columns = ["amount", "log_amount", "missing_training_column"]
    feature_columns_path.write_text(json.dumps(feature_columns), encoding="utf-8")

    transactions = pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2"],
            "user_id": ["user_1", "user_2"],
            "amount": [100.0, 12.0],
            "merchant_category": ["online_services", "grocery"],
            "transaction_time": ["2026-01-01 02:00:00", "2026-01-01 14:00:00"],
            "location": ["Toronto", "Vancouver"],
        }
    )

    scores = score_transactions(
        transactions,
        model_path=tmp_path / "unused_model.pkl",
        feature_columns_path=feature_columns_path,
    )

    assert fake_model.seen_columns == feature_columns
    assert list(scores.columns) == [
        "transaction_id",
        "fraud_probability",
        "risk_tier",
        "decision",
    ]
    assert scores.loc[0, "risk_tier"] == "High"
    assert scores.loc[0, "decision"] == "Manual Review"


def test_score_transactions_handles_realistic_transaction(tmp_path):
    clean_data_path = tmp_path / "clean_transactions.csv"
    model_path = tmp_path / "models" / "fraud_model.pkl"
    feature_columns_path = tmp_path / "models" / "feature_columns.json"
    metrics_path = tmp_path / "models" / "model_metrics.json"

    rows = []
    for index in range(12):
        is_fraud = int(index % 2 == 0)
        rows.append(
            {
                "transaction_id": f"txn_{index:03d}",
                "user_id": f"user_{index // 2:03d}",
                "amount": 0.75 if is_fraud else 42.0 + index,
                "merchant_category": "online_services" if is_fraud else "grocery",
                "transaction_time": (
                    f"2026-01-{(index % 9) + 1:02d} "
                    f"{'03' if is_fraud else '14'}:00:00"
                ),
                "location": "Toronto" if is_fraud else "Vancouver",
                "is_fraud": is_fraud,
            }
        )
    pd.DataFrame(rows).to_csv(clean_data_path, index=False)
    train_models(
        clean_data_path=clean_data_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        metrics_path=metrics_path,
        include_xgboost=False,
    )

    transaction = pd.DataFrame(
        {
            "transaction_id": ["txn_live"],
            "user_id": ["user_live"],
            "amount": [0.82],
            "merchant_category": ["online_services"],
            "transaction_time": ["2026-01-15T03:24:00"],
            "location": ["Toronto"],
        }
    )

    scores = score_transactions(
        transaction,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
    )

    assert scores.loc[0, "transaction_id"] == "txn_live"
    assert 0 <= scores.loc[0, "fraud_probability"] <= 1
    assert scores.loc[0, "risk_tier"] in {"Low", "Medium", "High"}
    assert scores.loc[0, "decision"] in {"Approve", "Monitor", "Manual Review"}


def test_missing_model_artifact_message_is_beginner_friendly(tmp_path):
    missing_model_path = tmp_path / "missing_model.pkl"

    try:
        load_model(missing_model_path)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected load_model to raise FileNotFoundError")

    assert "python -m src.ingestion" in message
    assert "python -m src.train_model" in message


def test_missing_feature_columns_message_is_beginner_friendly(tmp_path):
    missing_columns_path = tmp_path / "missing_feature_columns.json"

    try:
        load_feature_columns(missing_columns_path)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected load_feature_columns to raise FileNotFoundError")

    assert "python -m src.ingestion" in message
    assert "python -m src.train_model" in message
