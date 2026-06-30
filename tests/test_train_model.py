import json

import joblib
import pandas as pd

from src.train_model import train_models


def test_train_models_writes_model_artifacts(tmp_path):
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
                "amount": 0.75 if is_fraud else 35.0 + index,
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

    metrics_payload = train_models(
        clean_data_path=clean_data_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        metrics_path=metrics_path,
        include_xgboost=False,
    )

    assert model_path.exists()
    assert feature_columns_path.exists()
    assert metrics_path.exists()
    assert metrics_payload["best_model"] in {"logistic_regression", "random_forest"}

    model = joblib.load(model_path)
    feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert hasattr(model, "predict_proba")
    assert len(feature_columns) > 0
    assert saved_metrics["feature_count"] == len(feature_columns)
