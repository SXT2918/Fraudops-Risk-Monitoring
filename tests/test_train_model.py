import json

import joblib
import pandas as pd

from src.train_model import select_best_model, split_raw_transactions, train_models


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
    assert "merchant_risk_rate" not in feature_columns
    assert "user_transaction_count" not in feature_columns
    assert saved_metrics["validation_strategy"]["final_evaluation"].startswith("One untouched")
    assert saved_metrics["selection_metrics"]
    assert saved_metrics["test_metrics"]["pr_auc"] >= 0


def test_split_raw_transactions_uses_later_rows_as_holdout_when_feasible():
    rows = []
    for index in range(20):
        rows.append(
            {
                "transaction_id": f"txn_{index:03d}",
                "transaction_time": f"2026-01-{index + 1:02d} 12:00:00",
                "is_fraud": index % 2,
            }
        )
    transactions = pd.DataFrame(rows)

    train, test, split_info = split_raw_transactions(transactions, test_size=0.25)

    assert split_info["strategy"] == "later-time holdout"
    assert pd.to_datetime(train["transaction_time"]).max() < pd.to_datetime(
        test["transaction_time"]
    ).min()


def test_select_best_model_uses_cross_validation_not_holdout_results():
    selection_metrics = {
        "cv_winner": {
            "cv_pr_auc_mean": 0.82,
            "out_of_fold": {"recall": 0.7, "precision": 0.6, "f1": 0.65},
        },
        "other_model": {
            "cv_pr_auc_mean": 0.79,
            "out_of_fold": {"recall": 0.9, "precision": 0.9, "f1": 0.9},
        },
    }

    assert select_best_model(selection_metrics) == "cv_winner"
