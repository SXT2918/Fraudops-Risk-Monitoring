import pandas as pd

from src.features import build_features


def test_build_features_creates_core_risk_features():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "user_id": ["u1", "u1"],
            "amount": [0.75, 120.0],
            "merchant_category": ["online", "travel"],
            "transaction_time": ["2026-01-01 03:15:00", "2026-01-01 14:00:00"],
            "location": ["Toronto", "Toronto"],
            "is_fraud": [1, 0],
        }
    )

    features = build_features(df, training=True)

    assert "transaction_hour" in features.columns
    assert "is_night_transaction" in features.columns
    assert "is_micro_charge" in features.columns
    assert "log_amount" in features.columns
    assert "user_transaction_count" not in features.columns
    assert "average_user_transaction_amount" not in features.columns
    assert "merchant_risk_rate" not in features.columns
    assert features.loc[0, "is_night_transaction"] == 1
    assert features.loc[0, "is_micro_charge"] == 1


def test_feature_values_do_not_depend_on_fraud_labels():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "user_id": ["u1", "u2"],
            "amount": [7.5, 120.0],
            "merchant_category": ["online", "travel"],
            "transaction_time": ["2026-01-01 03:15:00", "2026-01-02 14:00:00"],
            "location": ["Toronto", "Vancouver"],
            "is_fraud": [1, 0],
        }
    )
    relabeled = transactions.copy()
    relabeled["is_fraud"] = [0, 1]

    original_features = build_features(transactions, training=True).drop(columns="is_fraud")
    relabeled_features = build_features(relabeled, training=True).drop(columns="is_fraud")

    pd.testing.assert_frame_equal(original_features, relabeled_features)
