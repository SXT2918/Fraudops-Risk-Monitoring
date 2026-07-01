import pandas as pd

from src.seed_scores import seed_scored_transactions


def test_seed_scored_transactions_scores_and_inserts(monkeypatch, tmp_path):
    clean_data_path = tmp_path / "clean_transactions.csv"
    pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2"],
            "user_id": ["user_1", "user_2"],
            "amount": [0.75, 45.0],
            "merchant_category": ["online_services", "grocery"],
            "transaction_time": ["2026-01-01 03:00:00", "2026-01-01 14:00:00"],
            "location": ["Toronto", "Vancouver"],
            "is_fraud": [1, 0],
        }
    ).to_csv(clean_data_path, index=False)

    inserted_counts = []

    def fake_score_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": transactions["transaction_id"],
                "fraud_probability": [0.9, 0.2],
                "risk_tier": ["High", "Low"],
                "decision": ["Manual Review", "Approve"],
            }
        )

    def fake_insert_scored_transactions(scores: pd.DataFrame) -> int:
        inserted_counts.append(len(scores))
        return len(scores)

    monkeypatch.setattr("src.seed_scores.score_transactions", fake_score_transactions)
    monkeypatch.setattr(
        "src.seed_scores.insert_scored_transactions",
        fake_insert_scored_transactions,
    )

    scores = seed_scored_transactions(input_path=clean_data_path, limit=2)

    assert len(scores) == 2
    assert inserted_counts == [2]
    assert scores.loc[0, "decision"] == "Manual Review"
