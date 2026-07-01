import pandas as pd

from src import ingestion


def test_run_ingestion_creates_processed_csv(monkeypatch, tmp_path):
    input_path = tmp_path / "raw_transactions.csv"
    output_path = tmp_path / "clean_transactions.csv"
    raw = pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2"],
            "user_id": ["user_1", "user_2"],
            "amount": [10.0, 0.75],
            "merchant_category": ["grocery", "online_services"],
            "transaction_time": ["2026-01-01 10:00:00", "2026-01-01 03:00:00"],
            "location": ["Toronto", "Toronto"],
            "is_fraud": [0, 1],
        }
    )
    raw.to_csv(input_path, index=False)

    monkeypatch.setattr(ingestion, "CLEAN_TRANSACTIONS_FILE", output_path)
    monkeypatch.setattr(ingestion, "insert_transactions", lambda clean: len(clean))

    clean = ingestion.run_ingestion(input_path)

    assert output_path.exists()
    assert len(clean) == 2
    saved = pd.read_csv(output_path)
    assert list(saved["transaction_id"]) == ["txn_2", "txn_1"]
