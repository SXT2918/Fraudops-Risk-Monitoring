import sqlite3

import pandas as pd

from src.database import (
    fetch_scored_transactions,
    initialize_database,
    insert_scored_transactions,
    insert_transactions,
)


def test_initialize_database_creates_required_tables(tmp_path):
    db_path = tmp_path / "fraudops.sqlite3"

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table';"
            ).fetchall()
        }
        transaction_columns = conn.execute("PRAGMA table_info(transactions);").fetchall()

    assert {"transactions", "scored_transactions"}.issubset(table_names)
    transaction_id_column = [col for col in transaction_columns if col[1] == "transaction_id"][0]
    assert transaction_id_column[5] == 1


def test_insert_and_fetch_scored_transactions(tmp_path):
    db_path = tmp_path / "fraudops.sqlite3"
    transactions = pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2"],
            "user_id": ["user_1", "user_2"],
            "amount": [100.0, 0.75],
            "merchant_category": ["electronics", "online_services"],
            "transaction_time": ["2026-01-01 10:00:00", "2026-01-01 03:00:00"],
            "location": ["Toronto", "Toronto"],
            "is_fraud": [0, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "transaction_id": ["txn_1", "txn_2"],
            "fraud_probability": [0.22, 0.91],
            "risk_tier": ["Low", "High"],
            "decision": ["Approve", "Manual Review"],
        }
    )

    insert_transactions(transactions, db_path=db_path)
    inserted = insert_scored_transactions(scores, db_path=db_path)
    fetched = fetch_scored_transactions(db_path=db_path)

    assert inserted == 2
    assert len(fetched) == 2
    assert set(fetched["transaction_id"]) == {"txn_1", "txn_2"}
    assert "scored_at" in fetched.columns
    assert fetched.loc[fetched["transaction_id"] == "txn_2", "risk_tier"].iloc[0] == "High"


def test_insert_transactions_preserves_transaction_primary_key(tmp_path):
    db_path = tmp_path / "fraudops.sqlite3"
    transactions = pd.DataFrame(
        {
            "transaction_id": ["txn_1"],
            "user_id": ["user_1"],
            "amount": [100.0],
            "merchant_category": ["electronics"],
            "transaction_time": ["2026-01-01 10:00:00"],
            "location": ["Toronto"],
            "is_fraud": [0],
        }
    )

    insert_transactions(transactions, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        transaction_columns = conn.execute("PRAGMA table_info(transactions);").fetchall()

    transaction_id_column = [col for col in transaction_columns if col[1] == "transaction_id"][0]
    assert transaction_id_column[5] == 1


def test_fetch_scored_transactions_returns_empty_table_when_no_scores(tmp_path):
    db_path = tmp_path / "fraudops.sqlite3"

    fetched = fetch_scored_transactions(db_path=db_path)

    assert fetched.empty
    assert "transaction_id" in fetched.columns
