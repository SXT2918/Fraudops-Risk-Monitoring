"""SQLite database layer for FraudOps.

The database creates a more realistic project than a notebook-only workflow.
For the first version we use SQLite because it runs locally with no setup.
Later this layer can be swapped for PostgreSQL.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import DATABASE_PATH


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create a SQLite connection and ensure parent directory exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    """Create required database tables if they do not already exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_category TEXT NOT NULL,
                transaction_time TEXT NOT NULL,
                location TEXT NOT NULL,
                is_fraud INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scored_transactions (
                transaction_id TEXT PRIMARY KEY,
                fraud_probability REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                decision TEXT NOT NULL,
                scored_at TEXT NOT NULL,
                FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
            );
            """
        )
        conn.commit()


def insert_transactions(df: pd.DataFrame, db_path: Path = DATABASE_PATH) -> int:
    """Insert or replace cleaned transactions into the transactions table.

    Args:
        df: Cleaned transactions.
        db_path: SQLite database path.

    Returns:
        Number of rows written.
    """
    initialize_database(db_path)
    records = df.copy()
    records["transaction_time"] = records["transaction_time"].astype(str)

    with get_connection(db_path) as conn:
        records.to_sql("transactions", conn, if_exists="replace", index=False)
    return len(records)


def fetch_transactions(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Read all transactions from SQLite into a DataFrame."""
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM transactions", conn)


def insert_scored_transactions(
    df: pd.DataFrame,
    db_path: Path = DATABASE_PATH,
) -> int:
    """Insert or replace scored transaction decisions in SQLite.

    Args:
        df: Scored transactions with transaction_id, fraud_probability,
            risk_tier, and decision columns.
        db_path: SQLite database path.

    Returns:
        Number of scored rows written.
    """
    required_columns = ["transaction_id", "fraud_probability", "risk_tier", "decision"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing scored transaction column(s): {', '.join(missing)}")

    initialize_database(db_path)
    scored_at = datetime.now(UTC).isoformat()
    records = [
        (
            str(row.transaction_id),
            float(row.fraud_probability),
            str(row.risk_tier),
            str(row.decision),
            scored_at,
        )
        for row in df[required_columns].itertuples(index=False)
    ]

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO scored_transactions (
                transaction_id,
                fraud_probability,
                risk_tier,
                decision,
                scored_at
            )
            VALUES (?, ?, ?, ?, ?);
            """,
            records,
        )
        conn.commit()
    return len(records)
