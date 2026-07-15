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

TRANSACTION_COLUMNS = [
    "transaction_id",
    "user_id",
    "amount",
    "merchant_category",
    "transaction_time",
    "location",
    "is_fraud",
]


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create a SQLite connection and ensure parent directory exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_transactions_table_sql(table_name: str = "transactions") -> str:
    """Return the CREATE TABLE statement for cleaned transactions."""
    return f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_category TEXT NOT NULL,
                transaction_time TEXT NOT NULL,
                location TEXT NOT NULL,
                is_fraud INTEGER NOT NULL
            );
            """


def transactions_table_has_primary_key(conn: sqlite3.Connection) -> bool:
    """Return whether transactions.transaction_id is currently a primary key."""
    columns = conn.execute("PRAGMA table_info(transactions);").fetchall()
    return any(column[1] == "transaction_id" and column[5] == 1 for column in columns)


def rebuild_transactions_table(conn: sqlite3.Connection) -> None:
    """Repair older pandas-created transactions tables that lost constraints."""
    conn.execute("DROP TABLE IF EXISTS transactions_new;")
    conn.execute(create_transactions_table_sql("transactions_new"))
    conn.execute(
        """
        INSERT OR REPLACE INTO transactions_new (
            transaction_id,
            user_id,
            amount,
            merchant_category,
            transaction_time,
            location,
            is_fraud
        )
        SELECT
            CAST(transaction_id AS TEXT),
            COALESCE(CAST(user_id AS TEXT), 'unknown'),
            COALESCE(CAST(amount AS REAL), 0),
            COALESCE(CAST(merchant_category AS TEXT), 'unknown'),
            COALESCE(CAST(transaction_time AS TEXT), ''),
            COALESCE(CAST(location AS TEXT), 'unknown'),
            COALESCE(CAST(is_fraud AS INTEGER), 0)
        FROM transactions
        WHERE transaction_id IS NOT NULL;
        """
    )
    conn.execute("DROP TABLE transactions;")
    conn.execute("ALTER TABLE transactions_new RENAME TO transactions;")


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    """Create required database tables if they do not already exist."""
    with get_connection(db_path) as conn:
        conn.execute(create_transactions_table_sql())
        if not transactions_table_has_primary_key(conn):
            rebuild_transactions_table(conn)
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
        conn.execute("DELETE FROM transactions;")
        records[TRANSACTION_COLUMNS].to_sql("transactions", conn, if_exists="append", index=False)
        conn.commit()
    return len(records)


def fetch_transactions(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Read all transactions from SQLite into a DataFrame."""
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM transactions", conn)


def fetch_scored_transactions(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Read scored transactions with optional transaction context."""
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT
                s.transaction_id,
                s.fraud_probability,
                s.risk_tier,
                s.decision,
                s.scored_at,
                t.amount,
                t.merchant_category,
                t.location,
                t.transaction_time
            FROM scored_transactions AS s
            LEFT JOIN transactions AS t
                ON s.transaction_id = t.transaction_id
            ORDER BY s.scored_at DESC, s.fraud_probability DESC;
            """,
            conn,
        )


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
