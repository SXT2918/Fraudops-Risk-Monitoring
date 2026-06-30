"""Raw data ingestion script.

Run from the project root:
    python -m src.ingestion

Or provide a custom input file:
    python -m src.ingestion --input data/raw/transactions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import CLEAN_TRANSACTIONS_FILE, DEFAULT_RAW_FILE
from src.database import insert_transactions
from src.validation import clean_transactions


def load_transactions_csv(input_path: Path) -> pd.DataFrame:
    """Load raw transactions from CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")
    return pd.read_csv(input_path)


def run_ingestion(input_path: Path = DEFAULT_RAW_FILE) -> pd.DataFrame:
    """Load, clean, save, and store transaction data."""
    raw = load_transactions_csv(input_path)
    clean = clean_transactions(raw)

    CLEAN_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_TRANSACTIONS_FILE, index=False)

    row_count = insert_transactions(clean)
    print(f"Loaded {len(raw):,} raw rows")
    print(f"Saved {len(clean):,} clean rows to {CLEAN_TRANSACTIONS_FILE}")
    print(f"Inserted {row_count:,} rows into SQLite database")
    return clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest FraudOps transaction data")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_FILE,
        help="Path to raw transaction CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(args.input)
