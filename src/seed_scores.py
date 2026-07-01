"""Seed scored transaction rows for dashboard demos.

Run from the project root:
    python -m src.seed_scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import CLEAN_TRANSACTIONS_FILE
from src.database import insert_scored_transactions
from src.score import score_transactions


def load_seed_transactions(input_path: Path = CLEAN_TRANSACTIONS_FILE, limit: int = 12) -> pd.DataFrame:
    """Load a small set of cleaned transactions to score for dashboard demos."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Clean transaction data was not found at {input_path}. "
            "Run `python -m src.ingestion` first."
        )
    if limit <= 0:
        raise ValueError("Seed limit must be greater than zero.")

    transactions = pd.read_csv(input_path)
    if transactions.empty:
        raise ValueError(
            f"The cleaned transaction file at {input_path} is empty. "
            "Run `python -m src.ingestion` again with valid transaction data."
        )
    return transactions.head(limit).copy()


def seed_scored_transactions(
    input_path: Path = CLEAN_TRANSACTIONS_FILE,
    limit: int = 12,
) -> pd.DataFrame:
    """Score sample transactions and insert results into SQLite."""
    transactions = load_seed_transactions(input_path=input_path, limit=limit)
    scores = score_transactions(transactions)
    inserted_count = insert_scored_transactions(scores)

    print(f"Loaded {len(transactions):,} transactions from {input_path}")
    print(f"Inserted {inserted_count:,} scored transactions into SQLite")
    print(scores.to_string(index=False))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed scored FraudOps transactions")
    parser.add_argument(
        "--input",
        type=Path,
        default=CLEAN_TRANSACTIONS_FILE,
        help="Clean transaction CSV to score",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of cleaned transactions to score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        seed_scored_transactions(input_path=args.input, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
