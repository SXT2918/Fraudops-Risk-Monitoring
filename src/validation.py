"""Data validation and cleaning utilities.

This module answers: "Can we trust the raw data enough to use it?"
It validates the incoming transaction file, standardizes types, removes
obvious issues, and creates a clean table for downstream modeling.
"""

from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_COLUMNS


class TransactionValidationError(ValueError):
    """Raised when transaction data fails validation."""


def validate_required_columns(df: pd.DataFrame) -> None:
    """Check that all required raw transaction columns are present.

    Args:
        df: Raw transaction DataFrame.

    Raises:
        TransactionValidationError: If one or more required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise TransactionValidationError(
            f"Missing required column(s): {', '.join(missing)}"
        )


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction data into a consistent modeling table.

    Cleaning steps:
    - keep required columns in a stable order
    - convert transaction_time to datetime
    - convert amount to numeric
    - fill missing categorical values
    - remove invalid negative amounts
    - remove duplicate transaction IDs
    - convert is_fraud to integer 0/1

    Args:
        df: Raw transaction DataFrame.

    Returns:
        Cleaned transaction DataFrame.
    """
    validate_required_columns(df)

    cleaned = df[REQUIRED_COLUMNS].copy()

    cleaned["transaction_time"] = pd.to_datetime(
        cleaned["transaction_time"], errors="coerce"
    )
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")

    # Drop rows that cannot be used safely for modeling.
    cleaned = cleaned.dropna(subset=["transaction_id", "user_id", "amount", "transaction_time"])
    cleaned = cleaned[cleaned["amount"] >= 0].copy()

    # Fill categorical fields with explicit unknown labels.
    cleaned["merchant_category"] = cleaned["merchant_category"].fillna("unknown")
    cleaned["location"] = cleaned["location"].fillna("unknown")

    # Normalize string fields.
    for col in ["transaction_id", "user_id", "merchant_category", "location"]:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Keep the first occurrence of a transaction ID.
    cleaned = cleaned.drop_duplicates(subset=["transaction_id"], keep="first")

    # Make target column consistent.
    cleaned["is_fraud"] = cleaned["is_fraud"].fillna(0).astype(int)
    cleaned["is_fraud"] = cleaned["is_fraud"].clip(lower=0, upper=1)

    cleaned = cleaned.sort_values("transaction_time").reset_index(drop=True)
    return cleaned
