import pandas as pd
import pytest

from src.validation import TransactionValidationError, clean_transactions, validate_required_columns


def test_validate_required_columns_raises_for_missing_column():
    df = pd.DataFrame({"transaction_id": ["t1"]})

    with pytest.raises(TransactionValidationError):
        validate_required_columns(df)


def test_clean_transactions_removes_duplicates_and_standardizes_types():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t1", "t2"],
            "user_id": ["u1", "u1", "u2"],
            "amount": [10.5, 10.5, "0.75"],
            "merchant_category": ["grocery", "grocery", None],
            "transaction_time": ["2026-01-01 10:00:00", "2026-01-01 10:00:00", "2026-01-02 03:00:00"],
            "location": ["Toronto", "Toronto", None],
            "is_fraud": [0, 0, 1],
        }
    )

    clean = clean_transactions(df)

    assert len(clean) == 2
    assert clean["transaction_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(clean["transaction_time"])
    assert clean.loc[clean["transaction_id"] == "t2", "merchant_category"].iloc[0] == "unknown"
