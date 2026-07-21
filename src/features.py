"""Leakage-safe feature engineering for transaction risk modeling.

Every model input in this module is available from the transaction being
scored. Labels are retained only as the training target; they never influence
the feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_amount_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Create human-readable transaction amount buckets."""
    result = df.copy()
    bins = [-0.01, 1, 10, 50, 100, 500, np.inf]
    labels = ["micro", "small", "medium", "large", "very_large", "extreme"]
    result["amount_bucket"] = pd.cut(result["amount"], bins=bins, labels=labels)
    result["amount_bucket"] = result["amount_bucket"].astype(str)
    return result


def build_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Build model-ready fraud-risk features.

    Args:
        df: Clean transaction data.
        training: If True, preserve ``is_fraud`` as the target column. The
            target is never used to calculate an input feature.

    Returns:
        DataFrame containing engineered numeric/model-ready features.
    """
    result = df.copy()
    result["transaction_time"] = pd.to_datetime(result["transaction_time"], errors="coerce")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce").fillna(0)

    result["transaction_hour"] = result["transaction_time"].dt.hour.fillna(0).astype(int)
    result["transaction_dayofweek"] = result["transaction_time"].dt.dayofweek.fillna(0).astype(int)
    result["is_weekend"] = result["transaction_dayofweek"].isin([5, 6]).astype(int)
    result["is_night_transaction"] = result["transaction_hour"].between(0, 5).astype(int)
    result["is_micro_charge"] = (result["amount"] < 1.00).astype(int)
    result["log_amount"] = np.log1p(result["amount"])

    result = add_amount_bucket(result)

    # One-hot encode low-cardinality categorical variables.
    categorical_cols = ["merchant_category", "location", "amount_bucket"]
    result = pd.get_dummies(result, columns=categorical_cols, drop_first=False, dtype=int)

    # Remove raw identifiers and timestamp before modeling.
    columns_to_drop = ["transaction_id", "user_id", "transaction_time"]
    result = result.drop(columns=[col for col in columns_to_drop if col in result.columns])

    if not training and "is_fraud" in result.columns:
        result = result.drop(columns=["is_fraud"])

    return result
