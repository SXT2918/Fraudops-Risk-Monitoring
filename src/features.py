"""Feature engineering for transaction risk modeling.

This module turns raw transaction records into model-ready columns.
The first version uses simple, explainable fraud-risk features that are
understandable to both technical and non-technical reviewers.
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


def add_user_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple user-level aggregate behavior features.

    In a real system, these would often be rolling-window features such as
    "transactions in the past 1 hour." For the first version, we use global
    aggregates so the project stays easy to understand and run locally.
    """
    result = df.copy()
    user_stats = (
        result.groupby("user_id")
        .agg(
            user_transaction_count=("transaction_id", "count"),
            average_user_transaction_amount=("amount", "mean"),
        )
        .reset_index()
    )
    result = result.merge(user_stats, on="user_id", how="left")
    return result


def add_merchant_risk_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add merchant-category fraud rate when labels are available."""
    result = df.copy()
    if "is_fraud" not in result.columns:
        result["merchant_risk_rate"] = 0.0
        return result

    merchant_stats = (
        result.groupby("merchant_category")
        .agg(merchant_risk_rate=("is_fraud", "mean"))
        .reset_index()
    )
    result = result.merge(merchant_stats, on="merchant_category", how="left")
    result["merchant_risk_rate"] = result["merchant_risk_rate"].fillna(0.0)
    return result


def build_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Build model-ready fraud-risk features.

    Args:
        df: Clean transaction data.
        training: Whether labels are available. If True and is_fraud exists,
            the output keeps the target column.

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
    result = add_user_behavior_features(result)
    result = add_merchant_risk_rate(result)

    # One-hot encode low-cardinality categorical variables.
    categorical_cols = ["merchant_category", "location", "amount_bucket"]
    result = pd.get_dummies(result, columns=categorical_cols, drop_first=False, dtype=int)

    # Remove raw identifiers and timestamp before modeling.
    columns_to_drop = ["transaction_id", "user_id", "transaction_time"]
    result = result.drop(columns=[col for col in columns_to_drop if col in result.columns])

    if not training and "is_fraud" in result.columns:
        result = result.drop(columns=["is_fraud"])

    return result
