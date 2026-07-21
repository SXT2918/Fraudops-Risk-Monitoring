"""Business impact metrics for fraud monitoring."""

from __future__ import annotations

import pandas as pd

from src.config import PREVENTED_LOSS_MULTIPLIER


def calculate_business_metrics(
    df: pd.DataFrame,
    probability_col: str = "fraud_probability",
    target_col: str = "is_fraud",
    threshold: float = 0.70,
) -> dict[str, float | int]:
    """Calculate review burden and an illustrative preventable-loss proxy.

    Args:
        df: DataFrame with amount, fraud probability, and true label.
        probability_col: Column containing model fraud probability.
        target_col: Column containing true fraud label.
        threshold: Probability threshold used to flag transactions.

    Returns:
        Dictionary of business metrics. ``estimated_prevented_loss`` is the
        value of correctly flagged fraud under a simplifying 100%-recovery
        assumption; it is not evidence of realized savings.
    """
    if df.empty:
        return {
            "total_transactions": 0,
            "flagged_transactions": 0,
            "fraud_cases_caught": 0,
            "false_alarms": 0,
            "false_negatives": 0,
            "estimated_prevented_loss": 0.0,
            "review_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    flagged = df[probability_col] >= threshold
    fraud = df[target_col] == 1

    true_positives = int((flagged & fraud).sum())
    false_positives = int((flagged & ~fraud).sum())
    false_negatives = int((~flagged & fraud).sum())
    total_fraud = int(fraud.sum())

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(total_fraud, 1)
    prevented_loss = float(df.loc[flagged & fraud, "amount"].sum() * PREVENTED_LOSS_MULTIPLIER)

    return {
        "total_transactions": int(len(df)),
        "flagged_transactions": int(flagged.sum()),
        "fraud_cases_caught": true_positives,
        "false_alarms": false_positives,
        "false_negatives": false_negatives,
        "estimated_prevented_loss": round(prevented_loss, 2),
        "review_rate": round(float(flagged.mean()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
