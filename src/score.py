"""Risk scoring utilities for trained fraud models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLEAN_TRANSACTIONS_FILE,
    FEATURE_COLUMNS_PATH,
    HIGH_RISK_MIN,
    LOW_RISK_MAX,
    MODEL_PATH,
)
from src.features import build_features

MODEL_SETUP_COMMANDS = (
    "Run `python -m src.ingestion` and then `python -m src.train_model` "
    "from the project root."
)


def probability_to_risk_tier(probability: float) -> str:
    """Convert a fraud probability into a business-friendly risk tier."""
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability < HIGH_RISK_MIN:
        return "Medium"
    return "High"


def risk_tier_to_decision(risk_tier: str) -> str:
    """Map risk tier to a review decision."""
    mapping = {
        "Low": "Approve",
        "Medium": "Monitor",
        "High": "Manual Review",
    }
    return mapping.get(risk_tier, "Manual Review")


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained fraud model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained fraud model found at {model_path}. "
            f"{MODEL_SETUP_COMMANDS}"
        )
    return joblib.load(model_path)


def load_feature_columns(feature_columns_path: Path = FEATURE_COLUMNS_PATH) -> list[str]:
    """Load the feature column order used during model training."""
    feature_columns_path = Path(feature_columns_path)
    if not feature_columns_path.exists():
        raise FileNotFoundError(
            f"No feature column file found at {feature_columns_path}. "
            f"{MODEL_SETUP_COMMANDS}"
        )

    columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError(
            f"Feature column file at {feature_columns_path} is invalid. "
            "Expected a JSON list of column names."
        )
    return columns


def align_feature_columns(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Align scored features to the exact training column order.

    Missing training columns are filled with zero. Extra columns created from
    new categories are dropped because the trained model has no coefficients or
    tree splits for them.
    """
    aligned = features.reindex(columns=list(feature_columns), fill_value=0)
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0)


def predict_fraud_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return fraud probabilities from a fitted binary classifier."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))
        if probabilities.ndim != 2:
            raise ValueError("Model returned probabilities in an unexpected format.")
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]

    raise ValueError(
        "Saved model does not support probability predictions. "
        "Train a classifier with `predict_proba` support."
    )


def score_transactions(
    df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> pd.DataFrame:
    """Score transactions with the saved model and business risk rules."""
    if "transaction_id" not in df.columns:
        raise ValueError("Scoring data must include a `transaction_id` column.")

    model = load_model(model_path)
    feature_columns = load_feature_columns(feature_columns_path)
    features = build_features(df, training=False)
    aligned_features = align_feature_columns(features, feature_columns)
    fraud_probabilities = predict_fraud_probabilities(model, aligned_features)

    scores = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].astype(str).values,
            "fraud_probability": fraud_probabilities,
        }
    )
    scores["risk_tier"] = scores["fraud_probability"].apply(probability_to_risk_tier)
    scores["decision"] = scores["risk_tier"].apply(risk_tier_to_decision)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score transactions with the FraudOps model")
    parser.add_argument(
        "--input",
        type=Path,
        default=CLEAN_TRANSACTIONS_FILE,
        help="Path to a transaction CSV to score",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save scored transactions as CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find transactions to score at {args.input}. "
            "Run `python -m src.ingestion` first or pass --input."
        )

    transactions = pd.read_csv(args.input)
    scores = score_transactions(transactions)
    print(scores.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output, index=False)
        print(f"\nSaved scored transactions to {args.output}")


if __name__ == "__main__":
    main()
