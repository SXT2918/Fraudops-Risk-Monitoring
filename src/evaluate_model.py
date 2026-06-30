"""Evaluate the saved FraudOps model on the held-out test split.

Run from the project root:
    python -m src.evaluate_model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.business_metrics import calculate_business_metrics
from src.config import (
    CLEAN_TRANSACTIONS_FILE,
    FEATURE_COLUMNS_PATH,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    THRESHOLD_COMPARISON_VALUES,
)
from src.score import (
    align_feature_columns,
    load_feature_columns,
    load_model,
    predict_fraud_probabilities,
)
from src.train_model import create_train_test_split, evaluate_predictions, load_clean_transactions


def load_training_metrics(metrics_path: Path = MODEL_METRICS_PATH) -> dict[str, Any]:
    """Load saved training metrics if they are available."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_threshold_comparison(
    test_metadata: pd.DataFrame,
    fraud_probabilities: Sequence[float],
    thresholds: Sequence[float] = THRESHOLD_COMPARISON_VALUES,
) -> pd.DataFrame:
    """Create a threshold table with operational fraud-monitoring metrics."""
    evaluation_frame = test_metadata.copy()
    evaluation_frame["fraud_probability"] = list(fraud_probabilities)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        business_metrics = calculate_business_metrics(evaluation_frame, threshold=threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": business_metrics["precision"],
                "recall": business_metrics["recall"],
                "false_positives": business_metrics["false_alarms"],
                "false_negatives": business_metrics["false_negatives"],
                "flagged_transactions": business_metrics["flagged_transactions"],
                "estimated_prevented_loss": business_metrics["estimated_prevented_loss"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_saved_model(
    clean_data_path: Path = CLEAN_TRANSACTIONS_FILE,
    model_path: Path = MODEL_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    metrics_path: Path = MODEL_METRICS_PATH,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reload saved artifacts and evaluate the model on the test split."""
    transactions = load_clean_transactions(Path(clean_data_path))
    model = load_model(Path(model_path))
    feature_columns = load_feature_columns(Path(feature_columns_path))
    saved_metrics = load_training_metrics(Path(metrics_path))

    (
        _x_train,
        x_test,
        _y_train,
        y_test,
        _meta_train,
        meta_test,
        _training_feature_columns,
    ) = create_train_test_split(transactions)

    aligned_test = align_feature_columns(x_test, feature_columns)
    fraud_probabilities = predict_fraud_probabilities(model, aligned_test)
    performance = evaluate_predictions(y_test, fraud_probabilities)
    threshold_table = build_threshold_comparison(meta_test, fraud_probabilities)

    summary = {
        "best_model": saved_metrics.get("best_model", "unknown"),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_columns)),
        "performance": performance,
    }
    return summary, threshold_table


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_evaluation_summary(summary: dict[str, Any], threshold_table: pd.DataFrame) -> None:
    """Print a clean console summary for portfolio demos and local runs."""
    performance = summary["performance"]
    confusion = performance["confusion_matrix"]

    print("FraudOps Model Performance Summary")
    print(f"Best saved model: {summary['best_model']}")
    print(f"Test rows: {summary['test_rows']}")
    print(f"Feature count: {summary['feature_count']}")
    print("")
    print("Model metrics at threshold 0.50")
    print(f"- Precision: {_format_metric(performance['precision'])}")
    print(f"- Recall: {_format_metric(performance['recall'])}")
    print(f"- F1: {_format_metric(performance['f1'])}")
    print(f"- ROC-AUC: {_format_metric(performance['roc_auc'])}")
    print(f"- PR-AUC: {_format_metric(performance['pr_auc'])}")
    print(
        "- Confusion matrix: "
        f"TN={confusion['true_negatives']}, "
        f"FP={confusion['false_positives']}, "
        f"FN={confusion['false_negatives']}, "
        f"TP={confusion['true_positives']}"
    )
    print("")
    print("Threshold comparison")
    print(
        threshold_table.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "estimated_prevented_loss": "${:,.2f}".format,
            },
        )
    )


def main() -> None:
    try:
        summary, threshold_table = evaluate_saved_model()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print_evaluation_summary(summary, threshold_table)


if __name__ == "__main__":
    main()
