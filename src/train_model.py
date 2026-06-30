"""Train and evaluate FraudOps transaction risk models.

Run from the project root:
    python -m src.train_model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLEAN_TRANSACTIONS_FILE,
    FEATURE_COLUMNS_PATH,
    MODEL_EVALUATION_THRESHOLD,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features import build_features
from src.score import predict_fraud_probabilities


def load_clean_transactions(clean_data_path: Path = CLEAN_TRANSACTIONS_FILE) -> pd.DataFrame:
    """Load cleaned transaction data created by the ingestion pipeline."""
    clean_data_path = Path(clean_data_path)
    if not clean_data_path.exists():
        raise FileNotFoundError(
            f"Clean transaction data was not found at {clean_data_path}. "
            "Run `python -m src.ingestion` first to create it."
        )

    df = pd.read_csv(clean_data_path)
    if df.empty:
        raise ValueError(
            f"The cleaned transaction file at {clean_data_path} is empty. "
            "Run `python -m src.ingestion` again with valid transaction data."
        )
    if "is_fraud" not in df.columns:
        raise ValueError("The cleaned transaction data must include an `is_fraud` target column.")
    return df


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create model features, target values, and evaluation metadata."""
    transactions = df.reset_index(drop=True).copy()
    features = build_features(transactions, training=True)
    if "is_fraud" not in features.columns:
        raise ValueError("Feature engineering did not produce the `is_fraud` target column.")

    target = features["is_fraud"].astype(int)
    model_features = features.drop(columns=["is_fraud"])
    metadata = transactions[["transaction_id", "amount", "is_fraud"]].copy()
    metadata["is_fraud"] = metadata["is_fraud"].astype(int)
    metadata["amount"] = pd.to_numeric(metadata["amount"], errors="coerce").fillna(0)
    return model_features, target, metadata


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build features and split data into stratified train and test sets."""
    model_features, target, metadata = prepare_model_frame(df)
    class_counts = target.value_counts()
    if target.nunique() < 2:
        raise ValueError("Model training needs both fraud and non-fraud examples.")
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Model training needs at least two examples of each class so the "
            "train/test split can be stratified."
        )

    split = train_test_split(
        model_features,
        target,
        metadata,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    x_train, x_test, y_train, y_test, meta_train, meta_test = split
    return (
        x_train,
        x_test,
        y_train,
        y_test,
        meta_train,
        meta_test,
        list(model_features.columns),
    )


def build_model_candidates(
    y_train: pd.Series,
    include_xgboost: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Create model candidates with class-imbalance handling where supported."""
    candidates: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("XGBoost is not installed; skipping the XGBoost model.")
        else:
            class_counts = y_train.value_counts()
            negative_count = int(class_counts.get(0, 0))
            positive_count = int(class_counts.get(1, 0))
            scale_pos_weight = negative_count / max(positive_count, 1)
            candidates["xgboost"] = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=1,
            )

    return candidates


def evaluate_predictions(
    y_true: pd.Series,
    fraud_probabilities: pd.Series | list[float],
    threshold: float = MODEL_EVALUATION_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate fraud probability predictions with model and confusion metrics."""
    probabilities = pd.Series(fraud_probabilities, index=y_true.index).astype(float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    roc_auc = None
    if y_true.nunique() == 2:
        roc_auc = round(float(roc_auc_score(y_true, probabilities)), 4)

    return {
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": roc_auc,
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def select_best_model(model_metrics: dict[str, dict[str, Any]]) -> str:
    """Select the best model by PR-AUC, then recall, precision, and F1."""
    trained_models = {
        name: metrics
        for name, metrics in model_metrics.items()
        if "error" not in metrics
    }
    if not trained_models:
        raise RuntimeError("No models trained successfully. Check the training errors above.")

    return max(
        trained_models,
        key=lambda name: (
            trained_models[name]["pr_auc"],
            trained_models[name]["recall"],
            trained_models[name]["precision"],
            trained_models[name]["f1"],
        ),
    )


def train_models(
    clean_data_path: Path = CLEAN_TRANSACTIONS_FILE,
    model_path: Path = MODEL_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    metrics_path: Path = MODEL_METRICS_PATH,
    include_xgboost: bool = True,
) -> dict[str, Any]:
    """Train candidate models, save the best model, and write metrics artifacts."""
    clean_data_path = Path(clean_data_path)
    model_path = Path(model_path)
    feature_columns_path = Path(feature_columns_path)
    metrics_path = Path(metrics_path)

    transactions = load_clean_transactions(clean_data_path)
    (
        x_train,
        x_test,
        y_train,
        y_test,
        _meta_train,
        meta_test,
        feature_columns,
    ) = create_train_test_split(transactions)

    candidates = build_model_candidates(y_train, include_xgboost=include_xgboost)
    model_metrics: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, model in candidates.items():
        try:
            model.fit(x_train, y_train)
            fraud_probabilities = predict_fraud_probabilities(model, x_test)
            model_metrics[model_name] = evaluate_predictions(y_test, fraud_probabilities)
            fitted_models[model_name] = model
        except Exception as exc:  # pragma: no cover - defensive reporting path
            model_metrics[model_name] = {"error": str(exc)}
            print(f"Skipped {model_name} because training or evaluation failed: {exc}")

    best_model_name = select_best_model(model_metrics)
    best_model = fitted_models[best_model_name]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    feature_columns_path.write_text(
        json.dumps(feature_columns, indent=2),
        encoding="utf-8",
    )

    metrics_payload: dict[str, Any] = {
        "best_model": best_model_name,
        "selection_rule": "Highest PR-AUC, then recall, precision, and F1.",
        "evaluation_threshold": MODEL_EVALUATION_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_columns)),
        "test_transaction_ids": meta_test["transaction_id"].astype(str).tolist(),
        "models": model_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Model training complete")
    for model_name, metrics in model_metrics.items():
        if "error" in metrics:
            print(f"- {model_name}: skipped ({metrics['error']})")
            continue
        print(
            f"- {model_name}: PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'}, "
            f"recall={metrics['recall']:.4f}, "
            f"precision={metrics['precision']:.4f}"
        )
    print(f"Best model: {best_model_name}")
    print(f"Saved model to {model_path}")
    print(f"Saved feature columns to {feature_columns_path}")
    print(f"Saved metrics to {metrics_path}")

    return metrics_payload


def main() -> None:
    try:
        train_models()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
