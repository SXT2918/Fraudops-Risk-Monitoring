"""Train and evaluate FraudOps transaction risk models.

Run from the project root:
    python -m src.train_model
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    """Create leakage-safe model features, targets, and evaluation metadata."""
    transactions = df.reset_index(drop=True).copy()
    features = build_features(transactions, training=True)
    if "is_fraud" not in features.columns:
        raise ValueError("Feature engineering did not produce the `is_fraud` target column.")

    target = features["is_fraud"].astype(int)
    model_features = features.drop(columns=["is_fraud"])
    metadata_columns = ["transaction_id", "amount", "is_fraud", "transaction_time"]
    metadata = transactions[metadata_columns].copy()
    metadata["is_fraud"] = metadata["is_fraud"].astype(int)
    metadata["amount"] = pd.to_numeric(metadata["amount"], errors="coerce").fillna(0)
    return model_features, target, metadata


def _contains_both_classes(df: pd.DataFrame) -> bool:
    """Return whether a frame contains fraud and non-fraud examples."""
    return df["is_fraud"].astype(int).nunique() == 2


def split_raw_transactions(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split raw rows before feature engineering.

    A later-time holdout best matches deployment: train on the past and test on
    the future. Tiny or irregular demo datasets can make that split lose a
    class, so a reproducible stratified fallback is used and recorded.
    """
    transactions = df.reset_index(drop=True).copy()
    target = transactions["is_fraud"].astype(int)
    class_counts = target.value_counts()
    if target.nunique() < 2:
        raise ValueError("Model training needs both fraud and non-fraud examples.")
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Model training needs at least two examples of each class so the "
            "train/test split can be stratified."
        )

    parsed_time = pd.to_datetime(transactions["transaction_time"], errors="coerce")
    n_test = max(1, int(math.ceil(len(transactions) * test_size)))
    n_train = len(transactions) - n_test
    use_temporal = parsed_time.notna().all() and n_train > 0

    if use_temporal:
        ordered = transactions.assign(_parsed_time=parsed_time).sort_values(
            "_parsed_time", kind="mergesort"
        )
        temporal_train = ordered.iloc[:n_train].drop(columns="_parsed_time")
        temporal_test = ordered.iloc[n_train:].drop(columns="_parsed_time")
        train_class_counts = temporal_train["is_fraud"].astype(int).value_counts()
        use_temporal = (
            _contains_both_classes(temporal_train)
            and _contains_both_classes(temporal_test)
            and int(train_class_counts.min()) >= 2
        )

    if use_temporal:
        raw_train = temporal_train.reset_index(drop=True)
        raw_test = temporal_test.reset_index(drop=True)
        strategy = "later-time holdout"
        fallback_reason = None
    else:
        raw_train, raw_test = train_test_split(
            transactions,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        raw_train = raw_train.reset_index(drop=True)
        raw_test = raw_test.reset_index(drop=True)
        strategy = "stratified random fallback"
        fallback_reason = (
            "A later-time holdout could not preserve both classes with enough "
            "training examples."
        )

    train_times = pd.to_datetime(raw_train["transaction_time"], errors="coerce")
    test_times = pd.to_datetime(raw_test["transaction_time"], errors="coerce")
    split_info = {
        "strategy": strategy,
        "fallback_reason": fallback_reason,
        "train_time_min": train_times.min().isoformat() if train_times.notna().any() else None,
        "train_time_max": train_times.max().isoformat() if train_times.notna().any() else None,
        "test_time_min": test_times.min().isoformat() if test_times.notna().any() else None,
        "test_time_max": test_times.max().isoformat() if test_times.notna().any() else None,
    }
    return raw_train, raw_test, split_info


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    dict[str, Any],
]:
    """Split raw rows, then build train-only feature columns.

    Test-only categories are aligned to the feature vocabulary learned from the
    training partition. This mirrors the behavior used for live scoring.
    """
    raw_train, raw_test, split_info = split_raw_transactions(
        df, test_size=test_size, random_state=random_state
    )
    x_train, y_train, meta_train = prepare_model_frame(raw_train)
    x_test_unaligned, y_test, meta_test = prepare_model_frame(raw_test)
    x_test = x_test_unaligned.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (
        x_train.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        meta_train.reset_index(drop=True),
        meta_test.reset_index(drop=True),
        list(x_train.columns),
        split_info,
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


def cross_validate_candidates(
    candidates: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, Any]]:
    """Evaluate candidates with out-of-fold predictions on training data only."""
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Cross-validation needs at least two training examples per class.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    selection_metrics: dict[str, dict[str, Any]] = {}
    y_reset = y_train.reset_index(drop=True)
    x_reset = x_train.reset_index(drop=True)

    for model_name, model in candidates.items():
        try:
            out_of_fold_probabilities = np.zeros(len(y_reset), dtype=float)
            fold_pr_auc: list[float] = []
            for fold_train_index, fold_validation_index in splitter.split(x_reset, y_reset):
                fold_model = clone(model)
                fold_model.fit(
                    x_reset.iloc[fold_train_index], y_reset.iloc[fold_train_index]
                )
                fold_probabilities = predict_fraud_probabilities(
                    fold_model, x_reset.iloc[fold_validation_index]
                )
                out_of_fold_probabilities[fold_validation_index] = fold_probabilities
                fold_pr_auc.append(
                    float(
                        average_precision_score(
                            y_reset.iloc[fold_validation_index], fold_probabilities
                        )
                    )
                )

            out_of_fold_metrics = evaluate_predictions(
                y_reset, out_of_fold_probabilities
            )
            selection_metrics[model_name] = {
                "cv_folds": n_splits,
                "cv_pr_auc_mean": round(float(np.mean(fold_pr_auc)), 4),
                "cv_pr_auc_std": round(float(np.std(fold_pr_auc)), 4),
                "out_of_fold": out_of_fold_metrics,
            }
        except Exception as exc:  # pragma: no cover - defensive reporting path
            selection_metrics[model_name] = {"error": str(exc)}
            print(f"Skipped {model_name} during cross-validation: {exc}")

    return selection_metrics


def select_best_model(model_metrics: dict[str, dict[str, Any]]) -> str:
    """Select by training-only CV PR-AUC, then out-of-fold threshold metrics."""
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
            trained_models[name]["cv_pr_auc_mean"],
            trained_models[name]["out_of_fold"]["recall"],
            trained_models[name]["out_of_fold"]["precision"],
            trained_models[name]["out_of_fold"]["f1"],
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
        split_info,
    ) = create_train_test_split(transactions)

    candidates = build_model_candidates(y_train, include_xgboost=include_xgboost)
    selection_metrics = cross_validate_candidates(candidates, x_train, y_train)
    best_model_name = select_best_model(selection_metrics)
    best_model = candidates[best_model_name]
    best_model.fit(x_train, y_train)
    test_probabilities = predict_fraud_probabilities(best_model, x_test)
    test_metrics = evaluate_predictions(y_test, test_probabilities)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    feature_columns_path.write_text(
        json.dumps(feature_columns, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_payload: dict[str, Any] = {
        "best_model": best_model_name,
        "selection_rule": (
            "Highest mean PR-AUC in stratified cross-validation on the training "
            "partition; ties use out-of-fold recall, precision, then F1."
        ),
        "validation_strategy": {
            "model_selection": "Stratified cross-validation on training rows only",
            "final_evaluation": "One untouched holdout evaluated after model selection",
            "holdout_split": split_info,
        },
        "evaluation_threshold": MODEL_EVALUATION_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_columns)),
        "test_transaction_ids": meta_test["transaction_id"].astype(str).tolist(),
        "selection_metrics": selection_metrics,
        "test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print("Model training complete")
    print("Training-only cross-validation")
    for model_name, metrics in selection_metrics.items():
        if "error" in metrics:
            print(f"- {model_name}: skipped ({metrics['error']})")
            continue
        print(
            f"- {model_name}: mean CV PR-AUC={metrics['cv_pr_auc_mean']:.4f} "
            f"(+/- {metrics['cv_pr_auc_std']:.4f})"
        )
    print(f"Best model: {best_model_name}")
    print(
        f"Untouched holdout: PR-AUC={test_metrics['pr_auc']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"precision={test_metrics['precision']:.4f}"
    )
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
