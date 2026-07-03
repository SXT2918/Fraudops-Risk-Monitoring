from pathlib import Path

import pandas as pd
import pytest

from src import ingestion
from src.config import DEFAULT_RAW_FILE, SQL_DIR
from src.database import insert_scored_transactions, insert_transactions
from src.sql_runner import (
    SQLRunnerError,
    execute_all_sql_files,
    execute_sql_file,
    list_sql_files,
)


REQUIRED_SQL_FILES = [
    "00_schema_overview.sql",
    "01_fraud_kpis.sql",
    "02_hourly_fraud_patterns.sql",
    "03_amount_bucket_analysis.sql",
    "04_merchant_risk_analysis.sql",
    "05_threshold_review_volume.sql",
    "06_scored_transaction_monitoring.sql",
    "07_business_impact_summary.sql",
]


def build_sample_sql_database(tmp_path, monkeypatch) -> tuple[Path, pd.DataFrame]:
    """Run the ingestion path into a temporary SQLite database."""
    db_path = tmp_path / "fraudops.sqlite3"
    clean_output_path = tmp_path / "clean_transactions.csv"

    monkeypatch.setattr(ingestion, "CLEAN_TRANSACTIONS_FILE", clean_output_path)
    monkeypatch.setattr(
        ingestion,
        "insert_transactions",
        lambda clean: insert_transactions(clean, db_path=db_path),
    )

    clean = ingestion.run_ingestion(DEFAULT_RAW_FILE)
    return db_path, clean


def seed_sample_scores(clean: pd.DataFrame, db_path: Path) -> None:
    """Insert deterministic scored rows without relying on model artifacts."""
    transaction_ids = clean["transaction_id"].head(5).astype(str).tolist()
    scores = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "fraud_probability": [0.92, 0.74, 0.51, 0.28, 0.08],
            "risk_tier": ["High", "High", "Medium", "Low", "Low"],
            "decision": ["Manual Review", "Manual Review", "Monitor", "Approve", "Approve"],
        }
    )
    insert_scored_transactions(scores, db_path=db_path)


def test_sql_folder_and_required_files_exist():
    assert SQL_DIR.exists()

    actual_files = {path.name for path in SQL_DIR.glob("*.sql")}

    assert set(REQUIRED_SQL_FILES).issubset(actual_files)


def test_list_sql_files_returns_required_files_in_order():
    sql_files = list_sql_files()
    file_names = [path.name for path in sql_files]

    assert file_names == sorted(file_names)
    assert set(REQUIRED_SQL_FILES).issubset(file_names)


def test_kpi_sql_runs_after_ingestion(monkeypatch, tmp_path):
    db_path, clean = build_sample_sql_database(tmp_path, monkeypatch)

    results = execute_sql_file(SQL_DIR / "01_fraud_kpis.sql", db_path=db_path)
    kpi_row = results[0].dataframe.iloc[0]

    assert len(results) == 1
    assert int(kpi_row["total_transactions"]) == len(clean)
    assert int(kpi_row["total_fraud_cases"]) == int(clean["is_fraud"].sum())


def test_all_sql_files_run_on_sample_data(monkeypatch, tmp_path):
    db_path, clean = build_sample_sql_database(tmp_path, monkeypatch)
    seed_sample_scores(clean, db_path)

    results = execute_all_sql_files(db_path=db_path)
    result_files = {result.file_path.name for result in results}

    assert results
    assert set(REQUIRED_SQL_FILES).issubset(result_files)


def test_missing_database_raises_friendly_error(tmp_path):
    missing_database = tmp_path / "missing.sqlite3"

    with pytest.raises(SQLRunnerError, match="python -m src.ingestion"):
        execute_sql_file(SQL_DIR / "01_fraud_kpis.sql", db_path=missing_database)
