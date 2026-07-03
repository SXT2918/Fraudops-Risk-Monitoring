# FraudOps SQL Analytics Layer

This folder contains reusable SQL analyses for the SQLite-backed FraudOps
transaction monitoring workflow. The queries are written for SQLite while
staying close to portable SQL patterns that could later move to PostgreSQL.

The SQL files translate cleaned transaction data and model-scored transaction
data into fraud operations metrics: fraud rate, review volume, high-risk
merchant categories, amount-bucket risk, scored-queue monitoring, and
business-impact proxies.

## Files

```text
sql/
|-- 00_schema_overview.sql
|-- 01_fraud_kpis.sql
|-- 02_hourly_fraud_patterns.sql
|-- 03_amount_bucket_analysis.sql
|-- 04_merchant_risk_analysis.sql
|-- 05_threshold_review_volume.sql
|-- 06_scored_transaction_monitoring.sql
|-- 07_business_impact_summary.sql
`-- README.md
```

## Run The Queries

From the project root:

```powershell
python -m src.ingestion
python -m src.train_model
python -m src.seed_scores
python -m src.sql_runner --list
python -m src.sql_runner --file sql/01_fraud_kpis.sql
python -m src.sql_runner --all
```

The sample dataset is intentionally small for local testing, so SQL outputs
validate the analytics workflow rather than represent production fraud behavior.

Estimated prevented loss in this project is a proxy based on high-risk scored
transaction amounts. It is not confirmed real savings and should not be treated
as production-grade fraud performance.
