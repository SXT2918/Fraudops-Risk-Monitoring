# FraudOps — Transaction Risk Monitoring Platform

FraudOps is a portfolio project for data analytics, risk analytics, data engineering, analytics engineering, and ML-adjacent internships.

The project will become an end-to-end fraud monitoring system that can:

1. ingest transaction data,
2. validate and clean it,
3. store it in a relational database,
4. engineer fraud-risk features,
5. train and evaluate fraud detection models,
6. score transactions with risk tiers,
7. expose a FastAPI scoring endpoint,
8. display fraud patterns and business impact in a Streamlit dashboard.

This starter version includes the project skeleton, sample data, ingestion, validation, feature engineering, simple business metrics, risk-tier logic, and tests.

---

## Why this project matters

A notebook-only ML project shows that you can train a model. A platform-style project shows that you can build something closer to a real business workflow.

FraudOps is designed to demonstrate:

- Python programming
- SQL / relational data storage
- ETL pipeline design
- data validation
- feature engineering
- fraud/risk analytics
- model-readiness
- business-impact thinking
- testing and maintainability

---

## Project structure

```text
fraudops-risk-monitoring/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── database.py
│   ├── ingestion.py
│   ├── validation.py
│   ├── features.py
│   ├── score.py
│   └── business_metrics.py
├── api/
├── dashboard/
├── tests/
├── assets/
├── requirements.txt
└── README.md
```

---

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run the starter ingestion pipeline

```bash
python -m src.ingestion
```

This uses:

```text
data/sample/sample_transactions.csv
```

It will create:

```text
data/processed/clean_transactions.csv
fraudops.sqlite3
```

---

## Run tests

```bash
pytest
```

---

## Current status

Completed in Step 1:

- project skeleton
- configuration file
- required column validation
- transaction cleaning
- SQLite database layer
- sample transaction dataset
- feature engineering starter
- business metric calculation
- risk-tier mapping
- pytest tests

Next steps:

1. train Logistic Regression, Random Forest, and XGBoost models,
2. tune fraud probability thresholds,
3. save best model and feature column order,
4. build FastAPI scoring endpoint,
5. build Streamlit dashboard,
6. add screenshots and architecture diagram.

---

## Future resume bullet

> Built FraudOps, an end-to-end transaction risk monitoring platform using Python, SQL, FastAPI, Streamlit, and XGBoost to ingest transaction data, engineer behavioral fraud features, score transactions, and visualize fraud patterns through an interactive dashboard.
