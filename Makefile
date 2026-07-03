.PHONY: install ingest train evaluate seed-scores sql-list sql-kpis sql-all api dashboard test

install:
	python -m pip install -r requirements.txt

ingest:
	python -m src.ingestion

train:
	python -m src.train_model

evaluate:
	python -m src.evaluate_model

seed-scores:
	python -m src.seed_scores

sql-list:
	python -m src.sql_runner --list

sql-kpis:
	python -m src.sql_runner --file sql/01_fraud_kpis.sql

sql-all:
	python -m src.sql_runner --all

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest
