.PHONY: install ingest train evaluate seed-scores api dashboard test

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

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest
