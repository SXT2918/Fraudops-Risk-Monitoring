# Contributing

FraudOps is a portfolio project, so contributions should keep the code readable, reproducible, and easy for reviewers to run locally.

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Development Workflow

Run the full local pipeline before opening or merging changes:

```powershell
python -m src.ingestion
python -m src.train_model
python -m src.evaluate_model
python -m src.seed_scores
pytest
```

## Code Guidelines

- Keep changes incremental and modular.
- Use `src/config.py` for shared paths and constants.
- Add tests for new behavior when practical.
- Keep error messages beginner-friendly.
- Do not commit local databases, generated CSVs, `.venv`, logs, or model pickle files.

## Documentation

Update `README.md` when changing setup, commands, API behavior, dashboard behavior, or the project architecture.
