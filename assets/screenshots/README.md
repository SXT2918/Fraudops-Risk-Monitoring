# Screenshot Inventory

These screenshots are used by the main project README.

| File | View |
| --- | --- |
| `swagger-docs.png` | FastAPI Swagger docs showing `/health`, `/score_transaction`, and `/score_batch` |
| `streamlit-overview.png` | Streamlit Overview tab with portfolio health KPIs |
| `streamlit-risk-monitoring.png` | Streamlit Risk Monitoring tab with scored transaction queue |
| `streamlit-fraud-pattern-analysis.png` | Streamlit Fraud Pattern Analysis tab with fraud charts |
| `streamlit-model-performance.png` | Streamlit Model Performance tab with metrics and threshold comparison |
| `streamlit-about.png` | Streamlit About tab with project purpose, stack, and architecture summary |

To refresh screenshots locally:

```powershell
python -m src.ingestion
python -m src.train_model
python -m src.evaluate_model
python -m src.seed_scores
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

Then capture:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8501`

Keep screenshots clean:

- Use local app URLs only.
- Avoid showing terminal windows with personal paths.
- Keep the project title visible.
- Crop browser chrome if needed.
