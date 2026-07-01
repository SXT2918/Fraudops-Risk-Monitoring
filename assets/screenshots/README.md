# Screenshot Checklist

Add screenshots here before publishing the GitHub portfolio project.

Recommended captures:

1. `swagger-docs.png`
   - Start the API with `uvicorn api.main:app --reload`.
   - Open `http://127.0.0.1:8000/docs`.
   - Capture the Swagger UI showing `/health`, `/score_transaction`, and `/score_batch`.

2. `streamlit-overview.png`
   - Run `python -m src.seed_scores`.
   - Start the dashboard with `streamlit run dashboard/app.py`.
   - Capture the Overview tab with KPI metrics visible.

3. `streamlit-risk-monitoring.png`
   - Capture the Risk Monitoring tab with seeded High / Manual Review rows visible.

4. `streamlit-model-performance.png`
   - Capture the Model Performance tab with model metrics and threshold comparison visible.

Keep screenshots clean:

- Use the local app URLs only.
- Avoid showing terminal windows with personal paths if you do not want them public.
- Crop browser chrome if needed, but keep the project title visible.
