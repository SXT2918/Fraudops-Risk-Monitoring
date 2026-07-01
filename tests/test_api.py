import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def sample_transaction(transaction_id: str = "txn_001") -> dict[str, str | float]:
    return {
        "transaction_id": transaction_id,
        "user_id": "user_123",
        "amount": 0.75,
        "merchant_category": "online_services",
        "transaction_time": "2026-01-15T03:24:00",
        "location": "Toronto",
    }


def fake_scores(input_frame: pd.DataFrame) -> pd.DataFrame:
    probabilities = [0.87 if index == 0 else 0.18 for index in range(len(input_frame))]
    risk_tiers = ["High" if probability >= 0.70 else "Low" for probability in probabilities]
    decisions = ["Manual Review" if tier == "High" else "Approve" for tier in risk_tiers]
    return pd.DataFrame(
        {
            "transaction_id": input_frame["transaction_id"].astype(str).tolist(),
            "fraud_probability": probabilities,
            "risk_tier": risk_tiers,
            "decision": decisions,
        }
    )


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "fraudops-risk-monitoring",
    }


def test_score_transaction_returns_scored_result(monkeypatch):
    monkeypatch.setattr("api.main.score_transactions", fake_scores)
    monkeypatch.setattr("api.main.insert_scored_transactions", lambda _scores: 1)

    response = client.post("/score_transaction", json=sample_transaction())

    assert response.status_code == 200
    result = response.json()
    assert result["transaction_id"] == "txn_001"
    assert result["fraud_probability"] == 0.87
    assert result["risk_tier"] == "High"
    assert result["decision"] == "Manual Review"


def test_score_batch_returns_multiple_scored_results(monkeypatch):
    inserted_counts = []

    def capture_insert(scores: pd.DataFrame) -> int:
        inserted_counts.append(len(scores))
        return len(scores)

    monkeypatch.setattr("api.main.score_transactions", fake_scores)
    monkeypatch.setattr("api.main.insert_scored_transactions", capture_insert)

    response = client.post(
        "/score_batch",
        json=[sample_transaction("txn_001"), sample_transaction("txn_002")],
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["transaction_id"] == "txn_001"
    assert results[1]["transaction_id"] == "txn_002"
    assert results[1]["risk_tier"] == "Low"
    assert inserted_counts == [2]


def test_score_transaction_missing_required_field_returns_validation_error():
    payload = sample_transaction()
    payload.pop("amount")

    response = client.post("/score_transaction", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "amount"


def test_score_transaction_missing_model_artifacts_returns_friendly_error(monkeypatch):
    def raise_missing_artifacts(_input_frame: pd.DataFrame) -> pd.DataFrame:
        raise FileNotFoundError("No trained fraud model found.")

    monkeypatch.setattr("api.main.score_transactions", raise_missing_artifacts)

    response = client.post("/score_transaction", json=sample_transaction())

    assert response.status_code == 503
    assert "python -m src.ingestion" in response.json()["detail"]
    assert "python -m src.train_model" in response.json()["detail"]
