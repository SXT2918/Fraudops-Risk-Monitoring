"""FastAPI service for FraudOps transaction risk scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.database import insert_scored_transactions
from src.score import score_transactions

SERVICE_NAME = "fraudops-risk-monitoring"
MODEL_SETUP_MESSAGE = (
    "Model artifacts are missing or unavailable. From the project root, run: "
    "`python -m src.ingestion` and then `python -m src.train_model`."
)

app = FastAPI(
    title="FraudOps Transaction Risk Monitoring API",
    description="Scores transactions for fraud risk and operational review decisions.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Health-check response schema."""

    status: str
    service: str


class TransactionRequest(BaseModel):
    """Incoming transaction fields required for fraud scoring."""

    transaction_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0)
    merchant_category: str = Field(..., min_length=1)
    transaction_time: datetime
    location: str = Field(..., min_length=1)


class ScoredTransactionResponse(BaseModel):
    """Fraud score returned by the API."""

    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_tier: Literal["Low", "Medium", "High"]
    decision: Literal["Approve", "Monitor", "Manual Review"]


def _transactions_to_dataframe(transactions: list[TransactionRequest]) -> pd.DataFrame:
    """Convert validated API payloads into the scorer's DataFrame format."""
    rows = []
    for transaction in transactions:
        row = transaction.model_dump()
        row["transaction_time"] = row["transaction_time"].isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def _score_payload(transactions: list[TransactionRequest]) -> list[ScoredTransactionResponse]:
    """Score validated transactions and persist the results when possible."""
    try:
        input_frame = _transactions_to_dataframe(transactions)
        scores = score_transactions(input_frame)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{MODEL_SETUP_MESSAGE} Details: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        insert_scored_transactions(scores)
    except Exception as exc:  # pragma: no cover - API should still return scores
        print(f"Warning: scored transactions were not saved to SQLite: {exc}")

    return [
        ScoredTransactionResponse(
            transaction_id=str(row.transaction_id),
            fraud_probability=round(float(row.fraud_probability), 6),
            risk_tier=str(row.risk_tier),
            decision=str(row.decision),
        )
        for row in scores.itertuples(index=False)
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple service health check."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


@app.post("/score_transaction", response_model=ScoredTransactionResponse)
def score_transaction(transaction: TransactionRequest) -> ScoredTransactionResponse:
    """Score one transaction for fraud probability and review decision."""
    return _score_payload([transaction])[0]


@app.post("/score_batch", response_model=list[ScoredTransactionResponse])
def score_batch(transactions: list[TransactionRequest]) -> list[ScoredTransactionResponse]:
    """Score a batch of transactions."""
    if not transactions:
        raise HTTPException(status_code=400, detail="Batch scoring needs at least one transaction.")
    return _score_payload(transactions)
