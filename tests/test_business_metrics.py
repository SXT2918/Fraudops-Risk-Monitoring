import pandas as pd

from src.business_metrics import calculate_business_metrics


def test_calculate_business_metrics():
    df = pd.DataFrame(
        {
            "amount": [100.0, 50.0, 20.0, 10.0],
            "fraud_probability": [0.90, 0.80, 0.20, 0.10],
            "is_fraud": [1, 0, 1, 0],
        }
    )

    metrics = calculate_business_metrics(df, threshold=0.70)

    assert metrics["total_transactions"] == 4
    assert metrics["flagged_transactions"] == 2
    assert metrics["fraud_cases_caught"] == 1
    assert metrics["false_alarms"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["estimated_prevented_loss"] == 100.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
