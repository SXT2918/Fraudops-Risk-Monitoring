from src.score import probability_to_risk_tier, risk_tier_to_decision


def test_probability_to_risk_tier():
    assert probability_to_risk_tier(0.10) == "Low"
    assert probability_to_risk_tier(0.50) == "Medium"
    assert probability_to_risk_tier(0.90) == "High"


def test_risk_tier_to_decision():
    assert risk_tier_to_decision("Low") == "Approve"
    assert risk_tier_to_decision("Medium") == "Monitor"
    assert risk_tier_to_decision("High") == "Manual Review"
