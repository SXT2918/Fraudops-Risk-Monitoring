"""Risk scoring utilities.

The real model will be added later. For now this file contains the business
mapping from fraud probability to risk tier and review decision.
"""

from __future__ import annotations

from src.config import HIGH_RISK_MIN, LOW_RISK_MAX


def probability_to_risk_tier(probability: float) -> str:
    """Convert a fraud probability into a business-friendly risk tier."""
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability < HIGH_RISK_MIN:
        return "Medium"
    return "High"


def risk_tier_to_decision(risk_tier: str) -> str:
    """Map risk tier to a review decision."""
    mapping = {
        "Low": "Approve",
        "Medium": "Monitor",
        "High": "Manual Review",
    }
    return mapping.get(risk_tier, "Manual Review")
