"""
test_prediction.py
-------------------
Unit tests for the core prediction / risk-scoring logic.

Run:
    pytest tests/test_prediction.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from predictor import predict_transaction, FraudPredictor


SAMPLE_LOW_RISK = {
    "amount": 250.82, "transaction_hour": 16, "transaction_day": 5,
    "merchant_category": "travel", "payment_method": "wallet", "device_type": "pos",
    "customer_age": 23, "account_age_days": 343, "previous_transactions": 32,
    "failed_attempts": 0, "transactions_last_24h": 1, "average_transaction_amount": 287.92,
    "distance_from_usual_location": 16.92, "is_new_device": 0, "is_international": 0,
    "ip_risk_score": 8.93, "velocity_score": 11.99, "previous_fraud_count": 0,
}

SAMPLE_HIGH_RISK = {
    "amount": 5661.85, "transaction_hour": 14, "transaction_day": 2,
    "merchant_category": "education", "payment_method": "upi", "device_type": "mobile",
    "customer_age": 45, "account_age_days": 1166, "previous_transactions": 24,
    "failed_attempts": 1, "transactions_last_24h": 1, "average_transaction_amount": 1006.29,
    "distance_from_usual_location": 6.86, "is_new_device": 1, "is_international": 0,
    "ip_risk_score": 30.46, "velocity_score": 20.59, "previous_fraud_count": 3,
}


def test_model_prediction_works():
    """A well-formed transaction should return a prediction without error."""
    result = predict_transaction(SAMPLE_LOW_RISK)
    assert result is not None
    assert isinstance(result, dict)


def test_risk_score_between_0_and_100():
    for sample in (SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK):
        result = predict_transaction(sample)
        assert 0 <= result["risk_score"] <= 100


def test_risk_level_is_valid():
    valid_levels = {"LOW", "MEDIUM", "HIGH"}
    for sample in (SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK):
        result = predict_transaction(sample)
        assert result["risk_level"] in valid_levels


def test_prediction_response_has_required_fields():
    result = predict_transaction(SAMPLE_LOW_RISK)
    required_fields = {"fraud_probability", "risk_score", "risk_level", "prediction", "reasons"}
    assert required_fields.issubset(result.keys())
    assert isinstance(result["reasons"], list)
    assert len(result["reasons"]) > 0


def test_fraud_probability_is_valid_probability():
    result = predict_transaction(SAMPLE_LOW_RISK)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_score_to_level_thresholds():
    assert FraudPredictor.score_to_level(0) == "LOW"
    assert FraudPredictor.score_to_level(30) == "LOW"
    assert FraudPredictor.score_to_level(31) == "MEDIUM"
    assert FraudPredictor.score_to_level(70) == "MEDIUM"
    assert FraudPredictor.score_to_level(71) == "HIGH"
    assert FraudPredictor.score_to_level(100) == "HIGH"


def test_missing_field_raises_value_error():
    incomplete = dict(SAMPLE_LOW_RISK)
    del incomplete["amount"]
    with pytest.raises(ValueError):
        predict_transaction(incomplete)


def test_predictions_are_deterministic():
    """Same input should always give the same output (no randomness in inference)."""
    r1 = predict_transaction(SAMPLE_HIGH_RISK)
    r2 = predict_transaction(SAMPLE_HIGH_RISK)
    assert r1["risk_score"] == r2["risk_score"]
    assert r1["fraud_probability"] == r2["fraud_probability"]
