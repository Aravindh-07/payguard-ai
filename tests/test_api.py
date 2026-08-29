"""
test_api.py
-----------
Integration tests for the FastAPI backend using TestClient (no live server needed).

Run:
    pytest tests/test_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

VALID_TRANSACTION = {
    "amount": 5000,
    "transaction_hour": 2,
    "transaction_day": 5,
    "merchant_category": "electronics",
    "payment_method": "card",
    "device_type": "mobile",
    "customer_age": 25,
    "account_age_days": 120,
    "previous_transactions": 20,
    "failed_attempts": 3,
    "transactions_last_24h": 8,
    "average_transaction_amount": 1500,
    "distance_from_usual_location": 250,
    "is_new_device": 1,
    "is_international": 1,
    "ip_risk_score": 80,
    "velocity_score": 85,
    "previous_fraud_count": 1,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_predict_endpoint_valid_input():
    response = client.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    for field in ("transaction_id", "fraud_probability", "risk_score", "risk_level", "prediction", "reasons"):
        assert field in body


def test_predict_response_risk_score_range():
    response = client.post("/predict", json=VALID_TRANSACTION)
    body = response.json()
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_endpoint_rejects_negative_amount():
    bad = dict(VALID_TRANSACTION)
    bad["amount"] = -100
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_fields():
    response = client.post("/predict", json={"amount": 100})
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_ip_risk_score():
    bad = dict(VALID_TRANSACTION)
    bad["ip_risk_score"] = 150  # out of 0-100 range
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_rejects_out_of_range_hour():
    bad = dict(VALID_TRANSACTION)
    bad["transaction_hour"] = 25
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
