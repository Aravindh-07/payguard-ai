"""
api.py
------
FastAPI backend for PayGuard AI.

Endpoints:
    POST /predict  - score a single transaction
    GET  /health    - liveness check
    GET  /          - basic service info

Run:
    uvicorn app.api:app --reload
"""

import os
import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.append(os.path.dirname(__file__))
from predictor import get_predictor, ModelNotFoundError, predict_transaction  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "payguard.db")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Warm-load the model so the first real request isn't slow, and fail fast
    # with a clear error in the logs if artifacts are missing.
    try:
        get_predictor()
    except ModelNotFoundError as e:
        print(f"WARNING: {e}")
    yield


app = FastAPI(
    title="PayGuard AI",
    description="Prototype payment fraud detection & risk scoring API. "
                 "Educational project — not connected to Razorpay or any live payment system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Database setup
# --------------------------------------------------------------------------
@contextmanager
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                amount REAL,
                merchant_category TEXT,
                payment_method TEXT,
                device_type TEXT,
                fraud_probability REAL,
                risk_score INTEGER,
                risk_level TEXT,
                prediction TEXT,
                raw_input TEXT
            )
        """)
        conn.commit()


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_hour: int = Field(..., ge=0, le=23)
    transaction_day: int = Field(0, ge=0, le=6, description="0=Mon ... 6=Sun")
    merchant_category: str
    payment_method: str
    device_type: str
    customer_age: int = Field(..., ge=13, le=120)
    account_age_days: int = Field(..., ge=0)
    previous_transactions: int = Field(..., ge=0)
    failed_attempts: int = Field(0, ge=0)
    transactions_last_24h: int = Field(0, ge=0)
    average_transaction_amount: float = Field(..., gt=0)
    distance_from_usual_location: float = Field(0, ge=0)
    is_new_device: int = Field(..., ge=0, le=1)
    is_international: int = Field(..., ge=0, le=1)
    ip_risk_score: float = Field(..., ge=0, le=100)
    velocity_score: float = Field(..., ge=0, le=100)
    previous_fraud_count: int = Field(0, ge=0)

    @field_validator("merchant_category", "payment_method", "device_type")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip().lower()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class TransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: int
    risk_level: str
    prediction: str
    reasons: list[str]


class HealthResponse(BaseModel):
    status: str


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "PayGuard AI",
        "description": "Prototype fraud detection API. Not affiliated with Razorpay.",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=TransactionResponse, tags=["fraud"])
def predict(transaction: TransactionRequest):
    try:
        result = predict_transaction(transaction.model_dump())
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    txn_id = f"TXN{uuid.uuid4().hex[:10].upper()}"

    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO transactions
                   (id, timestamp, amount, merchant_category, payment_method, device_type,
                    fraud_probability, risk_score, risk_level, prediction, raw_input)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    txn_id, datetime.now(timezone.utc).isoformat(), transaction.amount,
                    transaction.merchant_category, transaction.payment_method,
                    transaction.device_type, result["fraud_probability"], result["risk_score"],
                    result["risk_level"], result["prediction"], transaction.model_dump_json(),
                ),
            )
            conn.commit()
    except sqlite3.Error as e:
        # Logging failure shouldn't break the prediction response
        print(f"WARNING: failed to log transaction: {e}")

    return {
        "transaction_id": txn_id,
        "fraud_probability": result["fraud_probability"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "prediction": result["prediction"],
        "reasons": result["reasons"],
    }
