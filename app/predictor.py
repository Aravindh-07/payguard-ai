"""
predictor.py
------------
Core prediction service for PayGuard AI. Loads the trained model and
preprocessor once, and exposes `predict_transaction()` which is used by
both the FastAPI backend and the Streamlit dashboard so there is a single
source of truth for scoring logic.
"""

import json
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import ALL_FEATURES  # noqa: E402
from explain import explain_transaction  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

DEFAULT_THRESHOLDS = {"low_max": 30, "medium_max": 70}


class ModelNotFoundError(RuntimeError):
    """Raised when the trained model artifacts are missing."""


class FraudPredictor:
    """Wraps the trained model + preprocessor and exposes a stable prediction API."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        self._load()

    def _load(self):
        missing = [p for p in (MODEL_PATH, PREPROCESSOR_PATH) if not os.path.exists(p)]
        if missing:
            raise ModelNotFoundError(
                "Model artifacts not found: "
                + ", ".join(missing)
                + ". Run `python src/data_generator.py` then `python src/train.py` first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                self.metadata = json.load(f)
            self.thresholds = self.metadata.get("risk_thresholds", DEFAULT_THRESHOLDS)

    @staticmethod
    def score_to_level(score: int, thresholds: dict = None) -> str:
        thresholds = thresholds or DEFAULT_THRESHOLDS
        if score <= thresholds["low_max"]:
            return "LOW"
        if score <= thresholds["medium_max"]:
            return "MEDIUM"
        return "HIGH"

    def _validate_and_frame(self, transaction: dict) -> pd.DataFrame:
        missing = [f for f in ALL_FEATURES if f not in transaction]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")
        row = {f: transaction[f] for f in ALL_FEATURES}
        return pd.DataFrame([row])

    def predict_transaction(self, transaction: dict) -> dict:
        """Run a full prediction pipeline on a single transaction dict.

        Returns a dict with fraud_probability, risk_score, risk_level,
        prediction, and reasons — never a hardcoded/random value; everything
        is derived from the trained model's actual output probability.
        """
        df = self._validate_and_frame(transaction)
        X = self.preprocessor.transform(df)

        fraud_probability = float(self.model.predict_proba(X)[0, 1])
        risk_score = int(round(fraud_probability * 100))
        risk_score = max(0, min(100, risk_score))
        risk_level = self.score_to_level(risk_score, self.thresholds)
        prediction = "Potential Fraud" if risk_score > self.thresholds["low_max"] else "Legitimate"
        # A stricter "hard fraud" flag also available from raw model.predict, if desired:
        model_predicted_fraud = bool(self.model.predict(X)[0])

        reasons = explain_transaction(transaction)

        return {
            "fraud_probability": round(fraud_probability, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "prediction": prediction,
            "model_predicted_fraud": model_predicted_fraud,
            "reasons": reasons,
        }


_predictor_instance = None


def get_predictor() -> FraudPredictor:
    """Lazily-instantiated singleton so the model is loaded only once per process."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor()
    return _predictor_instance


def predict_transaction(transaction: dict) -> dict:
    """Convenience module-level function matching the spec's requested signature."""
    return get_predictor().predict_transaction(transaction)
