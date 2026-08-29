"""
train.py
--------
Trains and compares Logistic Regression, Random Forest, and (if available)
XGBoost on the synthetic transaction dataset, then saves the best model.

Model selection criterion (documented, not accuracy-driven):
    Because fraud is rare (~3% positive class), accuracy is nearly meaningless
    (a model predicting "not fraud" always would score >96% accuracy).
    We select the model with the highest F2-score (F-beta, beta=2) on the
    held-out test set. F2 weights recall roughly 4x more than precision,
    which matches the business priority stated in the spec: missing real
    fraud (a false negative) is costlier than an extra manual review of a
    legitimate transaction (a false positive) -- but F2 still penalizes a
    model that has *no* precision at all, so it won't pick a model that
    flags almost everything as fraud. ROC-AUC, PR-AUC, F1, precision and
    recall are all still computed and reported for full transparency.

Usage:
    python src/train.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

from preprocessing import build_preprocessor, ALL_FEATURES, TARGET

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python src/data_generator.py` first."
        )
    return pd.read_csv(DATA_PATH)


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "f2_score": round(fbeta_score(y_test, y_pred, beta=2, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']}")
    return metrics


def main():
    print("Loading dataset...")
    df = load_data()

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%")

    # Fit the shared preprocessor ONLY on training data to prevent leakage.
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    candidates = {}

    candidates["logistic_regression"] = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    candidates["random_forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )
    if XGBOOST_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        candidates["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        print("XGBoost not available - skipping (install xgboost to include it).")

    results = []
    fitted_models = {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train_t, y_train)
        elapsed = time.time() - t0
        print(f"Trained {name} in {elapsed:.1f}s")
        metrics = evaluate_model(name, model, X_test_t, y_test)
        metrics["train_time_seconds"] = round(elapsed, 2)
        results.append(metrics)
        fitted_models[name] = model

    # Model selection: highest F2-score (recall-weighted, imbalance-aware)
    best = max(results, key=lambda m: m["f2_score"])
    best_name = best["model"]
    best_model = fitted_models[best_name]
    print(f"\n>>> Selected best model: {best_name} (F2={best['f2_score']}, "
          f"recall={best['recall']}, precision={best['precision']}, PR-AUC={best['pr_auc']})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "fraud_model.joblib"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "scaler.joblib"))

    metadata = {
        "selected_model": best_name,
        "selection_criterion": "highest F2-score (recall weighted ~4x precision) on the "
                                "held-out test set, because the fraud class is rare (~3%), "
                                "accuracy is not a meaningful metric for imbalanced "
                                "classification, and missing real fraud is costlier than "
                                "an extra manual review.",
        "random_state": RANDOM_STATE,
        "features": ALL_FEATURES,
        "target": TARGET,
        "xgboost_available": XGBOOST_AVAILABLE,
        "risk_thresholds": {"low_max": 30, "medium_max": 70},
        "all_model_results": results,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(df),
        "dataset_fraud_rate_pct": round(float(y.mean() * 100), 3),
        "dataset_is_synthetic": True,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model -> {MODELS_DIR}/fraud_model.joblib")
    print(f"Saved preprocessor -> {MODELS_DIR}/scaler.joblib")
    print(f"Saved metadata -> {MODELS_DIR}/model_metadata.json")


if __name__ == "__main__":
    main()
