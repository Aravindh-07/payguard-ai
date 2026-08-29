"""
evaluate.py
-----------
Standalone evaluation script. Loads the saved model + preprocessor and the
full dataset, re-computes metrics, and prints the confusion matrix and
model comparison table stored during training (models/model_metadata.json).

Usage:
    python src/evaluate.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

from preprocessing import ALL_FEATURES, TARGET

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")


def main():
    model_path = os.path.join(MODELS_DIR, "fraud_model.joblib")
    preproc_path = os.path.join(MODELS_DIR, "scaler.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    for p in (model_path, preproc_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}. Run `python src/train.py` first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Selected model: {metadata['selected_model']}")
    print(f"Selection criterion: {metadata['selection_criterion']}\n")

    print("=== Model comparison (from training run) ===")
    for m in metadata["all_model_results"]:
        print(f"  {m['model']:<22} precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"f1={m['f1_score']:.4f}  roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}")

    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=metadata["random_state"]
    )

    X_test_t = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    print("\n=== Re-evaluation of saved model on held-out test split ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")


if __name__ == "__main__":
    main()
