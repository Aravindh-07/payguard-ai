"""
demo_transactions.py
---------------------
Runs a handful of illustrative example transactions through the trained
PayGuard AI model and prints the results. These examples are picked from
real rows of the generated dataset (not hardcoded outcomes) so the printed
risk levels are genuinely produced by the trained model, not asserted.

Usage:
    python src/demo_transactions.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.append(os.path.dirname(__file__))

from predictor import get_predictor  # noqa: E402
from preprocessing import ALL_FEATURES  # noqa: E402

import pandas as pd  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")


def find_examples(predictor, df: pd.DataFrame):
    """Score the whole dataset once and pick one clearly-low, one mid-range,
    and one clearly-high scoring transaction to use as demo examples."""
    X = predictor.preprocessor.transform(df[ALL_FEATURES])
    proba = predictor.model.predict_proba(X)[:, 1]
    df = df.copy()
    df["_proba"] = proba

    low_candidates = df[(df["_proba"] < 0.05) & (df["fraud_label"] == 0)]
    med_candidates = df[(df["_proba"] > 0.40) & (df["_proba"] < 0.60)]
    high_candidates = df[df["_proba"] > 0.85]

    examples = {}
    examples["Safe transaction"] = (
        low_candidates.iloc[0] if len(low_candidates) else df.sort_values("_proba").iloc[0]
    )
    examples["Medium-risk transaction"] = (
        med_candidates.iloc[0] if len(med_candidates)
        else df.iloc[(df["_proba"] - 0.5).abs().argsort()[:1]].iloc[0]
    )
    examples["Suspicious transaction"] = (
        high_candidates.sort_values("_proba", ascending=False).iloc[0] if len(high_candidates)
        else df.sort_values("_proba", ascending=False).iloc[0]
    )
    return examples


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run `python src/data_generator.py` first.")
        sys.exit(1)

    predictor = get_predictor()
    df = pd.read_csv(DATA_PATH)
    examples = find_examples(predictor, df)

    print("=" * 70)
    print("PayGuard AI — Demo Transactions")
    print("(Selected from the synthetic dataset; predictions are live, not hardcoded)")
    print("=" * 70)

    for label, row in examples.items():
        transaction = {f: row[f] for f in ALL_FEATURES}
        result = predictor.predict_transaction(transaction)

        print(f"\n### {label}  (source: {row['transaction_id']}) ###")
        print(f"  Amount: {row['amount']:.2f} | Merchant: {row['merchant_category']} | "
              f"Payment: {row['payment_method']} | Device: {row['device_type']}")
        print(f"  Fraud Probability : {result['fraud_probability']*100:.1f}%")
        print(f"  Risk Score        : {result['risk_score']} / 100")
        print(f"  Risk Level        : {result['risk_level']} RISK")
        print(f"  Decision          : {result['prediction']}")
        print("  Why?")
        for reason in result["reasons"]:
            print(f"    - {reason}")

    print("\n" + "=" * 70)
    print("Done. These results came directly from the trained model's predict_proba().")


if __name__ == "__main__":
    main()
