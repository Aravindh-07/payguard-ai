"""
explain.py
----------
Transparent, rule-based explainability layer for PayGuard AI.

This is NOT a SHAP explanation. It is a lightweight, human-readable
"reason engine" that inspects the raw transaction fields and flags the
factors that are statistically associated with higher fraud risk (based on
the same signals the model was trained to weight heavily). It is documented
here as a prototype explainability layer, not a formal feature-attribution
method like SHAP or LIME.

If you want true model-attribution explanations, integrate SHAP against the
saved model in `models/fraud_model.joblib` — see the README's "Future
Improvements" section.
"""

from typing import Dict, List

# Each rule: (condition_fn, reason_text, severity_weight)
# severity_weight is only used to order the reasons shown to the user.
RULES = [
    (lambda t: t["amount"] > 3 * max(t["average_transaction_amount"], 1),
     "Transaction amount is far above the customer's usual spending", 5),
    (lambda t: t["amount"] > 50000,
     "Unusually high transaction amount", 4),
    (lambda t: t["is_new_device"] == 1,
     "Transaction initiated from a new/unrecognized device", 4),
    (lambda t: t["is_international"] == 1,
     "International transaction", 3),
    (lambda t: t["ip_risk_score"] >= 60,
     "High IP risk score", 5),
    (lambda t: t["velocity_score"] >= 60,
     "High transaction velocity score", 5),
    (lambda t: t["failed_attempts"] >= 3,
     "Multiple recent failed payment attempts", 4),
    (lambda t: t["transactions_last_24h"] >= 8,
     "Unusually many transactions in the last 24 hours", 3),
    (lambda t: (t.get("distance_from_usual_location") or 0) >= 200,
     "Transaction location far from customer's usual location", 3),
    (lambda t: t["previous_fraud_count"] >= 1,
     "Customer has a prior fraud history", 5),
    (lambda t: t["account_age_days"] < 30,
     "Account is newly created (less than 30 days old)", 2),
    (lambda t: t["transaction_hour"] in (0, 1, 2, 3, 4),
     "Transaction occurred during unusual late-night hours", 2),
    (lambda t: t["previous_transactions"] < 3,
     "Customer has very limited transaction history", 2),
]


def explain_transaction(transaction: Dict, top_n: int = 6) -> List[str]:
    """Return a ranked list of human-readable reasons a transaction looks risky.

    `transaction` should contain the raw (pre-preprocessing) feature values.
    Missing optional keys are treated as "not triggered" rather than errors.
    """
    triggered = []
    for condition, reason, weight in RULES:
        try:
            if condition(transaction):
                triggered.append((weight, reason))
        except (KeyError, TypeError):
            continue

    triggered.sort(key=lambda x: x[0], reverse=True)
    reasons = [reason for _, reason in triggered[:top_n]]

    if not reasons:
        reasons = ["No major risk factors detected; transaction pattern looks typical."]

    return reasons
