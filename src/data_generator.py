"""
data_generator.py
------------------
Generates a realistic SYNTHETIC payment-transaction dataset for PayGuard AI.

IMPORTANT: This data is entirely synthetic. It is NOT real banking data and
is NOT sourced from Razorpay or any payment processor. It is generated
programmatically to resemble the statistical shape of real fraud data
(class imbalance, overlapping distributions, noisy labels) so that the
downstream ML pipeline has a realistic problem to solve.

Run directly to write data/transactions.csv:
    python src/data_generator.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TRANSACTIONS = 50_000

MERCHANT_CATEGORIES = [
    "electronics", "grocery", "fashion", "travel", "food_delivery",
    "utilities", "entertainment", "healthcare", "education", "gaming",
]
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "emi"]
DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n: int = N_TRANSACTIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate n synthetic transactions with realistic, non-separable fraud labels."""
    rng = np.random.default_rng(seed)

    transaction_id = [f"TXN{100000 + i}" for i in range(n)]

    # --- Base behavioural features -----------------------------------
    customer_age = rng.integers(18, 75, size=n)
    account_age_days = rng.integers(1, 3650, size=n)
    previous_transactions = rng.poisson(lam=25, size=n)
    previous_transactions = np.clip(previous_transactions, 0, None)

    # Average historical spend per customer (log-normal, realistic skew)
    average_transaction_amount = rng.lognormal(mean=6.5, sigma=0.9, size=n)
    average_transaction_amount = np.clip(average_transaction_amount, 50, 100_000)

    # Current transaction amount: usually close to the customer's average,
    # but sometimes wildly different (which is one fraud signal, not a rule).
    amount_multiplier = rng.lognormal(mean=0.0, sigma=0.7, size=n)
    amount = average_transaction_amount * amount_multiplier
    amount = np.clip(amount, 10, 500_000)

    transaction_hour = rng.integers(0, 24, size=n)
    transaction_day = rng.integers(0, 7, size=n)  # 0=Mon ... 6=Sun

    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n)
    payment_method = rng.choice(PAYMENT_METHODS, size=n)
    device_type = rng.choice(DEVICE_TYPES, size=n)

    failed_attempts = rng.poisson(lam=0.3, size=n)
    failed_attempts = np.clip(failed_attempts, 0, 10)

    transactions_last_24h = rng.poisson(lam=1.5, size=n)
    transactions_last_24h = np.clip(transactions_last_24h, 0, 40)

    distance_from_usual_location = rng.exponential(scale=15, size=n)  # km
    distance_from_usual_location = np.clip(distance_from_usual_location, 0, 3000)

    is_new_device = rng.binomial(1, 0.12, size=n)
    is_international = rng.binomial(1, 0.06, size=n)

    ip_risk_score = np.clip(rng.normal(loc=20, scale=15, size=n), 0, 100)
    velocity_score = np.clip(rng.normal(loc=15, scale=12, size=n), 0, 100)

    previous_fraud_count = rng.choice(
        [0, 1, 2, 3], size=n, p=[0.94, 0.045, 0.01, 0.005]
    )

    # --- Missing values (a small amount, to force real preprocessing) --
    missing_mask = rng.random(n) < 0.01
    distance_from_usual_location = distance_from_usual_location.astype(float)
    distance_from_usual_location[missing_mask] = np.nan

    # --- Latent fraud "risk logit" -------------------------------------
    # This is a weighted combination of suspicious signals. It deliberately
    # does NOT perfectly separate classes: we add noise and only push the
    # probability, not the label, so overlap between classes is guaranteed.
    z = (
        -4.5
        + 0.9 * (amount_multiplier > 3.0)
        + 0.000006 * amount
        + 1.6 * is_new_device
        + 1.3 * is_international
        + 0.028 * ip_risk_score
        + 0.026 * velocity_score
        + 0.55 * failed_attempts
        + 0.18 * transactions_last_24h
        + 0.012 * np.nan_to_num(distance_from_usual_location, nan=0.0)
        + 1.1 * previous_fraud_count
        + 0.7 * ((transaction_hour >= 0) & (transaction_hour <= 4))
        - 0.01 * (account_age_days / 30.0)
        - 0.02 * previous_transactions
    )

    # Random noise so the problem is not perfectly separable
    z += rng.normal(loc=0.0, scale=1.1, size=n)

    fraud_probability_latent = _sigmoid(z)
    fraud_label = rng.binomial(1, fraud_probability_latent)

    # Enforce realistic class imbalance (~3-4% fraud) by rare down-sampling
    # of "accidental" fraud labels that occurred from noise alone at low risk.
    low_risk_but_flagged = (fraud_label == 1) & (fraud_probability_latent < 0.35)
    drop_idx = np.where(low_risk_but_flagged)[0]
    if len(drop_idx) > 0:
        keep = rng.random(len(drop_idx)) < 0.15  # keep only 15% of these as noisy labels
        drop_idx = drop_idx[~keep]
        fraud_label[drop_idx] = 0

    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "amount": np.round(amount, 2),
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "device_type": device_type,
        "customer_age": customer_age,
        "account_age_days": account_age_days,
        "previous_transactions": previous_transactions,
        "failed_attempts": failed_attempts,
        "transactions_last_24h": transactions_last_24h,
        "average_transaction_amount": np.round(average_transaction_amount, 2),
        "distance_from_usual_location": np.round(distance_from_usual_location, 2),
        "is_new_device": is_new_device,
        "is_international": is_international,
        "ip_risk_score": np.round(ip_risk_score, 2),
        "velocity_score": np.round(velocity_score, 2),
        "previous_fraud_count": previous_fraud_count,
        "fraud_label": fraud_label,
    })

    return df


def main():
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transactions.csv")
    df.to_csv(out_path, index=False)

    fraud_rate = df["fraud_label"].mean() * 100
    print(f"Generated {len(df):,} synthetic transactions -> {out_path}")
    print(f"Fraud rate: {fraud_rate:.2f}% ({df['fraud_label'].sum():,} fraudulent)")
    print("NOTE: This dataset is entirely synthetic and randomly generated.")


if __name__ == "__main__":
    main()
