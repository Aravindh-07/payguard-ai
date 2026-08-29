"""
preprocessing.py
-----------------
Builds the Scikit-learn preprocessing pipeline (ColumnTransformer) used for
both training and inference. Keeping this in one place guarantees that the
exact same transformation is applied at prediction time as at training time,
which prevents train/serve skew and data leakage.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "amount",
    "transaction_hour",
    "transaction_day",
    "customer_age",
    "account_age_days",
    "previous_transactions",
    "failed_attempts",
    "transactions_last_24h",
    "average_transaction_amount",
    "distance_from_usual_location",
    "is_new_device",
    "is_international",
    "ip_risk_score",
    "velocity_score",
    "previous_fraud_count",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "payment_method",
    "device_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "fraud_label"


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for numeric + categorical features.

    - Numeric: median imputation (robust to outliers/missing) + standard scaling.
    - Categorical: most-frequent imputation + one-hot encoding.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the flattened feature names after transformation (for feature importance)."""
    return list(preprocessor.get_feature_names_out())
