"""
dashboard.py
------------
PayGuard AI - Streamlit dashboard.
Run:
    streamlit run app/dashboard.py
"""

import os
import sys
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predictor import get_predictor, ModelNotFoundError  # noqa: E402


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "transactions.csv"
)

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "payguard.db"
)


MERCHANT_CATEGORIES = [
    "electronics",
    "grocery",
    "fashion",
    "travel",
    "food_delivery",
    "utilities",
    "entertainment",
    "healthcare",
    "education",
    "gaming",
]

PAYMENT_METHODS = [
    "card",
    "upi",
    "netbanking",
    "wallet",
    "emi",
]

DEVICE_TYPES = [
    "mobile",
    "desktop",
    "tablet",
    "pos",
]


st.set_page_config(
    page_title="PayGuard AI",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
)


# --------------------------------------------------------------------------
# Data / model loading
# --------------------------------------------------------------------------

@st.cache_data
def load_transactions() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_predictor():
    return get_predictor()


def score_dataset_sample(
    df: pd.DataFrame,
    predictor,
    n: int = 3000
) -> pd.DataFrame:

    """
    Score a sample of the dataset using the trained model.
    """

    sample = df.sample(
        min(n, len(df)),
        random_state=42
    ).copy()

    from preprocessing import ALL_FEATURES

    X = predictor.preprocessor.transform(
        sample[ALL_FEATURES]
    )

    proba = predictor.model.predict_proba(X)[:, 1]

    sample["risk_score"] = (
        proba * 100
    ).round().astype(int).clip(0, 100)

    sample["risk_level"] = sample["risk_score"].apply(
        lambda s: predictor.score_to_level(
            s,
            predictor.thresholds
        )
    )

    sample["prediction"] = sample["risk_score"].apply(
        lambda s:
        "Potential Fraud"
        if s > predictor.thresholds["low_max"]
        else "Legitimate"
    )

    return sample


def get_logged_transactions() -> pd.DataFrame:

    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    try:

        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY timestamp DESC",
            conn
        )

        conn.close()

        return df

    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.title("\U0001F6E1\uFE0F PayGuard AI")

st.sidebar.caption(
    "Intelligent Payment Fraud Detection & Risk Scoring"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Transaction Analyzer",
        "Transaction Table",
        "About / Model Info",
    ],
)

st.sidebar.markdown("---")


# --------------------------------------------------------------------------
# Load model and dataset
# --------------------------------------------------------------------------

try:

    predictor = load_predictor()
    model_error = None

except ModelNotFoundError as e:

    predictor = None
    model_error = str(e)


df = load_transactions()


if model_error:

    st.error(
        f"Model not loaded: {model_error}"
    )

    st.stop()


if df.empty:

    st.warning(
        "No dataset found at `data/transactions.csv`. "
        "Run `python src/data_generator.py` to generate it."
    )

    st.stop()


# --------------------------------------------------------------------------
# PAGE: Overview
# --------------------------------------------------------------------------

if page == "Overview":

    st.title("\U0001F6E1\uFE0F PayGuard AI")

    st.caption(
        "Intelligent Payment Fraud Detection & Risk Scoring"
    )

    scored = score_dataset_sample(
        df,
        predictor,
        n=5000
    )

    total_txns = len(df)

    fraud_txns = int(
        df["fraud_label"].sum()
    )

    fraud_rate = (
        fraud_txns /
        total_txns *
        100
    )

    avg_amount = df["amount"].mean()

    high_risk_count = int(
        (scored["risk_level"] == "HIGH").sum()
    )


    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Transactions",
        f"{total_txns:,}"
    )

    c2.metric(
        "Fraudulent Transactions",
        f"{fraud_txns:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    c4.metric(
        "Avg Transaction Amount",
        f"\u20B9{avg_amount:,.0f}"
    )

    c5.metric(
        "High Risk (sampled)",
        f"{high_risk_count:,}"
    )


    st.markdown("---")


    col1, col2 = st.columns(2)


    with col1:

        fraud_counts = (
            df["fraud_label"]
            .map({
                0: "Legitimate",
                1: "Fraud"
            })
            .value_counts()
        )

        fig = px.pie(
            values=fraud_counts.values,
            names=fraud_counts.index,
            title="Fraud vs Legitimate Transactions",
            color=fraud_counts.index,
            color_discrete_map={
                "Legitimate": "#2E7D32",
                "Fraud": "#C62828"
            },
            hole=0.45,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    with col2:

        risk_dist = (
            scored["risk_level"]
            .value_counts()
            .reindex([
                "LOW",
                "MEDIUM",
                "HIGH"
            ])
            .fillna(0)
        )

        fig = px.bar(
            x=risk_dist.index,
            y=risk_dist.values,
            title="Risk-Level Distribution",
            labels={
                "x": "Risk Level",
                "y": "Count"
            },
            color=risk_dist.index,
            color_discrete_map={
                "LOW": "#2E7D32",
                "MEDIUM": "#F9A825",
                "HIGH": "#C62828"
            },
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    col3, col4 = st.columns(2)


    with col3:

        by_method = (
            df.groupby("payment_method")["fraud_label"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        fig = px.bar(
            x=by_method.index,
            y=by_method.values,
            title="Fraud Rate by Payment Method (%)",
            labels={
                "x": "Payment Method",
                "y": "Fraud Rate (%)"
            },
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    with col4:

        by_cat = (
            df.groupby("merchant_category")["fraud_label"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        fig = px.bar(
            x=by_cat.index,
            y=by_cat.values,
            title="Fraud Rate by Merchant Category (%)",
            labels={
                "x": "Merchant Category",
                "y": "Fraud Rate (%)"
            },
        )

        fig.update_xaxes(
            tickangle=-30
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    col5, col6 = st.columns(2)


    with col5:

        by_hour = (
            df.groupby("transaction_hour")["fraud_label"]
            .mean()
            * 100
        )

        fig = px.line(
            x=by_hour.index,
            y=by_hour.values,
            title="Fraud Rate by Hour of Day (%)",
            labels={
                "x": "Hour",
                "y": "Fraud Rate (%)"
            },
            markers=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    with col6:

        fig = px.histogram(
            df,
            x="amount",
            nbins=60,
            title="Transaction Amount Distribution",
            labels={
                "amount": "Amount"
            },
            log_y=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# --------------------------------------------------------------------------
# PAGE: Transaction Analyzer
# --------------------------------------------------------------------------

elif page == "Transaction Analyzer":

    st.title("\U0001F50D Transaction Analyzer")

    st.caption(
        "Enter transaction details to get an instant fraud risk assessment."
    )


    with st.form("transaction_form"):

        c1, c2, c3 = st.columns(3)


        # --------------------------------------------------------------
        # Column 1
        # --------------------------------------------------------------

        with c1:

            amount = st.number_input(
                "Transaction Amount (\u20B9)",
                min_value=1.0,
                value=5000.0,
                step=100.0
            )

            transaction_hour = st.slider(
                "Transaction Hour (0-23)",
                0,
                23,
                14
            )

            transaction_day = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda d: [
                    "Mon",
                    "Tue",
                    "Wed",
                    "Thu",
                    "Fri",
                    "Sat",
                    "Sun"
                ][d],
            )

            merchant_category = st.selectbox(
                "Merchant Category",
                MERCHANT_CATEGORIES
            )

            payment_method = st.selectbox(
                "Payment Method",
                PAYMENT_METHODS
            )

            device_type = st.selectbox(
                "Device Type",
                DEVICE_TYPES
            )


        # --------------------------------------------------------------
        # Column 2
        # --------------------------------------------------------------

        with c2:

            customer_age = st.number_input(
                "Customer Age",
                min_value=13,
                max_value=100,
                value=32
            )

            account_age_days = st.number_input(
                "Account Age (days)",
                min_value=0,
                value=365
            )

            previous_transactions = st.number_input(
                "Previous Transactions",
                min_value=0,
                value=25
            )

            failed_attempts = st.number_input(
                "Failed Attempts (recent)",
                min_value=0,
                value=0
            )

            transactions_last_24h = st.number_input(
                "Transactions in Last 24h",
                min_value=0,
                value=1
            )

            average_transaction_amount = st.number_input(
                "Customer's Average Transaction Amount (\u20B9)",
                min_value=1.0,
                value=1500.0
            )


        # --------------------------------------------------------------
        # Column 3
        # --------------------------------------------------------------

        with c3:

            distance_from_usual_location = st.number_input(
                "Distance From Usual Location (km)",
                min_value=0.0,
                value=5.0
            )

            is_new_device = (
                st.selectbox(
                    "New/Unrecognized Device?",
                    ["No", "Yes"]
                )
                == "Yes"
            )

            is_international = (
                st.selectbox(
                    "International Transaction?",
                    ["No", "Yes"]
                )
                == "Yes"
            )

            ip_risk_score = st.slider(
                "IP Risk Score (0-100)",
                0,
                100,
                15
            )

            velocity_score = st.slider(
                "Velocity Score (0-100)",
                0,
                100,
                15
            )

            previous_fraud_count = st.number_input(
                "Previous Fraud Count",
                min_value=0,
                value=0
            )


        submitted = st.form_submit_button(
            "Analyze Transaction",
            width="stretch",
            type="primary"
        )


    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    if submitted:

        transaction = {

            "amount": amount,

            "transaction_hour":
                transaction_hour,

            "transaction_day":
                transaction_day,

            "merchant_category":
                merchant_category,

            "payment_method":
                payment_method,

            "device_type":
                device_type,

            "customer_age":
                customer_age,

            "account_age_days":
                account_age_days,

            "previous_transactions":
                previous_transactions,

            "failed_attempts":
                failed_attempts,

            "transactions_last_24h":
                transactions_last_24h,

            "average_transaction_amount":
                average_transaction_amount,

            "distance_from_usual_location":
                distance_from_usual_location,

            "is_new_device":
                int(is_new_device),

            "is_international":
                int(is_international),

            "ip_risk_score":
                ip_risk_score,

            "velocity_score":
                velocity_score,

            "previous_fraud_count":
                previous_fraud_count,
        }


        try:

            result = predictor.predict_transaction(
                transaction
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()


        st.markdown("---")

        st.subheader("Prediction Result")


        level_colors = {
            "LOW": "#2E7D32",
            "MEDIUM": "#F9A825",
            "HIGH": "#C62828"
        }

        color = level_colors.get(
            result["risk_level"],
            "#616161"
        )


        r1, r2, r3, r4 = st.columns(4)


        r1.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.1f}%"
        )


        r2.metric(
            "Risk Score",
            f"{result['risk_score']} / 100"
        )


        r3.markdown(
            f"""
            <div style='
                padding:0.6em;
                border-radius:8px;
                background:{color};
                color:white;
                text-align:center;
                font-weight:bold;
            '>
                {result['risk_level']} RISK
            </div>
            """,
            unsafe_allow_html=True,
        )


        r4.metric(
            "Decision",
            result["prediction"]
        )


        st.progress(
            min(
                result["risk_score"],
                100
            ) / 100
        )


        st.markdown("#### Why?")


        for reason in result["reasons"]:

            st.markdown(
                f"- {reason}"
            )


        # --------------------------------------------------------------
        # Log transaction
        # --------------------------------------------------------------

        try:

            import uuid

            conn = sqlite3.connect(
                DB_PATH
            )


            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
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
                """
            )


            conn.execute(
                """
                INSERT OR REPLACE INTO transactions
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"TXN{uuid.uuid4().hex[:10].upper()}",
                    datetime.now(timezone.utc).isoformat(),
                    amount,
                    merchant_category,
                    payment_method,
                    device_type,
                    result["fraud_probability"],
                    result["risk_score"],
                    result["risk_level"],
                    result["prediction"],
                    str(transaction),
                ),
            )


            conn.commit()
            conn.close()


        except Exception:
            pass


# --------------------------------------------------------------------------
# PAGE: Transaction Table
# --------------------------------------------------------------------------

elif page == "Transaction Table":

    st.title("\U0001F4CB Transaction Table")


    scored = score_dataset_sample(
        df,
        predictor,
        n=3000
    )


    fc1, fc2, fc3, fc4 = st.columns(4)


    with fc1:

        risk_filter = st.multiselect(
            "Risk Level",
            ["LOW", "MEDIUM", "HIGH"],
            default=[
                "LOW",
                "MEDIUM",
                "HIGH"
            ]
        )


    with fc2:

        pred_filter = st.multiselect(
            "Prediction",
            [
                "Legitimate",
                "Potential Fraud"
            ],
            default=[
                "Legitimate",
                "Potential Fraud"
            ]
        )


    with fc3:

        method_filter = st.multiselect(
            "Payment Method",
            PAYMENT_METHODS,
            default=PAYMENT_METHODS
        )


    with fc4:

        cat_filter = st.multiselect(
            "Merchant Category",
            MERCHANT_CATEGORIES,
            default=MERCHANT_CATEGORIES
        )


    filtered = scored[
        scored["risk_level"].isin(risk_filter)
        & scored["prediction"].isin(pred_filter)
        & scored["payment_method"].isin(method_filter)
        & scored["merchant_category"].isin(cat_filter)
    ]


    display_cols = [
        "transaction_id",
        "amount",
        "risk_score",
        "risk_level",
        "prediction",
        "payment_method",
        "merchant_category",
        "transaction_hour",
    ]


    st.write(
        f"Showing {len(filtered):,} "
        f"of {len(scored):,} sampled transactions"
    )


    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        width="stretch",
        height=500
    )


    st.markdown("---")


    st.subheader(
        "Recently Analyzed"
    )


    logged = get_logged_transactions()


    if logged.empty:

        st.caption(
            "No transactions logged yet. "
            "Use the Transaction Analyzer or API."
        )

    else:

        st.dataframe(
            logged[
                [
                    "id",
                    "timestamp",
                    "amount",
                    "risk_score",
                    "risk_level",
                    "prediction"
                ]
            ].head(50),
            width="stretch",
        )


# --------------------------------------------------------------------------
# PAGE: About / Model Info
# --------------------------------------------------------------------------

else:

    st.title(
        "\u2139\uFE0F About PayGuard AI"
    )


    st.markdown(
        """
        **PayGuard AI** is an intelligent payment
        fraud detection and risk-scoring system.

        The system uses machine learning to analyze
        transaction patterns and generate fraud
        probability, risk scores, and risk levels.
        """
    )


    if predictor and predictor.metadata:

        st.subheader(
            "Model Details"
        )


        meta = predictor.metadata


        st.write(
            f"**Selected model:** "
            f"`{meta.get('selected_model')}`"
        )


        st.write(
            f"**Selection criterion:** "
            f"{meta.get('selection_criterion')}"
        )


        st.write(
            f"**Trained at:** "
            f"{meta.get('trained_at')}"
        )


        st.write(
            f"**Dataset size:** "
            f"{meta.get('dataset_size'):,} transactions "
            f"({meta.get('dataset_fraud_rate_pct')}% fraud)"
        )


        st.markdown(
            "#### Model Comparison"
        )


        results_df = pd.DataFrame(
            meta.get(
                "all_model_results",
                []
            )
        )


        if not results_df.empty:

            cols = [
                c
                for c in [
                    "model",
                    "precision",
                    "recall",
                    "f1_score",
                    "f2_score",
                    "roc_auc",
                    "pr_auc"
                ]
                if c in results_df.columns
            ]


            st.dataframe(
                results_df[cols],
                width="stretch"
            )


        st.markdown(
            "#### Risk Thresholds"
        )


        st.write(
            meta.get(
                "risk_thresholds"
            )
        )
