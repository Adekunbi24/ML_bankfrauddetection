
import os
from pathlib import Path

import joblib
import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ML- Bank Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env", override=True)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "random_forest_model.joblib"
)

FRAUD_THRESHOLD = 0.80


# --------------------------------------------------
# PostgreSQL connection
# --------------------------------------------------

def get_connection_args():
    connection_args = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv(
            "DB_NAME", "banking_transaction_intelligence"
        ),
        "user": os.getenv("DB_USER", "postgres"),
    }

    password = os.getenv("DB_PASSWORD")

    if password:
        connection_args["password"] = password

    return connection_args


# --------------------------------------------------
# Database analytics
# --------------------------------------------------

@st.cache_data(ttl=300)
def get_transaction_metrics():
    query = """
        SELECT
            COUNT(*) AS total_transactions,
            COUNT(*) FILTER (
                WHERE is_fraud = 1
            ) AS fraud_cases
        FROM raw_transactions
    """

    with psycopg.connect(**get_connection_args()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            total_transactions, fraud_cases = cur.fetchone()

    fraud_rate = (
        fraud_cases / total_transactions * 100
        if total_transactions > 0
        else 0
    )

    return total_transactions, fraud_cases, fraud_rate


@st.cache_data(ttl=300)
def get_transaction_type_stats():
    query = """
        SELECT
            transaction_type,
            COUNT(*) AS total_transactions,
            COUNT(*) FILTER (
                WHERE is_fraud = 1
            ) AS fraud_cases
        FROM raw_transactions
        GROUP BY transaction_type
        ORDER BY total_transactions DESC
    """

    with psycopg.connect(**get_connection_args()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "Transaction Type",
            "Total Transactions",
            "Fraud Cases",
        ],
    )


@st.cache_data(ttl=300)
def get_fraud_amount_metrics():
    query = """
        SELECT
            COALESCE(
                SUM(amount) FILTER (WHERE is_fraud = 1),
                0
            ) AS total_fraud_amount,

            AVG(amount) FILTER (
                WHERE is_fraud = 1
            ) AS average_fraud_amount,

            AVG(amount) FILTER (
                WHERE is_fraud = 0
            ) AS average_nonfraud_amount

        FROM raw_transactions
    """

    with psycopg.connect(**get_connection_args()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()

    if result is None:
        return (0.0, 0.0, 0.0)

    return tuple(
        float(value) if value is not None else 0.0
        for value in result
    )


@st.cache_data(ttl=300)
def get_fraud_trends():
    query = """
        SELECT
            step AS "Simulation Step",
            COUNT(*) FILTER (
                WHERE is_fraud = 1
            ) AS "Fraud Cases"
        FROM raw_transactions
        GROUP BY step
        ORDER BY step
    """

    with psycopg.connect(**get_connection_args()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "Simulation Step",
            "Fraud Cases",
        ],
    )


# --------------------------------------------------
# Machine learning model
# --------------------------------------------------

@st.cache_resource
def load_fraud_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ML- Bank Fraud Detection")
    st.caption("Transaction Risk Intelligence")

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Fraud Screening",
            "Transaction Analytics",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("Synthetic PaySim Dataset")
    st.caption("Machine Learning + PostgreSQL")


# --------------------------------------------------
# Dashboard styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main application */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #142B4A;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    /* Dashboard headings */
    h1, h2, h3 {
        color: #142B4A;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F3F6FA;
        padding: 20px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton button,
    .stFormSubmitButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# --------------------------------------------------
# Dashboard header
# --------------------------------------------------


# --------------------------------------------------
# Dashboard header
# --------------------------------------------------

st.title("ML- Bank Fraud Detection")

st.caption(
    "Machine learning-based fraud screening and "
    "transaction analytics using PostgreSQL."
)

st.caption(
    "Portfolio demonstration using synthetic PaySim data. "
    "No real banking transactions are processed."
)

st.divider()


# --------------------------------------------------
# Page 1: Overview
# --------------------------------------------------

if selected_page == "Overview":

    st.subheader("Dashboard Overview")

    with st.spinner("Loading database metrics..."):
        total, fraud, fraud_rate = get_transaction_metrics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions Analyzed", f"{total:,}")
    col2.metric("Transactions Labeled as Fraud", f"{fraud:,}")
    col3.metric("Percentage Labeled as Fraud", f"{fraud_rate:.4f}%")
    col4.metric("Fraud Alert Threshold", f"{FRAUD_THRESHOLD:.0%}")

    st.caption(
        f"Out of {total:,} simulated transactions, {fraud:,} were "
        f"labeled as fraud in the dataset. That is approximately "
        f"{fraud / total * 10000:.0f} out of every 10,000 transactions."
        if total > 0 else "No transactions available."
    )

    st.divider()

    # ------------------------------------------
    # Model Performance
    # ------------------------------------------

    st.divider()

    st.subheader("How Well Does Our Fraud Detection System Work?")

    st.caption(
        "These results come from testing the model on "
        "simulated banking transactions."
    )

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.metric("Correct Fraud Alerts", "88.36%")
        st.write(
            "About 88 out of every 100 transactions flagged "
            "by the model were labeled as fraud."
        )

    with perf_col2:
        st.metric("Fraud Cases Caught", "68.65%")
        st.write(
            "The model caught about 69 out of every 100 "
            "transactions labeled as fraud."
        )

    st.caption(
        "The model also missed about 31 out of every 100 "
        "fraud cases. Results are based on synthetic PaySim "
        "test data, using an 80% screening threshold. "
        "Performance on real banking transactions is unknown."
    )

    st.divider()
    

    st.subheader("Fraud Amount Analysis")

    with st.spinner("Loading fraud amount metrics..."):
        (
            total_fraud_amount,
            average_fraud_amount,
            average_nonfraud_amount,
        ) = get_fraud_amount_metrics()

    amount_col1, amount_col2, amount_col3 = st.columns(3)

    amount_col1.metric(
        "Total Fraud-Labeled Amount",
        f"{total_fraud_amount:,.2f}",
    )

    amount_col2.metric(
        "Average Fraud Transaction",
        f"{average_fraud_amount:,.2f}",
    )

    amount_col3.metric(
        "Average Nonfraud Transaction",
        f"{average_nonfraud_amount:,.2f}",
    )

    st.caption(
        "Amounts are from the synthetic PaySim dataset. "
        "They are not verified real-world financial losses "
        "or amounts in a confirmed currency."
    )

    st.divider()


    st.subheader("How Fraud Activity Changes Over Time")

    with st.spinner("Loading fraud trends..."):
        fraud_trends = get_fraud_trends()

    fraud_trends = fraud_trends.rename(
        columns={"Simulation Step": "Simulated Hour"}
    )

    st.line_chart(
        fraud_trends,
        x="Simulated Hour",
        y="Fraud Cases",
    )

    st.caption(
        "Each hour represents one hour in a simulated banking system. "
        "The blue line shows how many transactions were labeled as fraud "
        "during each hour. Higher points mean more fraud cases. "
        "This is simulated data, not activity from a real bank."
    )

    st.divider()


    st.subheader("About This Dashboard")


    st.write(
        "This application analyzes synthetic banking transactions "
        "stored in PostgreSQL and uses a trained Random Forest "
        "model to screen proposed transactions for fraud risk."
    )

    st.info(
        "Use the sidebar to screen a transaction or explore "
        "transaction patterns."
    )


# --------------------------------------------------
# Page 2: Fraud Screening
# --------------------------------------------------

elif selected_page == "Fraud Screening":

    st.subheader("Pre-Transaction Fraud Screening")

    st.write(
        "Enter a proposed transaction to check whether "
        "the model flags it for fraud review."
    )

    with st.form("fraud_screening_form"):

        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT"],
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=1000.0,
        )

        sender_balance = st.number_input(
            "Sender's balance before transaction",
            min_value=0.0,
            value=5000.0,
        )

        recipient_balance = st.number_input(
            "Recipient's balance before transaction",
            min_value=0.0,
            value=0.0,
        )

        submitted = st.form_submit_button(
            "Screen Transaction"
        )

    
    if submitted:

        transaction = pd.DataFrame([{
            "transaction_type": transaction_type,
            "amount": amount,
            "sender_old_balance": sender_balance,
            "recipient_old_balance": recipient_balance,
        }])

        with st.spinner("Screening transaction..."):
            model = load_fraud_model()
            fraud_score = float(
                model.predict_proba(transaction)[0, 1]
            )

        # ------------------------------------------
        # Transaction Screening Result
        # ------------------------------------------

        st.divider()
        st.subheader("Transaction Screening Result")

        st.write(
            "The model has reviewed the transaction details "
            "you entered. Here is the screening result."
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Model Risk Score",
                f"{fraud_score:.2%}",
            )

        with result_col2:
            st.metric(
                "Fraud Alert Threshold",
                f"{FRAUD_THRESHOLD:.0%}",
            )

        if fraud_score >= FRAUD_THRESHOLD:
            st.error(
                "FLAG FOR REVIEW — The model identified "
                "this transaction as requiring further review."
            )
        else:
            st.success(
                "NOT FLAGGED — The model did not flag "
                "this transaction for review."
            )

        st.markdown("#### What does this result mean?")

        st.write(
            "The model calculates a risk score based on "
            "the transaction information provided. "
            "Transactions scoring 80% or higher are "
            "flagged for additional review."
        )

        st.caption(
            "This is a demonstration using synthetic PaySim data. "
            "The risk score is not a verified real-world fraud "
            "probability. A transaction that is not flagged "
            "is not guaranteed to be safe. No real payments "
            "are processed."
        )


# --------------------------------------------------
# Page 3: Transaction Analytics
# --------------------------------------------------


elif selected_page == "Transaction Analytics":

    st.subheader("Understanding Banking Transactions")

    st.write(
        "Explore how different types of transactions are used "
        "and where fraud was identified in the simulated banking data."
    )

    # Load transaction data automatically from PostgreSQL
    with st.spinner("Loading transaction analytics..."):
        transaction_stats = get_transaction_type_stats()

    transaction_stats["Fraud Rate (%)"] = (
        transaction_stats["Fraud Cases"]
        / transaction_stats["Total Transactions"]
        * 100
    )

    # ------------------------------------------
    # Chart 1: Transaction Volume
    # ------------------------------------------

    st.markdown("### How Customers Make Transactions")

    st.write(
        "This chart compares the number of transactions "
        "recorded for each transaction type."
    )

    st.bar_chart(
        transaction_stats,
        x="Transaction Type",
        y="Total Transactions",
    )

    st.caption(
        "The taller the bar, the more transactions were "
        "recorded in that category."
    )

    st.divider()

    # ------------------------------------------
    # Chart 2: Fraud Cases
    # ------------------------------------------

    st.markdown("### Where Fraud Was Found")

    st.write(
        "This chart shows how many transactions were labeled "
        "as fraud in each transaction category."
    )

    st.bar_chart(
        transaction_stats,
        x="Transaction Type",
        y="Fraud Cases",
    )

    st.caption(
        "These are fraud labels from the simulated dataset, "
        "not confirmed fraud cases from a real bank."
    )

    st.divider()

    # ------------------------------------------
    # Chart 3: Fraud Rate
    # ------------------------------------------

    st.markdown("### Percentage of Fraud in Each Transaction Type")

    st.write(
        "This chart shows the percentage of transactions "
        "labeled as fraud within each category."
    )

    st.bar_chart(
        transaction_stats,
        x="Transaction Type",
        y="Fraud Rate (%)",
    )

    st.caption(
        "For example, a fraud rate of 0.77% means approximately "
        "77 out of every 10,000 transactions in that category "
        "were labeled as fraud."
    )

    st.divider()

    # ------------------------------------------
    # Transaction Summary
    # ------------------------------------------

    st.markdown("### Transaction Breakdown")

    st.write(
        "The table below summarizes the number of transactions, "
        "fraud cases, and fraud percentages for each category."
    )

    st.dataframe(
        transaction_stats,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "All figures are calculated from the synthetic PaySim "
        "dataset stored in PostgreSQL."
    )
