
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv(override=True)

MODEL_PATH = Path("models/random_forest_model.joblib")

FEATURES = [
    "transaction_type",
    "amount",
    "sender_old_balance",
    "recipient_old_balance",
]

NUMERIC_FEATURES = [
    "amount",
    "sender_old_balance",
    "recipient_old_balance",
]

CATEGORICAL_FEATURES = ["transaction_type"]


def get_engine():
    url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv(
            "DB_NAME", "banking_transaction_intelligence"
        ),
    )
    return create_engine(url)


def load_data(engine):
    query = text("""
        SELECT
            step,
            transaction_type,
            amount,
            sender_old_balance,
            recipient_old_balance,
            is_fraud
        FROM raw_transactions
        WHERE transaction_type IN ('TRANSFER', 'CASH_OUT')
        ORDER BY step, source_row_number
    """)

    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def train():
    print("Loading transaction data...", flush=True)

    engine = get_engine()
    try:
        df = load_data(engine)
    finally:
        engine.dispose()

    print(f"Loaded {len(df):,} transactions", flush=True)

    if df.empty:
        raise ValueError("No training data found")

    # Keep the earliest 80% of simulation steps for training.
    # Later steps are held out for testing.
    train_cutoff = df["step"].quantile(0.60)
    test_cutoff = df["step"].quantile(0.80)

    train_df = df[df["step"] <= train_cutoff].copy()
   
    validation_df = df[
        (df["step"] > train_cutoff)
        & (df["step"] <= test_cutoff)
    ].copy()

    test_df = df[df["step"] > test_cutoff].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("One or more data splits are empty")

    print(
        f"Training: {len(train_df):,} rows; "
        f"Validation: {len(validation_df):,} rows; "
        f"Testing: {len(test_df):,} rows",
        flush=True,
    )
    X_train = train_df[FEATURES]
    y_train = train_df["is_fraud"].astype(int)

    X_validation = validation_df[FEATURES]
    y_validation = validation_df["is_fraud"].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df["is_fraud"].astype(int)

    if any(
    y.nunique() < 2
    for y in (y_train, y_validation, y_test)
):
        raise ValueError(
            "Both training and test sets must contain "
            "fraud and non-fraud transactions"
        )

    print(
        f"Training: {len(train_df):,} rows; "
        f"Testing: {len(test_df):,} rows",
        flush=True,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
    "classifier",
    RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2,
    ),
),
    ])
    # Use a smaller, reproducible sample for the first Random Forest experiment.
    sample_size = min(300_000, len(X_train))

    sample_indices = X_train.sample(
        n=sample_size,
        random_state=42,
    ).index

    X_train_sample = X_train.loc[sample_indices]
    y_train_sample = y_train.loc[sample_indices]

    print(
        f"Random Forest training sample: {len(X_train_sample):,} rows",
        flush=True,
    )

    print("Training model...", flush=True)
    model.fit(X_train_sample, y_train_sample)

    probabilities = model.predict_proba(X_validation)[:, 1]

    print("\nValidation results at different thresholds:")

    for threshold in [0.50, 0.70, 0.80, 0.90, 0.95]:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"False alarms: {fp:,} | "
            f"Fraud detected: {tp:,}"
        )

    print(
        "\nValidation average precision:",
        round(
            average_precision_score(y_validation, probabilities),
            4,
        ),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    train()
