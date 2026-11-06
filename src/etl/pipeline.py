
from pathlib import Path
import numpy as np
import pandas as pd


def generate_transactions(n=1000, seed=42):
    """Generate reproducible synthetic banking transactions."""
    if n <= 0:
        raise ValueError("Number of transactions must be positive")

    rng = np.random.default_rng(seed)

    transactions = pd.DataFrame({
        "transaction_id": [f"TXN-{i:06d}" for i in range(1, n + 1)],
        "account_id": rng.integers(1001, 1201, size=n),
        "transaction_date": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D"),
        "transaction_type": rng.choice(
            ["deposit", "withdrawal", "transfer", "payment"], size=n
        ),
        "amount": np.round(
            rng.lognormal(mean=4, sigma=1, size=n), 2
        ),
    })

    return transactions


def validate_transactions(df):
    """Validate required fields and basic data quality."""
    required = [
        "transaction_id", "account_id",
        "transaction_date", "transaction_type", "amount"
    ]

    if df.empty:
        raise ValueError("Dataset is empty")

    if df[required].isna().any().any():
        raise ValueError("Missing values found")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found")

    if not np.isfinite(df["amount"]).all() or (df["amount"] <= 0).any():
        raise ValueError("Invalid transaction amounts")

    if pd.to_datetime(df["transaction_date"], errors="coerce").isna().any():
        raise ValueError("Invalid transaction dates")

    return df


def run_pipeline():
    raw = generate_transactions()
    clean = validate_transactions(raw).copy()

    clean["transaction_month"] = (
        clean["transaction_date"].dt.strftime("%Y-%m")
    )
    clean["is_high_value"] = clean["amount"] >= 1000

    output = Path("data/processed/transactions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)

    print(f"ETL successful: {len(clean)} synthetic transactions")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    run_pipeline()
