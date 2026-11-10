
import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "dashboard"

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

connection_args = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "banking_transaction_intelligence"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

with psycopg.connect(**connection_args) as conn:
    db_conn = cast(Any, conn)

    metrics = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_transactions,
            COUNT(*) FILTER (WHERE is_fraud = 1) AS fraud_cases
        FROM raw_transactions
        """,
        db_conn,
    ).iloc[0]

    total = int(metrics["total_transactions"])
    fraud = int(metrics["fraud_cases"])

    with open(OUTPUT_DIR / "transaction_metrics.json", "w") as f:
        json.dump({
            "total_transactions": total,
            "fraud_cases": fraud,
            "fraud_rate": fraud / total * 100 if total else 0,
        }, f, indent=2)

    transaction_types = pd.read_sql_query(
        """
        SELECT
            transaction_type AS "Transaction Type",
            COUNT(*) AS "Total Transactions",
            COUNT(*) FILTER (WHERE is_fraud = 1) AS "Fraud Cases"
        FROM raw_transactions
        GROUP BY transaction_type
        ORDER BY "Total Transactions" DESC
        """,
        db_conn,
    )

    transaction_types.to_csv(
        OUTPUT_DIR / "transaction_type_stats.csv", index=False
    )

    amounts = pd.read_sql_query(
        """
        SELECT
            COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0)
                AS total_fraud_amount,
            AVG(amount) FILTER (WHERE is_fraud = 1)
                AS average_fraud_amount,
            AVG(amount) FILTER (WHERE is_fraud = 0)
                AS average_nonfraud_amount
        FROM raw_transactions
        """,
        db_conn,
    ).iloc[0]

    with open(OUTPUT_DIR / "fraud_amount_metrics.json", "w") as f:
        json.dump({
            key: float(amounts[key]) if pd.notna(amounts[key]) else 0.0
            for key in amounts.index
        }, f, indent=2)

    trends = pd.read_sql_query(
        """
        SELECT
            step AS "Simulation Step",
            COUNT(*) FILTER (WHERE is_fraud = 1) AS "Fraud Cases"
        FROM raw_transactions
        GROUP BY step
        ORDER BY step
        """,
        db_conn,
    )

    trends.to_csv(OUTPUT_DIR / "fraud_trends.csv", index=False)

print("Dashboard analytics exported successfully.")
print(f"Files saved to: {OUTPUT_DIR}")
