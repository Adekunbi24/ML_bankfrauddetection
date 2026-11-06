
import csv
import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv(override=True)

CSV_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
BATCH_SIZE = 50_000

SOURCE_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud",
]

TARGET_COLUMNS = [
    "source_row_number", "step", "transaction_type", "amount",
    "sender_account", "sender_old_balance", "sender_new_balance",
    "recipient_account", "recipient_old_balance",
    "recipient_new_balance", "is_fraud", "is_flagged_fraud",
]

RENAME_COLUMNS = {
    "type": "transaction_type",
    "nameOrig": "sender_account",
    "oldbalanceOrg": "sender_old_balance",
    "newbalanceOrig": "sender_new_balance",
    "nameDest": "recipient_account",
    "oldbalanceDest": "recipient_old_balance",
    "newbalanceDest": "recipient_new_balance",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

NUMERIC_COLUMNS = [
    "amount", "sender_old_balance", "sender_new_balance",
    "recipient_old_balance", "recipient_new_balance",
]


def validate_batch(batch):
    if batch.isna().any().any():
        raise ValueError("Batch contains missing values")

    if (batch["amount"] < 0).any():
        raise ValueError("Batch contains negative transaction amounts")

    if (batch["step"] < 1).any():
        raise ValueError("Batch contains invalid simulation steps")

    for column in ("is_fraud", "is_flagged_fraud"):
        if not batch[column].isin([0, 1]).all():
            raise ValueError(f"Invalid values in {column}")

    if not batch["source_row_number"].is_unique:
        raise ValueError("Duplicate source row numbers within batch")


def load_transactions():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

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

    total_processed = 0

    with psycopg.connect(**connection_args) as conn:
        for batch_number, batch in enumerate(
            pd.read_csv(CSV_PATH, chunksize=BATCH_SIZE),
            start=1,
        ):
            if list(batch.columns) != SOURCE_COLUMNS:
                raise ValueError("Unexpected PaySim CSV columns")

            batch = batch.rename(columns=RENAME_COLUMNS)

            batch.insert(
                0,
                "source_row_number",
                range(
                    total_processed + 1,
                    total_processed + len(batch) + 1,
                ),
            )

            validate_batch(batch)

            # PostgreSQL NUMERIC columns have two decimal places.
            for column in NUMERIC_COLUMNS:
                batch[column] = batch[column].round(2)

            # Copy into a temporary staging table first.
            # This allows ON CONFLICT to skip previously imported rows.
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("""
    CREATE TEMP TABLE IF NOT EXISTS paysim_staging
    AS SELECT
        source_row_number,
        step,
        transaction_type,
        amount,
        sender_account,
        sender_old_balance,
        sender_new_balance,
        recipient_account,
        recipient_old_balance,
        recipient_new_balance,
        is_fraud,
        is_flagged_fraud
    FROM raw_transactions
    WITH NO DATA
""")

                    with cur.copy("""
                        COPY paysim_staging (
                            source_row_number, step, transaction_type,
                            amount, sender_account, sender_old_balance,
                            sender_new_balance, recipient_account,
                            recipient_old_balance, recipient_new_balance,
                            is_fraud, is_flagged_fraud
                        ) FROM STDIN
                    """) as copy:
                        for record in batch[TARGET_COLUMNS].itertuples(
                            index=False, name=None
                        ):
                            copy.write_row(record)

                    cur.execute("""
                        INSERT INTO raw_transactions (
                            source_row_number, step, transaction_type,
                            amount, sender_account, sender_old_balance,
                            sender_new_balance, recipient_account,
                            recipient_old_balance, recipient_new_balance,
                            is_fraud, is_flagged_fraud
                        )
                        SELECT
                            source_row_number, step, transaction_type,
                            amount, sender_account, sender_old_balance,
                            sender_new_balance, recipient_account,
                            recipient_old_balance, recipient_new_balance,
                            is_fraud, is_flagged_fraud
                        FROM paysim_staging
                        ON CONFLICT (source_row_number) DO NOTHING
                    """)

                    inserted = cur.rowcount

            total_processed += len(batch)

            print(
                f"Batch {batch_number}: "
                f"{len(batch):,} processed, "
                f"{inserted:,} inserted"
            )

    print(f"ETL completed. {total_processed:,} rows processed.")


if __name__ == "__main__":
    load_transactions()
