
import joblib
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.ml.train_fraud_model import (
    FEATURES,
    get_engine,
    load_data,
)

MODEL_PATH = Path("models/random_forest_model.joblib")
THRESHOLD = 0.80


def evaluate():
    print("Loading saved Random Forest model...", flush=True)
    model = joblib.load(MODEL_PATH)

    print("Loading transaction data...", flush=True)
    engine = get_engine()
    try:
        df = load_data(engine)
    finally:
        engine.dispose()

    # Reproduce the original chronological test split.
    test_cutoff = df["step"].quantile(0.80)
    test_df = df[df["step"] > test_cutoff].copy()

    X_test = test_df[FEATURES]
    y_test = test_df["is_fraud"].astype(int)

    print(f"Testing: {len(test_df):,} transactions", flush=True)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\nResults at threshold {THRESHOLD}:")
    print(f"Fraud correctly detected: {tp:,}")
    print(f"Fraud missed: {fn:,}")
    print(f"False alarms: {fp:,}")
    print(f"Legitimate transactions correctly classified: {tn:,}")

    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    average_precision = float(average_precision_score(y_test, probabilities))

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("Average precision:", round(average_precision, 4))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    evaluate()
