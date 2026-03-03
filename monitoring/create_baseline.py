import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telecom_processed.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models", "churn")
BASELINE_DIR = os.path.join(BASE_DIR, "monitoring", "baseline")

os.makedirs(BASELINE_DIR, exist_ok=True)


def load_signature(version):
    path = os.path.join(MODEL_DIR, version, "signature.json")
    with open(path, "r") as f:
        return json.load(f)["features"]


def create_baseline(version):
    df = pd.read_csv(DATA_PATH)
    features = load_signature(version)

    # Fix legacy column naming
    if "PaymentMethod_Credit_card_(automatic)" in df.columns:
        df = df.rename(columns={
            "PaymentMethod_Credit_card_(automatic)": "PaymentMethod_Credit_card_automatic"
        })

    # Remove target if present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    df = df[features]

    baseline_path = os.path.join(BASELINE_DIR, f"{version}_baseline.csv")
    df.to_csv(baseline_path, index=False)

    print(f"Baseline saved: {baseline_path}")


if __name__ == "__main__":
    create_baseline("v1")
    create_baseline("v2")
