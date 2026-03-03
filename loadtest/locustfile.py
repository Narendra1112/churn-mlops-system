import random
import pandas as pd
from locust import HttpUser, task, between

DATA_PATH = "data/Telecom_processed.csv"

_df = pd.read_csv(DATA_PATH)
if "Churn" in _df.columns:
    _df = _df.drop(columns=["Churn"])

# Convert once (fast). Avoid per-request pandas sampling overhead.
_ROWS = _df.to_dict(orient="records")
N = len(_ROWS)

def row_to_payload(row: dict) -> dict:
    # Your dataset DOES NOT have raw strings (Contract/InternetService/PaymentMethod)
    # It has one-hot columns. So we must reconstruct strings.
    # InternetService
    if int(row.get("InternetService_No", 0)) == 1:
        internet = "No"
    elif int(row.get("InternetService_Fiber_optic", 0)) == 1:
        internet = "Fiber optic"
    else:
        internet = "DSL"

    # Contract
    if int(row.get("Contract_Two_year", 0)) == 1:
        contract = "Two year"
    elif int(row.get("Contract_One_year", 0)) == 1:
        contract = "One year"
    else:
        contract = "Month-to-month"

    # PaymentMethod (your CSV column names have parentheses)
    if int(row.get("PaymentMethod_Credit_card_(automatic)", 0)) == 1:
        pay = "Credit card (automatic)"
    elif int(row.get("PaymentMethod_Electronic_check", 0)) == 1:
        pay = "Electronic check"
    else:
        pay = "Mailed check"

    return {
        "UserId": str(random.randint(1, 10_000_000)),
        "Gender": int(row["Gender"]),
        "SeniorCitizen": int(row["SeniorCitizen"]),
        "Partner": int(row["Partner"]),
        "Dependents": int(row["Dependents"]),
        "Tenure": float(row["Tenure"]),
        "PhoneService": int(row["PhoneService"]),
        "MultipleLines": int(row["MultipleLines"]),
        "OnlineSecurity": int(row["OnlineSecurity"]),
        "OnlineBackup": int(row["OnlineBackup"]),
        "DeviceProtection": int(row["DeviceProtection"]),
        "TechSupport": int(row["TechSupport"]),
        "StreamingTV": int(row["StreamingTV"]),
        "StreamingMovies": int(row["StreamingMovies"]),
        "PaperlessBilling": int(row["PaperlessBilling"]),
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TotalCharges": float(row["TotalCharges"]),
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": pay,
    }

class ChurnUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def predict(self):
        row = _ROWS[random.randrange(N)]
        payload = row_to_payload(row)
        self.client.post("/predict", json=payload)