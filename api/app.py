from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import time
import joblib
import os
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from loguru import logger
import dill  


# PATHS

MODEL_PATH = "models/xgb_churn_best.pkl"
PREDICTION_LOG_PATH = "data/prediction_log.csv"


# MODEL AUTO-RELOAD CACHE

_model_cache = {
    "model": None,
    "mtime": None,
}

def load_model():
    """Reload the model ONLY if the file changed on disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")

    current_mtime = os.path.getmtime(MODEL_PATH)

    # Use cache if unchanged
    if _model_cache["model"] and _model_cache["mtime"] == current_mtime:
        return _model_cache["model"]

    logger.info("Reloading latest model...")

    # Load with dill compatibility
    model = joblib.load(MODEL_PATH)

    _model_cache["model"] = model
    _model_cache["mtime"] = current_mtime
    logger.info("Model loaded")

    return model


# FINAL FEATURE LIST (MATCHES TRAINING)

FEATURES = [
    "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "InternetService_Fiber_optic", "InternetService_No",
    "Contract_One_year", "Contract_Two_year",
    "PaymentMethod_Credit_card_(automatic)",
    "PaymentMethod_Electronic_check",
    "PaymentMethod_Mailed_check"
]


# FASTAPI APP

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI serving retrained XGBoost model from Airflow",
    version="2.0.0"
)


# PROMETHEUS

instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)

PRED_COUNT = Counter("churn_predictions_total", "Predictions", ["label"])
PRED_LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency")


# INPUT SCHEMA

class CustomerInput(BaseModel):
    MonthlyCharges: float
    Tenure: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    PaperlessBilling: str
    MultipleLines: str
    OnlineBackup: str


# ENCODING (FINAL FIXED VERSION)

def encode_raw_input(user: dict) -> pd.DataFrame:
    df = pd.DataFrame([user])

    # Contract encoding
    df["Contract_One_year"] = (df["Contract"] == "One year").astype(int)
    df["Contract_Two_year"] = (df["Contract"] == "Two year").astype(int)
    df.drop(columns=["Contract"], inplace=True)

    # InternetService encoding
    df["InternetService_Fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["InternetService_No"] = (df["InternetService"] == "No").astype(int)
    df.drop(columns=["InternetService"], inplace=True)

    # Payment Method (MATCHES model exactly)
    df["PaymentMethod_Electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["PaymentMethod_Mailed_check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    df["PaymentMethod_Credit_card_(automatic)"] = (
        df["PaymentMethod"] == "Credit card (automatic)"
    ).astype(int)
    df.drop(columns=["PaymentMethod"], inplace=True)

    # Binary fields
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
    df["MultipleLines"] = df["MultipleLines"].map({"Yes": 1, "No": 0})
    df["OnlineBackup"] = df["OnlineBackup"].map({"Yes": 1, "No": 0})

    # Ensure all features exist
    for col in FEATURES:
        if col not in df:
            df[col] = 0

    return df[FEATURES].astype(float)


# ROUTES

@app.get("/health")
def health():
    return {"status": "OK", "message": "FastAPI running with Airflow model"}

@app.post("/predict")
def predict(data: CustomerInput):
    start = time.time()

    model = load_model()
    payload = data.dict()
    input_df = encode_raw_input(payload)

    # Predict
    prob = float(model.predict_proba(input_df)[0][1])
    label = int(prob >= 0.5)
    label_str = "churn" if label == 1 else "no_churn"

    # Prometheus metrics
    PRED_COUNT.labels(label=label_str).inc()
    PRED_LATENCY.observe(time.time() - start)

    # Drift logging
    record = payload.copy()
    record["prediction"] = label
    record["probability"] = prob
    record["timestamp"] = datetime.utcnow().isoformat()

    os.makedirs("data", exist_ok=True)
    header_needed = not os.path.exists(PREDICTION_LOG_PATH)

    pd.DataFrame([record]).to_csv(
        PREDICTION_LOG_PATH,
        mode="a",
        index=False,
        header=header_needed
    )

    return {
        "prediction": label,
        "probability_percent": round(prob * 100, 2)
    }
