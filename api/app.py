from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from api.concurrency import BackpressureError, inference_slot
from api.inference_runtime import run_predict_threadpool
from api.timeouts import InferenceTimeoutError, with_inference_timeout

# CONSTANTS / PATHS

PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "data/prediction_log.jsonl")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "models/churn/manifest.json")


MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "32"))

# FASTAPI
app = FastAPI(title="Customer Churn API", version="6.1.0")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# PROMETHEUS METRICS (MODEL-AWARE)
# Routing + prediction volume
PRED_COUNT = Counter(
    "churn_predictions_total",
    "Predictions emitted by model",
    ["label", "model_version"],
)

ROUTE_COUNT = Counter(
    "churn_routing_total",
    "Routing decisions",
    ["model_version", "route"],  
)

# Latency: separate API total vs inference compute
API_LATENCY = Histogram(
    "churn_api_latency_seconds",
    "End-to-end /predict request latency (includes validation, encoding, model load, inference, logging)",
    ["model_version"],
)

INFER_LATENCY = Histogram(
    "churn_inference_latency_seconds",
    "Model inference compute latency only (predict_proba execution)",
    ["model_version"],
)

# Saturation
INFLIGHT = Gauge(
    "churn_inflight_requests",
    "Current in-flight inference requests (bounded by MAX_INFLIGHT)",
)

# Explicit error counters for dashboards 
OVERLOAD_429 = Counter(
    "churn_overload_rejections_total",
    "Requests rejected due to backpressure / overload (HTTP 429)",
    ["model_version"],
)

TIMEOUT_504 = Counter(
    "churn_inference_timeouts_total",
    "Requests failed due to inference timeout (HTTP 504)",
    ["model_version"],
)

ERROR_500 = Counter(
    "churn_internal_errors_total",
    "Internal server errors (HTTP 500) during prediction path",
    ["model_version", "stage"],  # bounded: artifact_load|inference|unknown
)

VALIDATION_400 = Counter(
    "churn_validation_errors_total",
    "Input validation errors that return HTTP 400",
    ["model_version", "field"],  # bounded: Contract|InternetService|PaymentMethod
)

# Keep these even if you don't set yet—Grafana panels can exist.
PSI_OVERALL = Gauge(
    "churn_psi_overall",
    "Overall PSI drift score for a model version vs baseline",
    ["model_version"],
)
PSI_FEATURE = Gauge(
    "churn_psi_feature",
    "Per-feature PSI drift score for a model version vs baseline",
    ["model_version", "feature"], 
)
PSI_BREACH = Gauge(
    "churn_psi_breach",
    "Drift breach flag (1=breach, 0=ok)",
    ["model_version"],
)

# MODEL / SIGNATURE / MANIFEST CACHES

_manifest_cache: Dict[str, Any] = {"obj": None, "mtime": None}
_signature_cache: Dict[str, Dict[str, Any]] = {}  # version -> {"features": [...], "mtime": ...}
_model_cache: Dict[str, Dict[str, Any]] = {}      # version -> {"model": model, "mtime": ...}


def model_path(version: str) -> str:
    return f"models/churn/{version}/model.json"


def signature_path(version: str) -> str:
    return f"models/churn/{version}/signature.json"


def load_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest missing: {MANIFEST_PATH}")

    mtime = os.path.getmtime(MANIFEST_PATH)
    if _manifest_cache["obj"] is not None and _manifest_cache["mtime"] == mtime:
        return _manifest_cache["obj"]

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)

    _manifest_cache["obj"] = obj
    _manifest_cache["mtime"] = mtime
    return obj


def load_signature_for_version(version: str) -> list[str]:
    path = signature_path(version)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Signature missing: {path}")

    mtime = os.path.getmtime(path)
    cached = _signature_cache.get(version)
    if cached and cached["mtime"] == mtime:
        return cached["features"]

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    features = obj["features"]
    _signature_cache[version] = {"features": features, "mtime": mtime}
    return features


def load_model_for_version(version: str) -> xgb.XGBClassifier:
    path = model_path(version)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file missing: {path}")

    mtime = os.path.getmtime(path)
    cached = _model_cache.get(version)
    if cached and cached["mtime"] == mtime:
        return cached["model"]

    logger.info(f"Loading model version: {version}")
    model = xgb.XGBClassifier()
    model.load_model(path)

    _model_cache[version] = {"model": model, "mtime": mtime}
    return model

# ROUTING (DETERMINISTIC)

def _stable_bucket(user_key: str) -> int:
    digest = hashlib.sha256(user_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100  # 0..99


def pick_version_and_route(user_key: Optional[str]) -> Tuple[str, str]:
    m = load_manifest()
    stable = m["stable_version"]
    cand = m.get("candidate_version")
    routing = m.get("routing", {"stable": 100, "candidate": 0})
    cand_pct = int(routing.get("candidate", 0))

    if not cand or cand_pct <= 0:
        return stable, "stable"

    if user_key:
        bucket = _stable_bucket(user_key) 
    else:
        bucket = random.randint(1, 100)

    if bucket <= cand_pct:
        return cand, "candidate"
    return stable, "stable"


# VALID CATEGORY GUARDS

VALID_CONTRACTS = ["Month-to-month", "One year", "Two year"]
VALID_INTERNET = ["DSL", "Fiber optic", "No"]
VALID_PAYMENT = [
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
]


# INPUT SCHEMA

class CustomerInput(BaseModel):
    UserId: Optional[str] = Field(default=None, description="Optional routing key for deterministic canary")
    Gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    Tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str


# ENCODING (MATCH TRAINING CONTRACT)

def encode_raw_input(user: Dict[str, Any], features: list[str], model_version: str) -> pd.DataFrame:
    if user["Contract"] not in VALID_CONTRACTS:
        VALIDATION_400.labels(model_version=model_version, field="Contract").inc()
        raise HTTPException(status_code=400, detail="Invalid Contract")

    if user["InternetService"] not in VALID_INTERNET:
        VALIDATION_400.labels(model_version=model_version, field="InternetService").inc()
        raise HTTPException(status_code=400, detail="Invalid InternetService")

    if user["PaymentMethod"] not in VALID_PAYMENT:
        VALIDATION_400.labels(model_version=model_version, field="PaymentMethod").inc()
        raise HTTPException(status_code=400, detail="Invalid PaymentMethod")

    df = pd.DataFrame([user])

    if "UserId" in df.columns:
        df.drop(columns=["UserId"], inplace=True)

    # One-hot: Contract
    df["Contract_One_year"] = (df["Contract"] == "One year").astype(int)
    df["Contract_Two_year"] = (df["Contract"] == "Two year").astype(int)
    df.drop(columns=["Contract"], inplace=True)

    # One-hot: InternetService
    df["InternetService_Fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["InternetService_No"] = (df["InternetService"] == "No").astype(int)
    df.drop(columns=["InternetService"], inplace=True)

    # One-hot: PaymentMethod
    df["PaymentMethod_Credit_card_automatic"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    df["PaymentMethod_Electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["PaymentMethod_Mailed_check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    df.drop(columns=["PaymentMethod"], inplace=True)

    for col in features:
        if col not in df:
            df[col] = 0

    return df[features].astype(float)


# ROUTES

@app.get("/health")
def health():
    m = load_manifest()
    return {
        "status": "OK",
        "stable_version": m.get("stable_version"),
        "candidate_version": m.get("candidate_version"),
        "routing": m.get("routing", {"stable": 100, "candidate": 0}),
    }


@app.get("/manifest")
def manifest():
    try:
        m = load_manifest()
        return JSONResponse(content=m)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load manifest: {e}")


@app.post("/predict")
async def predict(data: CustomerInput):
    t0 = time.perf_counter()
    payload = data.dict()

    version: Optional[str] = None
    try:
        version, route = pick_version_and_route(payload.get("UserId"))
        ROUTE_COUNT.labels(model_version=version, route=route).inc()

        async def _do_inference():
            # Acquire bounded concurrency slot (may raise BackpressureError)
            async with inference_slot():
                INFLIGHT.inc()
                try:
                    model = load_model_for_version(version)
                    features = load_signature_for_version(version)
                    input_df = encode_raw_input(payload, features, model_version=version)

                    # Pure inference compute timing
                    t_inf0 = time.perf_counter()
                    proba = await run_predict_threadpool(lambda: model.predict_proba(input_df))
                    INFER_LATENCY.labels(model_version=version).observe(time.perf_counter() - t_inf0)

                    prob = float(proba[0][1])
                    label = int(prob >= 0.5)
                    label_str = "churn" if label == 1 else "no_churn"
                    PRED_COUNT.labels(label=label_str, model_version=version).inc()

                    # Write log (JSONL)
                    record = payload.copy()
                    encoded = input_df.iloc[0].to_dict()
                    for feat_name, feat_val in encoded.items():
                        record[feat_name] = float(feat_val)

                    record["prediction"] = label
                    record["probability"] = prob
                    record["model_version"] = version
                    record["timestamp"] = datetime.now(timezone.utc).isoformat()

                    os.makedirs("data", exist_ok=True)
                    with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    return {
                        "prediction": label,
                        "probability_percent": round(prob * 100, 2),
                        "model_version": version,
                    }
                finally:
                    INFLIGHT.dec()

        try:
            return await with_inference_timeout(_do_inference())

        except BackpressureError:
            OVERLOAD_429.labels(model_version=version or "unknown").inc()
            raise HTTPException(status_code=429, detail="Too many in-flight inferences")

        except InferenceTimeoutError:
            TIMEOUT_504.labels(model_version=version or "unknown").inc()
            raise HTTPException(status_code=504, detail="Inference timed out")

    except HTTPException:
        # validation errors are already counted via VALIDATION_400
        raise

    except FileNotFoundError:
        ERROR_500.labels(model_version=version or "unknown", stage="artifact_load").inc()
        raise HTTPException(status_code=500, detail="Model artifacts missing")

    except Exception as e:
        ERROR_500.labels(model_version=version or "unknown", stage="inference").inc()
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        if version:
            API_LATENCY.labels(model_version=version).observe(time.perf_counter() - t0)