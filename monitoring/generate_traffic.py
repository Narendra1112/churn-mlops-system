# monitoring/generate_traffic.py
from __future__ import annotations

import os
import json
import time
import random
import argparse
import uuid
from collections import Counter

import requests
import pandas as pd


# ----------------------------
# Paths / Defaults
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_PATH = os.path.join(BASE_DIR, "models", "churn", "manifest.json")
BASELINE_DIR = os.path.join(BASE_DIR, "monitoring", "baseline")

# Docker default: talk to service name "api"
# Host override: export API_BASE=http://localhost:8000
DEFAULT_API_BASE = os.getenv("API_BASE", "http://api:8000").rstrip("/")


# ----------------------------
# Helpers
# ----------------------------
def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_version(manifest: dict) -> str:
    stable = manifest.get("stable_version")
    candidate = manifest.get("candidate_version")
    routing = manifest.get("routing") or {"stable": 100, "candidate": 0}

    stable_w = int(routing.get("stable", 100) or 0)
    cand_w = int(routing.get("candidate", 0) or 0)

    if not stable:
        raise ValueError("manifest.json missing stable_version")

    if not candidate or cand_w <= 0:
        return stable

    total = stable_w + cand_w
    if total <= 0:
        return stable

    pick = random.randint(1, total)
    return stable if pick <= stable_w else candidate


def baseline_path_for(version: str) -> str:
    return os.path.join(BASELINE_DIR, f"{version}_baseline.csv")


def load_baseline_df(version: str) -> pd.DataFrame:
    path = baseline_path_for(version)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline missing for {version}: {path}\n"
            f"Fix: create baseline during training for {version} (Airflow task) "
            f"or copy an existing baseline if you intentionally reuse it."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Baseline file is empty: {path}")
    return df


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def to_api_payload_from_row(row: dict, user_id: str) -> dict:
    # InternetService
    if "InternetService_No" in row and safe_int(row.get("InternetService_No")) == 1:
        internet = "No"
    elif "InternetService_Fiber_optic" in row and safe_int(row.get("InternetService_Fiber_optic")) == 1:
        internet = "Fiber optic"
    else:
        internet = "DSL"

    # Contract
    if "Contract_Two_year" in row and safe_int(row.get("Contract_Two_year")) == 1:
        contract = "Two year"
    elif "Contract_One_year" in row and safe_int(row.get("Contract_One_year")) == 1:
        contract = "One year"
    else:
        contract = "Month-to-month"

    # PaymentMethod
    if "PaymentMethod_Mailed_check" in row and safe_int(row.get("PaymentMethod_Mailed_check")) == 1:
        pm = "Mailed check"
    elif "PaymentMethod_Electronic_check" in row and safe_int(row.get("PaymentMethod_Electronic_check")) == 1:
        pm = "Electronic check"
    elif "PaymentMethod_Credit_card_(automatic)" in row and safe_int(row.get("PaymentMethod_Credit_card_(automatic)")) == 1:
        pm = "Credit card (automatic)"
    else:
        pm = "Bank transfer (automatic)"

    payload = {
        "UserId": user_id,
        "Gender": safe_int(row.get("Gender")),
        "SeniorCitizen": safe_int(row.get("SeniorCitizen")),
        "Partner": safe_int(row.get("Partner")),
        "Dependents": safe_int(row.get("Dependents")),
        "Tenure": max(1, min(72, safe_int(row.get("Tenure"), 1))),
        "PhoneService": safe_int(row.get("PhoneService")),
        "MultipleLines": safe_int(row.get("MultipleLines")),
        "OnlineSecurity": safe_int(row.get("OnlineSecurity")),
        "OnlineBackup": safe_int(row.get("OnlineBackup")),
        "DeviceProtection": safe_int(row.get("DeviceProtection")),
        "TechSupport": safe_int(row.get("TechSupport")),
        "StreamingTV": safe_int(row.get("StreamingTV")),
        "StreamingMovies": safe_int(row.get("StreamingMovies")),
        "PaperlessBilling": safe_int(row.get("PaperlessBilling")),
        "MonthlyCharges": round(safe_float(row.get("MonthlyCharges")), 2),
        "TotalCharges": round(safe_float(row.get("TotalCharges")), 2),
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": pm,
    }
    return payload


def apply_drift(payload: dict) -> dict:
    payload["SeniorCitizen"] = 1 if random.random() < 0.85 else 0
    payload["PhoneService"] = 1 if random.random() < 0.35 else 0
    payload["MonthlyCharges"] = round(min(120.0, max(20.0, payload["MonthlyCharges"] + random.uniform(20, 50))), 2)
    payload["TotalCharges"] = round(min(6000.0, max(100.0, payload["TotalCharges"] + random.uniform(500, 2000))), 2)
    payload["PaymentMethod"] = random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)"])
    return payload


def post_with_retries(url: str, payload: dict, timeout_sec: float, retries: int) -> requests.Response:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return requests.post(url, json=payload, timeout=timeout_sec)
        except Exception as e:
            last_exc = e
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {repr(last_exc)}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # base of the API, not full url
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="e.g. http://api:8000 (docker) or http://localhost:8000 (host)")
    parser.add_argument("--total", type=int, default=1200)
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--sleep-ms", type=int, default=0, help="sleep between requests (ms)")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=1)


    parser.add_argument("--manifest-refresh-sec", type=float, default=2.0, help="reload manifest periodically")

    # routing overrides
    parser.add_argument("--force-version", default=None, help="send all requests as a fixed version (debug)")
    parser.add_argument("--stable-only", action="store_true", help="always route to stable_version")
    parser.add_argument("--candidate-only", action="store_true", help="always route to candidate_version")

    args = parser.parse_args()

    if args.stable_only and args.candidate_only:
        raise ValueError("Choose only one: --stable-only or --candidate-only")

    # ✅ FIX: build predict URL from api-base
    url = args.api_base.rstrip("/") + "/predict"

    counts = Counter()
    errors = 0

    baseline_cache = {}

    manifest = load_manifest()
    last_manifest_reload = time.time()

    for i in range(args.total):
        now = time.time()
        if args.manifest_refresh_sec > 0 and (now - last_manifest_reload) >= args.manifest_refresh_sec:
            manifest = load_manifest()
            last_manifest_reload = now

        stable = manifest.get("stable_version")
        candidate = manifest.get("candidate_version")

        if args.force_version:
            version_target = args.force_version
        elif args.stable_only:
            if not stable:
                raise ValueError("manifest.json missing stable_version")
            version_target = stable
        elif args.candidate_only:
            if not candidate:
                raise ValueError("manifest.json missing candidate_version")
            version_target = candidate
        else:
            version_target = choose_version(manifest)

        if version_target not in baseline_cache:
            baseline_cache[version_target] = load_baseline_df(version_target)

        df = baseline_cache[version_target]
        row = df.sample(n=1).iloc[0].to_dict()

        payload = to_api_payload_from_row(row, user_id=f"user_{i}")

        if args.mode == "drift":
            payload = apply_drift(payload)

        try:
            r = post_with_retries(url, payload, timeout_sec=args.timeout, retries=args.retries)
            if r.status_code != 200:
                errors += 1
                if errors <= 5:
                    print("Error:", r.status_code, r.text)
            else:
                resp = r.json()
                mv = resp.get("model_version", "unknown")
                counts[mv] += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print("Error:", repr(e))

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print("Traffic generation complete.")
    print("URL:", url)
    print("Mode:", args.mode)
    print("Errors:", errors)
    print("Version counts:", dict(counts))


if __name__ == "__main__":
    main()
