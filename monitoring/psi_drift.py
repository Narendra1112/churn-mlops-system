# monitoring/psi_drift.py
from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRED_LOG = os.path.join(BASE_DIR, "data", "prediction_log.jsonl")
BASELINE_DIR = os.path.join(BASE_DIR, "monitoring", "baseline")
OUT_DIR = os.path.join(BASE_DIR, "monitoring", "psi_reports")
MODEL_DIR = os.path.join(BASE_DIR, "models", "churn")

os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# UTIL
# -------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_signature(version: str) -> List[str]:
    path = os.path.join(MODEL_DIR, version, "signature.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    feats = obj.get("features")
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"Invalid signature.json for {version}: missing features[]")
    return feats


def load_baseline(version: str) -> pd.DataFrame:
    path = os.path.join(BASELINE_DIR, f"{version}_baseline.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline missing: {path}")
    return pd.read_csv(path)


def load_live(version: str, window_rows: int) -> pd.DataFrame:
    if not os.path.exists(PRED_LOG):
        raise FileNotFoundError(f"Prediction log missing: {PRED_LOG}")

    rows: List[dict] = []
    with open(PRED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("model_version") == version:
                rows.append(obj)

    if not rows:
        raise ValueError(f"No live rows for version={version} in prediction_log.jsonl")

    df = pd.DataFrame(rows)

    # keep most recent window if timestamp exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df = df.tail(int(window_rows))

    drop_cols = [c for c in ["prediction", "probability", "probability_percent", "timestamp", "model_version"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


# -------------------------
# PSI
# -------------------------

def _psi_from_counts(exp_perc: np.ndarray, act_perc: np.ndarray) -> float:
    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size < 50 or actual.size < 50:
        return 0.0

    # Quantile bins from expected
    qs = np.linspace(0, 1, bins + 1)
    try:
        breakpoints = np.unique(np.quantile(expected, qs))
    except Exception:
        return 0.0

    # If constant or too few unique values, PSI is 0 for this feature
    if breakpoints.size < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    return _psi_from_counts(exp_perc, act_perc)


def psi_binary(expected: np.ndarray, actual: np.ndarray) -> float:
    # for 0/1 features
    expected = expected.astype(float)
    actual = actual.astype(float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size < 50 or actual.size < 50:
        return 0.0

    exp_ones = np.mean(expected > 0.5)
    act_ones = np.mean(actual > 0.5)

    exp_perc = np.array([1 - exp_ones, exp_ones])
    act_perc = np.array([1 - act_ones, act_ones])

    return _psi_from_counts(exp_perc, act_perc)


def classify_drift(overall_psi_mean: float, n_drifted: int) -> str:
    if overall_psi_mean >= 0.25 or n_drifted >= 3:
        return "ALERT"
    if overall_psi_mean >= 0.10 or n_drifted >= 1:
        return "MONITOR"
    return "STABLE"


def compute_psi_report(version: str, *, window_minutes: int, sample_window_rows: int = 1000) -> Dict[str, Any]:
    features = load_signature(version)

    baseline_id = f"{version}_baseline.csv"
    base = load_baseline(version)
    live = load_live(version, window_rows=sample_window_rows)

    # enforce schema
    missing_live = [c for c in features if c not in live.columns]
    if missing_live:
        raise ValueError(f"Live data missing features for version={version}: {missing_live}")

    missing_base = [c for c in features if c not in base.columns]
    if missing_base:
        raise ValueError(f"Baseline missing features for version={version}: {missing_base}")

    base = base[features]
    live = live[features]

    psi_per_feature: Dict[str, float] = {}
    drifted_features: List[str] = []

    for col in features:
        b = pd.to_numeric(base[col], errors="coerce").to_numpy()
        a = pd.to_numeric(live[col], errors="coerce").to_numpy()

        # binary detection: all values are in {0,1} ignoring NaNs
        b_unique = np.unique(b[~np.isnan(b)])
        a_unique = np.unique(a[~np.isnan(a)])
        is_binary = (
            (b_unique.size > 0 and np.all(np.isin(b_unique, [0.0, 1.0]))) and
            (a_unique.size > 0 and np.all(np.isin(a_unique, [0.0, 1.0])))
        )

        if is_binary:
            psi = psi_binary(b, a)
        else:
            psi = psi_numeric(b, a, bins=10)

        psi = float(round(psi, 6))
        psi_per_feature[col] = psi
        if psi >= 0.25:
            drifted_features.append(col)

    overall = float(np.mean(list(psi_per_feature.values()))) if psi_per_feature else 0.0
    overall_psi_mean = float(round(overall, 6))
    drift_status = classify_drift(overall_psi_mean, len(drifted_features))

    return {
        "generated_at_utc": utc_now_iso(),
        "window_minutes": int(window_minutes),
        "baseline_id": baseline_id,
        "version": version,
        "sample_window_rows": int(sample_window_rows),
        "overall_psi_mean": overall_psi_mean,
        "threshold_drift": 0.25,
        "drift_status": drift_status,
        "drifted_features": drifted_features,
        "psi": psi_per_feature,
    }


def save_report(report: Dict[str, Any]) -> None:
    out = os.path.join(OUT_DIR, f"psi_{report['version']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved PSI report: {out}")


if __name__ == "__main__":
    WINDOW_MINUTES = int(os.getenv("PSI_WINDOW_MINUTES", "60"))
    SAMPLE_ROWS = int(os.getenv("PSI_SAMPLE_ROWS", "1000"))

    manifest_path = os.path.join(BASE_DIR, "models", "churn", "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    versions = []
    stable = m.get("stable_version")
    cand = m.get("candidate_version")

    if stable:
        versions.append(stable)
    if cand:
        versions.append(cand)

    versions = list(dict.fromkeys(versions))  # unique preserve order

    for v in versions:
        try:
            r = compute_psi_report(v, window_minutes=WINDOW_MINUTES, sample_window_rows=SAMPLE_ROWS)
            save_report(r)
        except Exception as e:
            print(f"[{v}] PSI failed: {e}")
