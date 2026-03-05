# ops/auto_promote.py
from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone
from collections import Counter
from typing import Optional, Tuple, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_PATH = os.path.join(BASE_DIR, "models", "churn", "manifest.json")
MODELS_DIR = os.path.join(BASE_DIR, "models", "churn")

PRED_LOG = os.path.join(BASE_DIR, "data", "prediction_log.jsonl")
PSI_DIR = os.path.join(BASE_DIR, "monitoring", "psi_reports")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "logs", "promotion_audit.jsonl")

# Traffic gating
MIN_CANDIDATE_TRAFFIC = 200
TRAFFIC_WINDOW_MINUTES = 60
MAX_LOG_LINES_TO_SCAN = 50_000

# Drift gating
DRIFTED_FEATURES_THRESHOLD = 5
PSI_ALERT_THRESHOLD = 0.25

# Manifest write reliability
MANIFEST_WRITE_RETRIES = 5
MANIFEST_WRITE_BACKOFF_SEC = 0.05

# Single-writer lock
LOCK_PATH = os.path.join(BASE_DIR, "models", "churn", ".manifest.lock")
LOCK_TTL_SEC = 120

# PSI freshness control
MAX_PSI_AGE_MULTIPLIER = 2.0
REQUIRE_PSI_WINDOW_MATCH = True



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_PATH)


def save_manifest(m: Dict[str, Any]) -> None:
    for attempt in range(MANIFEST_WRITE_RETRIES):
        try:
            atomic_write_json(MANIFEST_PATH, m)
            return
        except PermissionError:
            time.sleep(MANIFEST_WRITE_BACKOFF_SEC * (attempt + 1))
    atomic_write_json(MANIFEST_PATH, m)


# LOCK

def _read_lock_meta() -> Optional[dict]:
    if not os.path.exists(LOCK_PATH):
        return None
    try:
        with open(LOCK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def acquire_manifest_lock() -> None:
    meta = _read_lock_meta()
    if meta and isinstance(meta.get("created_epoch"), (int, float)):
        age = time.time() - float(meta["created_epoch"])
        if age > LOCK_TTL_SEC:
            try:
                os.remove(LOCK_PATH)
            except Exception:
                pass

    payload = {"pid": os.getpid(), "created_epoch": time.time(), "created_utc": utc_now_iso()}
    try:
        with open(LOCK_PATH, "x", encoding="utf-8") as f:
            f.write(json.dumps(payload))
    except FileExistsError:
        raise RuntimeError(f"manifest lock held: {LOCK_PATH} meta={meta}")


def release_manifest_lock() -> None:
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception:
        pass

# VALIDATION

def model_artifacts_exist(version: str) -> Tuple[bool, str]:
    vdir = os.path.join(MODELS_DIR, version)
    model_path = os.path.join(vdir, "model.json")
    sig_path = os.path.join(vdir, "signature.json")

    missing = []
    if not os.path.isdir(vdir):
        missing.append(f"missing dir: {vdir}")
    if not os.path.exists(model_path):
        missing.append(f"missing: {model_path}")
    if not os.path.exists(sig_path):
        missing.append(f"missing: {sig_path}")

    if missing:
        return False, "; ".join(missing)
    return True, "OK"


def load_psi(version: str) -> Dict[str, Any]:
    path = os.path.join(PSI_DIR, f"psi_{version}.json")
    if not os.path.exists(path):
        return {}
    return load_json(path)


def parse_ts_to_epoch_seconds(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def validate_psi_report(report: Dict[str, Any], *, expected_window_min: int) -> Tuple[bool, str]:
    if not report:
        return False, "PSI report missing"

    gen = report.get("generated_at_utc")
    window = report.get("window_minutes")
    baseline_id = report.get("baseline_id")

    if not gen or not isinstance(gen, str):
        return False, "PSI missing generated_at_utc"
    if window is None:
        return False, "PSI missing window_minutes"
    try:
        window = int(window)
    except Exception:
        return False, f"PSI invalid window_minutes: {window}"
    if not baseline_id or not isinstance(baseline_id, str):
        return False, "PSI missing baseline_id"

    if REQUIRE_PSI_WINDOW_MATCH and window != int(expected_window_min):
        return False, f"PSI window mismatch: report={window} expected={expected_window_min}"

    gen_epoch = parse_ts_to_epoch_seconds(gen)
    if gen_epoch is None:
        return False, f"PSI invalid generated_at_utc: {gen}"

    max_age = float(MAX_PSI_AGE_MULTIPLIER) * window * 60.0
    age = time.time() - gen_epoch
    if age > max_age:
        return False, f"PSI report stale: age_sec={age:.1f} max_age_sec={max_age:.1f}"

    return True, "OK"


def psi_status(report: Dict[str, Any]) -> str:
    if not report:
        return "MISSING"
    overall = float(report.get("overall_psi_mean", 0.0))
    drifted = report.get("drifted_features", []) or []
    if overall >= PSI_ALERT_THRESHOLD or len(drifted) > DRIFTED_FEATURES_THRESHOLD:
        return "ALERT"
    return "OK"


# TRAFFIC

def count_traffic_bounded(window_minutes: int = TRAFFIC_WINDOW_MINUTES) -> Counter:
    c = Counter()
    if not os.path.exists(PRED_LOG):
        return c

    cutoff = time.time() - (window_minutes * 60)

    with open(PRED_LOG, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        chunk_size = min(size, 8 * 1024 * 1024)
        if size > chunk_size:
            f.seek(size - chunk_size)
        else:
            f.seek(0)
        data = f.read().decode("utf-8", errors="ignore")

    lines = data.splitlines()
    if len(lines) > MAX_LOG_LINES_TO_SCAN:
        lines = lines[-MAX_LOG_LINES_TO_SCAN:]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        v = obj.get("model_version")
        if not v:
            continue

        ts = parse_ts_to_epoch_seconds(obj.get("timestamp"))
        if ts is not None and ts < cutoff:
            continue

        c[v] += 1

    return c



# AUDIT

def append_audit_log(entry: dict) -> None:
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# MAIN

def main():
    acquire_manifest_lock()
    try:
        m = load_manifest()
        stable = m.get("stable_version")
        cand = m.get("candidate_version")
        routing = m.get("routing", {"stable": 100, "candidate": 0})

        if not stable:
            print("NOT PROMOTED: manifest missing stable_version")
            return

        if cand is None:
            print("NOT PROMOTED: candidate_version is null")
            return

        cand_pct = int(routing.get("candidate", 0) or 0)
        if cand_pct <= 0:
            print("NOT PROMOTED: routing.candidate <= 0 (candidate receives no traffic)")
            return

        print("---- AUTO PROMOTE CHECK ----")
        print("timestamp_utc    :", utc_now_iso())
        print("stable_version   :", stable)
        print("candidate_version:", cand)
        print("routing          :", routing)

        ok, msg = model_artifacts_exist(cand)
        if not ok:
            print("NOT PROMOTED: candidate artifacts invalid ->", msg)
            return

        traffic = count_traffic_bounded()
        stable_n = traffic.get(stable, 0)
        cand_n = traffic.get(cand, 0)

        print("traffic stable   :", stable_n)
        print("traffic candidate:", cand_n)

        cand_psi = load_psi(cand)
        psi_ok, psi_reason = validate_psi_report(cand_psi, expected_window_min=TRAFFIC_WINDOW_MINUTES)

        status = psi_status(cand_psi) if psi_ok else "MISSING"
        overall = float(cand_psi.get("overall_psi_mean", 0.0)) if (cand_psi and psi_ok) else None
        drifted = cand_psi.get("drifted_features", []) if (cand_psi and psi_ok) else []

        print("psi status       :", status)
        print("psi validation   :", "OK" if psi_ok else psi_reason)
        print("overall_psi_mean  :", overall)
        print("drifted_features  :", drifted)
        print("----------------------------")

        reasons = []

        if cand_n < MIN_CANDIDATE_TRAFFIC:
            reasons.append(f"candidate traffic too low: {cand_n} < {MIN_CANDIDATE_TRAFFIC}")

        if not psi_ok:
            reasons.append(f"PSI invalid: {psi_reason}")
        else:
            if len(drifted) > DRIFTED_FEATURES_THRESHOLD:
                reasons.append(f"drifted_features count too high: {len(drifted)} > {DRIFTED_FEATURES_THRESHOLD}")
            if overall is None:
                reasons.append("PSI report missing for candidate")
            elif overall >= PSI_ALERT_THRESHOLD:
                reasons.append(f"overall_psi_mean too high: {overall} >= {PSI_ALERT_THRESHOLD}")

        if reasons:
            print("NOT PROMOTED:")
            for r in reasons:
                print("-", r)

            append_audit_log({
                "timestamp_utc": utc_now_iso(),
                "old_stable": stable,
                "candidate_version": cand,
                "candidate_traffic": cand_n,
                "overall_psi_mean": overall,
                "drifted_feature_count": len(drifted),
                "result": "fail",
                "error": "; ".join(reasons),
            })
            return

        # PROMOTE
        old_stable = stable
        old_candidate = cand

        m["previous_stable_version"] = old_stable
        m["stable_version"] = old_candidate
        m["candidate_version"] = None
        m["routing"] = {"stable": 100, "candidate": 0}
        m["last_promotion_utc"] = utc_now_iso()

        if "candidate_metrics" in m:
            m["stable_metrics"] = m.get("candidate_metrics")
            m.pop("candidate_metrics", None)

        audit_entry = {
            "timestamp_utc": m["last_promotion_utc"],
            "old_stable": old_stable,
            "new_stable": old_candidate,
            "candidate_version": old_candidate,
            "candidate_traffic": cand_n,
            "overall_psi_mean": overall,
            "drifted_feature_count": len(drifted),
            "result": "success",
            "error": None,
        }

        try:
            save_manifest(m)
            append_audit_log(audit_entry)
        except Exception as e:
            audit_entry["result"] = "fail"
            audit_entry["error"] = repr(e)
            append_audit_log(audit_entry)
            raise

        print("PROMOTED:")
        print(f"- stable_version updated: {old_stable} -> {m['stable_version']}")
        print(f"- previous_stable_version set: {m['previous_stable_version']}")
        print("- candidate cleared, routing set to 100/0")

    finally:
        release_manifest_lock()


if __name__ == "__main__":
    main()
