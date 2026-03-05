import os
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BASE_DIR / "models" / "churn" / "manifest.json"
PRED_LOG = BASE_DIR / "data" / "prediction_log.jsonl"


def die(msg: str, code: int = 1):
    print(f"[FAIL] {msg}")
    sys.exit(code)


def ok(msg: str):
    print(f"[OK] {msg}")


def must_exist(path: Path, label: str):
    if not path.exists():
        die(f"{label} missing: {path}")
    ok(f"{label} exists: {path}")


def load_manifest() -> dict:
    must_exist(MANIFEST_PATH, "manifest.json")
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        die(f"manifest.json unreadable/invalid JSON: {e}")


def validate_version_dir(version: str):
    vdir = BASE_DIR / "models" / "churn" / version
    must_exist(vdir, f"version dir {version}")
    must_exist(vdir / "model.json", f"{version}/model.json")
    must_exist(vdir / "signature.json", f"{version}/signature.json")

    try:
        with open(vdir / "signature.json", "r", encoding="utf-8") as f:
            sig = json.load(f)
        feats = sig.get("features")
        if not isinstance(feats, list) or not feats:
            die(f"{version}/signature.json has no valid 'features' list")
        ok(f"{version} signature features count: {len(feats)}")
    except Exception as e:
        die(f"{version}/signature.json invalid: {e}")


def main():
    print("=== PRE-FLIGHT CHECK ===")

    m = load_manifest()

    stable = m.get("stable_version")
    cand = m.get("candidate_version")
    routing = m.get("routing", {"stable": 100, "candidate": 0})

    if not stable:
        die("manifest missing stable_version")
    ok(f"stable_version: {stable}")

    validate_version_dir(stable)

    if cand:
        ok(f"candidate_version: {cand}")
        validate_version_dir(cand)
    else:
        ok("candidate_version: null")

    try:
        s = int(routing.get("stable", 0))
        c = int(routing.get("candidate", 0))
    except Exception:
        die(f"routing invalid: {routing}")

    if s < 0 or c < 0 or (s + c) != 100:
        die(f"routing must be non-negative and sum to 100. Got: {routing}")
    ok(f"routing: stable={s}, candidate={c}")

    if cand is None and c != 0:
        die("candidate_version is null but routing.candidate != 0 (inconsistent)")
    if cand is not None and c == 0:
        print("[WARN] candidate_version is set but routing.candidate == 0 (candidate will receive no traffic)")

    if PRED_LOG.exists():
        ok(f"prediction_log.jsonl exists: {PRED_LOG}")
    else:
        print(f"[WARN] prediction_log.jsonl not found yet (fine if you haven't generated traffic): {PRED_LOG}")

    print("=== PRE-FLIGHT PASSED ===")


if __name__ == "__main__":
    main()
