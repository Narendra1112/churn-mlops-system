import os
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
    


BASE_DIR = "/opt/airflow"
DATA_PATH = f"{BASE_DIR}/data/Telecom_processed.csv"
MANIFEST_PATH = f"{BASE_DIR}/models/churn/manifest.json"
CHURN_MODELS_DIR = f"{BASE_DIR}/models/churn"
BASELINE_DIR = f"{BASE_DIR}/monitoring/baseline"

DEFAULT_ROUTING = {"stable": 80, "candidate": 20}



def _load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _atomic_write_json(path: str, obj: dict):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _next_version() -> str:
    """
    Finds next version as v{max+1} based on existing folders under models/churn/v*
    """
    os.makedirs(CHURN_MODELS_DIR, exist_ok=True)
    existing = []
    for name in os.listdir(CHURN_MODELS_DIR):
        if name.startswith("v") and name[1:].isdigit():
            p = os.path.join(CHURN_MODELS_DIR, name)
            if os.path.isdir(p):
                existing.append(int(name[1:]))
    n = max(existing) + 1 if existing else 1
    return f"v{n}"




def extract_data(**context):
    import pandas as pd

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found.")
    if df.empty:
        raise ValueError("Training data is empty.")

    context["ti"].xcom_push(key="n_rows", value=int(df.shape[0]))
    context["ti"].xcom_push(key="n_cols", value=int(df.shape[1]))


def train_model(**context):
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        eval_metric="logloss",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    feature_names = list(X.columns)
    context["ti"].xcom_push(key="feature_names", value=feature_names)

    tmp_model_path = "/tmp/churn_candidate_model.json"
    model.save_model(tmp_model_path)
    context["ti"].xcom_push(key="tmp_model_path", value=tmp_model_path)

    tmp_test_path = "/tmp/churn_candidate_test.csv"
    test_df = X_test.copy()
    test_df["Churn"] = y_test.values
    test_df.to_csv(tmp_test_path, index=False)
    context["ti"].xcom_push(key="tmp_test_path", value=tmp_test_path)


def evaluate_model(**context):
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, f1_score

    tmp_model_path = context["ti"].xcom_pull(key="tmp_model_path", task_ids="train_model")
    tmp_test_path = context["ti"].xcom_pull(key="tmp_test_path", task_ids="train_model")

    test_df = pd.read_csv(tmp_test_path)
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    model = xgb.XGBClassifier()
    model.load_model(tmp_model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred))

    context["ti"].xcom_push(key="auc", value=auc)
    context["ti"].xcom_push(key="f1", value=f1)


def publish_candidate(**context):
    os.makedirs(BASELINE_DIR, exist_ok=True)

    new_version = _next_version()
    new_dir = os.path.join(CHURN_MODELS_DIR, new_version)
    os.makedirs(new_dir, exist_ok=True)

    tmp_model_path = context["ti"].xcom_pull(key="tmp_model_path", task_ids="train_model")
    feature_names = context["ti"].xcom_pull(key="feature_names", task_ids="train_model")
    auc = context["ti"].xcom_pull(key="auc", task_ids="evaluate_model")
    f1 = context["ti"].xcom_pull(key="f1", task_ids="evaluate_model")

    if not tmp_model_path or not os.path.exists(tmp_model_path):
        raise FileNotFoundError(f"tmp_model_path missing: {tmp_model_path}")

    if not feature_names or not isinstance(feature_names, list):
        raise ValueError("feature_names missing/invalid from XCom")

    context["ti"].xcom_push(key="new_version", value=new_version)

    
    model_out = os.path.join(new_dir, "model.json")
    with open(tmp_model_path, "rb") as src, open(model_out, "wb") as dst:
        dst.write(src.read())

   
    sig_out = os.path.join(new_dir, "signature.json")
    signature = {"features": feature_names}
    _atomic_write_json(sig_out, signature)

    
    m = _load_manifest()
    if not m.get("stable_version"):
        raise ValueError("manifest.json missing stable_version")

    m["candidate_version"] = new_version
    m["routing"] = DEFAULT_ROUTING
    m["last_candidate_published_utc"] = datetime.utcnow().isoformat()
    m["candidate_metrics"] = {"auc": auc, "f1": f1}

    _atomic_write_json(MANIFEST_PATH, m)


def create_baseline(**context):
    """
    Baseline = training feature distribution for PSI comparisons.
    Output: /opt/airflow/monitoring/baseline/{version}_baseline.csv
    """

    new_version = context["ti"].xcom_pull(key="new_version", task_ids="publish_candidate")
    feature_names = context["ti"].xcom_pull(key="feature_names", task_ids="train_model")

    if not new_version:
        raise ValueError("new_version missing from XCom (publish_candidate)")
    if not feature_names or not isinstance(feature_names, list):
        raise ValueError("feature_names missing/invalid from XCom (train_model)")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Training data missing features for baseline: {missing}")

    baseline_df = df[feature_names].copy()

    cap = int(os.getenv("BASELINE_MAX_ROWS", "5000"))
    if cap > 0 and len(baseline_df) > cap:
        baseline_df = baseline_df.sample(n=cap, random_state=42)

    os.makedirs(BASELINE_DIR, exist_ok=True)
    out_path = os.path.join(BASELINE_DIR, f"{new_version}_baseline.csv")
    baseline_df.to_csv(out_path, index=False)

    if baseline_df.empty:
        raise ValueError("Baseline dataframe is empty after selection/cap")

    context["ti"].xcom_push(key="baseline_path", value=out_path)


# ---- DAG ----

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="churn_train_eval_publish_candidate",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,   
    catchup=False,
    tags=["churn", "mlops"],
) as dag:

    t1 = PythonOperator(task_id="extract_data", python_callable=extract_data)
    t2 = PythonOperator(task_id="train_model", python_callable=train_model, execution_timeout=timedelta(minutes=30))
    t3 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t4 = PythonOperator(task_id="publish_candidate", python_callable=publish_candidate)
    t5 = PythonOperator(task_id="create_baseline", python_callable=create_baseline)

    t1 >> t2 >> t3 >> t4 >> t5
