import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

BASE_DIR = "/opt/airflow"
DATA_PATH = f"{BASE_DIR}/data/Telecom_processed.csv"
MLRUNS_PATH = f"{BASE_DIR}/mlruns"
MODEL_DIR = f"{BASE_DIR}/models"
MODEL_PATH = f"{MODEL_DIR}/xgb_churn_best.pkl"

EXPERIMENT_NAME = "Churn_MLOps_Final"
REGISTERED_MODEL_NAME = "ChurnPredictionModel"

def train_and_log_churn_model():
    import pandas as pd
    import joblib
    import mlflow
    import mlflow.xgboost
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score
    )

    # Load data
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # FIXED: Compatible XGBoost version + autolog
    mlflow.xgboost.autolog()  # Auto-logs params/metrics
    
    model = xgb.XGBClassifier(
        eval_metric="logloss",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics (autologged but explicit for safety)
    auc = roc_auc_score(y_test, y_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # MLflow logging
    os.makedirs(MLRUNS_PATH, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="airflow_retrain"):
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("balanced_accuracy", bal_acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

    
    # Save .pkl for FastAPI
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")  # Debug log

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "churn_retraining_daily",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["churn", "mlops"],
) as dag:

    retrain = PythonOperator(
        task_id="train_and_log_model",
        python_callable=train_and_log_churn_model,
        execution_timeout=timedelta(minutes=30),  # FIXED: Prevent timeout kills
    )
