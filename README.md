# Customer Churn Prediction — Full MLOps Pipeline

(MLflow, Airflow, FastAPI, Streamlit, Prometheus, Grafana, Docker, Data Drift Detection)

This repository implements a production-grade MLOps system for telecom customer churn prediction, built with a fully automated pipeline that handles model training, retraining, versioning, deployment, monitoring, and drift detection.

This project demonstrates how real companies deploy, observe, retrain, and maintain ML models in production, combining machine learning with robust DevOps and observability practices.

##  Live Demo

**Deployed Application:** [Try it here on Render](https://churn-mlops-system-1.onrender.com/)  
This live web application provides real-time telecom churn prediction using a production-ready pipeline powered by **FastAPI** and **Streamlit**.
Users can submit customer details, receive instant churn probabilities, and explore:

Model inference in real time (FastAPI)

User-friendly prediction UI (Streamlit)

API-level metrics via Prometheus

Grafana dashboards for latency, throughput, and churn trends

Data drift insights from the drift monitoring module
![Streamlit App Demo](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/Streamlit.png)





## Project Overview

Customer churn prediction helps identify users who are likely to discontinue a service—critical for telecom providers, subscription platforms (e.g., Netflix, Amazon Prime), and any customer-retention-driven business.
By predicting churn early, companies can intervene with personalized offers, better support, or loyalty incentives to reduce revenue loss.

This project delivers a complete end-to-end MLOps pipeline for churn prediction, integrating both automation and observability


## Tech Stack

- **Machine Learning & Data**
  - Python, XGBoost, Scikit‑learn, Pandas, NumPy

- **Model Tracking**
  - MLflow (experiments, metrics, artifacts)

- **Orchestration**
  - Apache Airflow (daily automated retraining DAG)

- **Model Serving**
  - FastAPI (real‑time prediction API)

- **Frontend / UI**
  - Streamlit (interactive churn dashboard)

- **Monitoring & Observability**
  - Prometheus (API + system metrics)
  - Grafana (dashboards and alerts)
  - cAdvisor (container CPU/memory)

- **Data Quality / Drift**
  - Custom drift module (KS test + JS divergence, JSON/HTML reports)

- **Deployment & DevOps**
  - Docker, Docker Compose
  - Environment variables for configuration



# Dataset

The dataset used in this project comes from the [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
, which is also available on [IBM sample data](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?).
The processed dataset contains engineered features including:
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Service usage: PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Billing and tenure: Tenure, MonthlyCharges, TotalCharges, PaperlessBilling
- One‑hot encoded categories for InternetService, Contract, and PaymentMethod
  (e.g., InternetService_Fiber_optic, Contract_One_year, PaymentMethod_Electronic_check)


Churn Indicator: whether the customer discontinued the service in the previous month

## How it was used

The dataset was cleaned and preprocessed to handle missing values and encode categorical variables. Feature importance was analyzed using XGBoost, and the top 10 predictors (such as tenure, monthly charges, and contract type) were selected for model training. The final model was deployed via FastAPI for real-time churn prediction and visualized through Streamlit and Grafana dashboards. The processed dataset is also used by the Airflow retraining DAG and the data‑drift monitoring script.

## Features

- Real‑time churn prediction API built with **FastAPI**.
- **Streamlit** dashboard for live user input and prediction visualization.
- **MLflow** experiment tracking with metrics and stored model artifacts.
- Manual **data drift monitoring** (KS test + JS divergence) with JSON/HTML reports.
- **Prometheus** metrics for request counts, latency histograms, and container stats.
- **Grafana** dashboards: churn ratio, prediction volume, P95 latency, CPU/memory usage.
- **Airflow** DAG for automated daily model retraining (optional but included).
- **Docker‑based** setup (Docker Compose) for full local reproducibility.


##  Run Locally

git clone https://github.com/Narendra1112/churn-mlops-system.git
cd churn-mlops-system
**Install dependencies**
pip install -r requirements.txt
**Start FastAPI backend**
uvicorn api.app:app --reload
**Start Streamlit UI**
streamlit run streamlit/app.py
**Start monitoring stack (FastAPI + Prometheus + Grafana)**
docker-compose up -d
**Optional: Enable automated daily retraining (Airflow)**
docker-compose -f docker-compose-airflow.yml up -d
**Optional: Start MLflow experiment tracking**
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000


# API Example

## Input

{
  "MonthlyCharges": 25.0,
  "Tenure": 25,
  "TotalCharges": 500.0,
  "Contract": "Month-to-month",
  "InternetService": "DSL",
  "PaymentMethod": "Electronic check",
  "PaperlessBilling": "Yes",
  "MultipleLines": "Yes",
  "OnlineBackup": "Yes"
}


## Output

{
  "prediction": 1,
  probability_percent": 74.25
}

## Monitoring Views

FastAPI Docs: http://localhost:8000/docs
![image](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/FastAPI.png)

Streamlit UI: http://localhost:8501
![image](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/Streamlit.png)

Prometheus: http://localhost:9090
![image](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/Prometheus.png)

Grafana: http://localhost:3000
![image](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/Grafana.png)

Airflow: http://localhost:8080
![image](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/Airflow.png)

Mlflow: http://localhost:5000
![image](https://github.com/Narendra1112/churn-mlops-system/blob/main/assests/Mlflow.png)



## Key Insights

- Short‑tenure customers are more likely to churn.
- Month‑to‑month contracts and higher monthly charges correlate with increased churn risk.
- Automatic payments and long‑term contracts are strong signals of customer retention.

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with appropriate attribution.
