import streamlit as st
import requests
import json
import os

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
HEALTH_URL = f"{API_BASE}/health"
API_URL = f"{API_BASE}/predict"

VALID_CONTRACTS = ["Month-to-month", "One year", "Two year"]
VALID_INTERNET = ["DSL", "Fiber optic", "No"]
VALID_PAYMENT = [
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
]

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")

st.caption(f"API_BASE = {API_BASE}")

with st.expander("API status"):
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        st.json(r.json())
    except Exception as e:
        st.error(f"Health check failed: {e}")

st.divider()
st.subheader("Customer Inputs")


user_id = st.text_input("UserId (optional)", value="")

col1, col2 = st.columns(2)


with col1:
    Gender_label = st.selectbox("Gender", ["Male", "Female"])
    Senior_label = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner_label = st.selectbox("Has Partner", ["No", "Yes"])
    Dependents_label = st.selectbox("Has Dependents", ["No", "Yes"])
    Phone_label = st.selectbox("Phone Service", ["No", "Yes"])
    Multiple_label = st.selectbox("Multiple Lines", ["No", "Yes"])
    Security_label = st.selectbox("Online Security", ["No", "Yes"])
    Backup_label = st.selectbox("Online Backup", ["No", "Yes"])
    

with col2:
    Support_label = st.selectbox("Tech Support", ["No", "Yes"])
    TV_label = st.selectbox("Streaming TV", ["No", "Yes"])
    Movies_label = st.selectbox("Streaming Movies", ["No", "Yes"])
    Paper_label = st.selectbox("Paperless Billing", ["No", "Yes"])
    Protection_label = st.selectbox("Device Protection", ["No", "Yes"])

    Tenure = st.number_input("Tenure (months)", 0.0, 120.0, 12.0, step=1.0)
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 100000.0, 1500.0, step=10.0)

Contract = st.selectbox("Contract", VALID_CONTRACTS)
InternetService = st.selectbox("Internet Service", VALID_INTERNET)
PaymentMethod = st.selectbox("Payment Method", VALID_PAYMENT)


def yes_no_encode(val):
    return 1 if val == "Yes" else 0

payload = {
    **({"UserId": user_id.strip()} if user_id.strip() else {}),
    "Gender": 1 if Gender_label == "Female" else 0,  
    "SeniorCitizen": yes_no_encode(Senior_label),
    "Partner": yes_no_encode(Partner_label),
    "Dependents": yes_no_encode(Dependents_label),
    "PhoneService": yes_no_encode(Phone_label),
    "MultipleLines": yes_no_encode(Multiple_label),
    "OnlineSecurity": yes_no_encode(Security_label),
    "OnlineBackup": yes_no_encode(Backup_label),
    "DeviceProtection": yes_no_encode(Protection_label),
    "TechSupport": yes_no_encode(Support_label),
    "StreamingTV": yes_no_encode(TV_label),
    "StreamingMovies": yes_no_encode(Movies_label),
    "PaperlessBilling": yes_no_encode(Paper_label),
    "Tenure": float(Tenure),
    "MonthlyCharges": float(MonthlyCharges),
    "TotalCharges": float(TotalCharges),
    "Contract": Contract,
    "InternetService": InternetService,
    "PaymentMethod": PaymentMethod,
}

st.divider()
st.subheader("Request Payload")
st.code(json.dumps(payload, indent=2), language="json")

if st.button("Predict", type="primary"):
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            out = resp.json()
            st.success("Prediction completed")
            st.json(out)

            pred = out.get("prediction")
            prob = out.get("probability_percent")
            mv = out.get("model_version")

            st.write(f"**Model version:** {mv}")
            st.write(f"**Prediction:** {'Churn' if pred == 1 else 'No Churn'}")
            st.write(f"**Probability:** {prob}%")

    except Exception as e:
        st.error(f"Request failed: {e}")
