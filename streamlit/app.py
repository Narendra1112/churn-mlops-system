import os
import streamlit as st
import requests

# ------------------------------------------------------------------
# API base URL
# - If running Streamlit locally: defaults to http://localhost:8000
# - If running Streamlit in Docker: set API_BASE=http://churn_api:8000
#   in the streamlit service environment in docker-compose.
# ------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_URL = f"{API_BASE}/predict"

st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")

st.title("Customer Churn Prediction")
st.write("Enter customer details below to get churn prediction.")

# Input fields
MonthlyCharges = st.number_input(
    "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
)
Tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, value=12)
TotalCharges = st.number_input(
    "Total Charges", min_value=0.0, max_value=10000.0, value=500.0
)

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Credit card (automatic)"],
)

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])

# Prepare payload
payload = {
    "MonthlyCharges": MonthlyCharges,
    "Tenure": Tenure,
    "TotalCharges": TotalCharges,
    "Contract": Contract,
    "InternetService": InternetService,
    "PaymentMethod": PaymentMethod,
    "PaperlessBilling": PaperlessBilling,
    "MultipleLines": MultipleLines,
    "OnlineBackup": OnlineBackup,
}

st.caption(f"Using API at: {API_URL}")

if st.button("Predict Churn"):
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        if res.status_code == 200:
            output = res.json()
            pred = output["prediction"]
            prob = output["probability_percent"]

            st.subheader("Prediction Result")
            if pred == 1:
                st.error(f"Customer WILL CHURN (Probability: {prob}%)")
            else:
                st.success(f"Customer WILL NOT CHURN (Probability: {prob}%)")
        else:
            st.error(f"API returned status {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
