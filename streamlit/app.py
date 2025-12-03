import os
import streamlit as st
import requests

API_URL =  "https://localhost:8000/predict"

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

# Prepare payload – matches FastAPI request model
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
        res = requests.post(API_URL, json=payload, timeout=20)

        if res.status_code == 200:
            output = res.json()

            # Try to read probability from either key
            prob = output.get("probability_percent") or output.get("churn_probability")

            raw_pred = output.get("prediction")

            # Normalize prediction: handle 0/1 or string labels
            is_churn = False
            if isinstance(raw_pred, (int, float)):
                is_churn = int(raw_pred) == 1
            elif isinstance(raw_pred, str):
                label = raw_pred.lower()
                is_churn = label in ("churn", "yes", "will churn", "customer is likely to churn")

            st.subheader("Prediction Result")

            if is_churn:
                if prob is not None:
                    st.error(f"Customer WILL CHURN (Probability: {prob}%)")
                else:
                    st.error("Customer WILL CHURN")
            else:
                if prob is not None:
                    st.success(f"Customer WILL NOT CHURN (Probability: {prob}%)")
                else:
                    st.success("Customer WILL NOT CHURN")

            # Optional: show raw response for debugging
            # st.json(output)

        else:
            st.error(f"API returned status {res.status_code}: {res.text}")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
