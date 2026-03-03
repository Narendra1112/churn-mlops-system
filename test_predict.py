import json
import requests

payload = {
    "UserId": "tag_test",
    "Gender": 1,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "Tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
}

print("SENT:", json.dumps(payload, indent=2))

r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
print("STATUS:", r.status_code)
print("RECV:", r.text)
