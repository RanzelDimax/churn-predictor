
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

st.title("🔮 Customer Churn Predictor")
st.markdown("Enter customer details below to predict if they are likely to churn.")

# Input fields
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Raw input dictionary
raw_input = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': "Yes",  # default, to avoid missing in encoding
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Create DataFrame
input_df = pd.DataFrame([raw_input])

# Perform one-hot encoding to match training
input_encoded = pd.get_dummies(input_df)

# Add any missing columns from training and set them to 0
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure column order matches training
input_encoded = input_encoded[feature_names]

# Scale numeric features
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **churn**. (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ This customer is likely to **stay**. (Confidence: {1 - probability:.2%})")

