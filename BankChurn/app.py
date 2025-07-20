import streamlit as st
import pickle
import numpy as np
import os

# Get the current file's directory (safe for deployment)
base_dir = os.path.dirname(__file__)

# Join with your model and scaler files
model_path = os.path.join(base_dir, 'churn_model_rf.pkl')
scaler_path = os.path.join(base_dir, 'rf_scaler.pkl')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Bank Churn Predictor", layout="centered")
st.title("🔍 Customer Churn Predictor")

st.markdown("This tool predicts whether a customer is likely to leave the bank (churn) based on input features.")

# Input form
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
age = st.number_input("Age", min_value=18, max_value=100)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)
balance = st.number_input("Account Balance", min_value=0.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=5)
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)

# Encode categorical inputs
credit_card = 1 if credit_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0

# Prepare features
features = np.array([[credit_score, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]])

# Scale features
scaled_features = scaler.transform(features)

# Predict button
if st.button("Predict"):
    result = model.predict(scaled_features)
    if result[0] == 1:
        st.error("❌ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")
