import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load model and scaler using dynamic path
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'churn_model_rf.pkl')
scaler_path = os.path.join(base_path, 'rf_scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Customer Churn Prediction App")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)
balance = st.number_input("Account Balance", min_value=0.0, step=100.0)
products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)
country = st.selectbox("Country", ["France", "Germany", "Spain"])

# Convert categorical inputs
gender = 1 if gender == "Male" else 0
credit_card = 1 if credit_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0
country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0

# Build the feature array in correct order
features = pd.DataFrame([[
    credit_score, gender, age, tenure, balance, products_number,
    credit_card, active_member, estimated_salary, country_Germany, country_Spain
]], columns=[
    'credit_score', 'gender', 'age', 'tenure', 'balance', 'products_number',
    'credit_card', 'active_member', 'estimated_salary', 'country_Germany',
    'country_Spain'
])

# Scale the features
scaled_features = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_features)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

