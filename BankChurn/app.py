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
import streamlit as st

import streamlit as st

# Set page config
st.set_page_config(page_title="Bank Churn Prediction", layout="centered")

# Custom styling using markdown (Streamlit supports CSS injection through unsafe_allow_html)
st.markdown("""
    <style>
    .doc-button {
        display: inline-block;
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 30px;
        border-radius: 10px;
        text-decoration: none;
        text-align: center;
        margin-top: 20px;
        transition: all 0.3s ease;
    }

    .doc-button:hover {
        background: linear-gradient(90deg, #005ce6 0%, #00b8e6 100%);
        transform: scale(1.05);
    }

    .center {
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# Title
st.title("💼 Bank Customer Churn Prediction")

# Subheading
st.subheader("📊 Understand & Predict Why Customers Leave")

# Description
st.markdown("""
This interactive application demonstrates how Machine Learning can be applied to predict customer churn in the banking sector.  
It includes stages like:
- Data preprocessing
- Feature encoding
- Model training and evaluation
- And final predictions

Built using **Streamlit**, this app brings AI-powered insights right to your screen.
""")

# CTA Button - Gamma Documentation
st.markdown("""
<div class="center">
    <a href="https://gamma.app/docs/Bank-Customer-Churn-Prediction-12hm02f12jtwndi?mode=doc" class="doc-button" target="_blank">
        📄 View Full Project Documentation
    </a>
</div>
""", unsafe_allow_html=True)

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

