# streamlit
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score
import sys
import os

print("scikit-learn version:", sklearn.__version__)


# Add project root to PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import load_pipeline, predict_proba, predict_label
from src.data_prep import feature_engineer


st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📡 Telco Customer Churn Predictor")
st.write("Fill customer details and click Predict.")

# load pipeline once
pipeline = load_pipeline()


col1, col2, col3, col4 = st.columns(4)
with col1:
    customerID = st.text_input("Customer ID", "0001-ABCD")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

with col4:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=2)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=2000.0, value=89.10)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=178.2)

# Predict
if st.button("Predict Churn"):
    input_dict = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # get probability and label
    prob = predict_proba(input_dict, pipeline=pipeline)
    label = predict_label(input_dict, pipeline=pipeline)

    st.subheader("Result")
    st.write(f"**Churn probability:** {prob*100:.2f}%")
    if label == 1:
        st.error("Likely to churn")
    else:
        st.success("Not likely to churn")
