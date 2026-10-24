import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="AI Credit Risk Assessment", layout="wide")

st.title("🏦 AI-Powered Credit Risk & Loan Eligibility System")
st.markdown("Automated applicant risk evaluation platform for retail loan underwriting.")

# Load Model Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('credit_model_pipeline.pkl')

try:
    pipeline = load_pipeline()
    model = pipeline['model']
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")

# Sidebar Form for User Inputs
st.sidebar.header("📋 Applicant Profile Data")

income = st.sidebar.number_input("Total Annual Income ($)", min_value=10000, value=150000, step=5000)
credit = st.sidebar.number_input("Requested Credit Amount ($)", min_value=10000, value=400000, step=10000)
annuity = st.sidebar.number_input("Annual Loan Annuity ($)", min_value=1000, value=25000, step=1000)
age = st.sidebar.slider("Applicant Age (Years)", 18, 70, 35)
employment_years = st.sidebar.slider("Years Employed", 0, 40, 5)

st.sidebar.subheader("External Credit Scores")
ext_source_2 = st.sidebar.slider("External Bureau Score 2", 0.0, 1.0, 0.5)
ext_source_3 = st.sidebar.slider("External Bureau Score 3", 0.0, 1.0, 0.5)

# Feature Calculations
days_employed = - (employment_years * 365)
credit_income_percent = credit / income
annuity_income_percent = annuity / income

input_data = pd.DataFrame([{
    'AMT_INCOME_TOTAL': income,
    'AMT_CREDIT': credit,
    'AMT_ANNUITY': annuity,
    'DAYS_EMPLOYED': days_employed,
    'EXT_SOURCE_2': ext_source_2,
    'EXT_SOURCE_3': ext_source_3,
    'CREDIT_INCOME_PERCENT': credit_income_percent,
    'ANNUITY_INCOME_PERCENT': annuity_income_percent,
    'AGE_YEARS': age
}])

# Prediction Trigger
if st.button("Evaluate Credit Risk"):
    proba = model.predict_proba(input_data)[0][1]
    risk_score = round(proba * 100, 2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Default Risk Score", value=f"{risk_score}%")
        if risk_score < 35:
            st.success("Recommendation: LOW RISK — Approve Loan")
        elif risk_score < 65:
            st.warning("Recommendation: MEDIUM RISK — Manual Review Required")
        else:
            st.error("Recommendation: HIGH RISK — Reject Application")

    with col2:
        st.subheader("Financial Ratios")
        st.write(f"**Credit to Income Ratio:** {credit_income_percent:.2f}x")
        st.write(f"**Annuity to Income Ratio:** {annuity_income_percent:.2%}")

    # Prediction Logging
    log_file = "prediction_logs.csv"
    log_entry = input_data.copy()
    log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry['risk_score'] = risk_score
    
    if not os.path.isfile(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)
        
    st.info("Assessment result logged successfully.")