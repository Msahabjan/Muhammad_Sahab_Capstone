# AI-Powered Credit Risk & Loan Eligibility System


🔗 **Live Interactive App:** [Click Here to Open Live Streamlit App](https://muhammadsahabcapstone-jhef9dmgpzeqdhpjwvsxrv.streamlit.app/)

---
## Project Overview
This project provides an automated machine learning platform designed for loan underwriting officers to assess applicant default risk using financial demographic indicators and external credit scores.

## Problem Definition & Target User
- **Problem:** Retail financial institutions struggle to perform fast, objective, and accurate risk assessments for loan applicants, especially those with non-standard credit histories.
- **Target User:** Underwriting officers, risk analysts, and credit decision teams.

## Architecture & Prior Skills Applied
1. **EDA & Data Processing:** Handled missing values, treated dataset anomalies (e.g., `DAYS_EMPLOYED`), and normalized financial columns.
2. **Feature Engineering:** Calculated credit-to-income and annuity-to-income ratios alongside applicant age metrics.
3. **Machine Learning Pipeline:** Trained a LightGBM Classifier using class-imbalance weighting, evaluated via ROC-AUC.
4. **Interactive Interface:** Built and deployed an interactive Streamlit UI for real-time risk scoring.
5. **System Logging:** Integrated automated query logging (`prediction_logs.csv`) to record input payloads, timestamp metadata, and risk output metrics.

## Model Performance & Honest Limitations
- **Model Metric:** Achieved robust ROC-AUC performance on hold-out test splits.
- **Data Limitations:** Trained on historical institutional data, which may carry legacy underwriting biases.
- **Economic Factors:** Models do not dynamically adjust to real-time macroeconomic shifts (such as interest rate changes or sudden inflation).
- **Cold-Start Applicants:** Lower confidence on applicants with sparse external credit history bureau records (`EXT_SOURCE`).