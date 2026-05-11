# Telco Churn Early Warning System

This project predicts which telecom customers are most likely to churn and helps prioritize them for retention outreach. We used the IBM Telco Customer Churn dataset and built a machine learning pipeline that compares Logistic Regression, Random Forest, and Gradient Boosting models.

The main goal was not just to predict churn, but to make the results useful for decision-making. The Streamlit dashboard lets users adjust the churn threshold, compare precision and recall trade-offs, view model performance, and explore which features are driving churn risk.

## What the project does

- Cleans and preprocesses the Telco Customer Churn dataset
- Handles class imbalance using class-weighted learning and SMOTE comparison
- Trains Logistic Regression, Random Forest, and Gradient Boosting models
- Evaluates models using recall, precision, F1-score, ROC-AUC, and confusion matrices
- Ranks customers by churn probability
- Provides an interactive Streamlit dashboard for exploring results

## Main files

- `dashboard.py` — Streamlit dashboard for viewing model results and churn predictions
- `train_and_save.py` — training script used to build and save model artifacts
- `dashboard_artifacts.pkl` — saved models and outputs used by the dashboard
- `requirements.txt` — Python packages needed to run the project
- `neurips_2026.tex` — final project report template in LaTeX format

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
