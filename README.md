# Telecom Customer Churn Prediction — CHURNIQ 20-Mark ML Project

## What this package contains

This project is structured as an academic ML submission plus a polished Streamlit deployment.

### Academic core
`notebooks/Telecom_Customer_Churn_20_Marks.ipynb`

The notebook covers the full ML workflow:
- problem statement and dataset audit
- missing-value handling for `TotalCharges`
- EDA and churn analysis
- preprocessing with `ColumnTransformer` and `Pipeline`
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting
- Accuracy, Precision, Recall, F1 and ROC-AUC
- confusion matrices
- Stratified 5-fold cross-validation
- recall-first model selection
- feature/association analysis
- case-study answers and limitations

### Deployment
`deployment/app.py`

The web application is **CHURNIQ**, a dark, compact telecom intelligence dashboard designed for a clean viva demonstration.

## Run

```bash
python3 -m pip install -r deployment/requirements.txt
python3 -m streamlit run deployment/app.py
```

Open `http://localhost:8501`.

## Web app structure

### 1. Overview
A compact executive dashboard with project telemetry, contract/churn signal charts and the five-stage ML workflow.

### 2. Customer Analytics
Interactive filters, churn-rate analysis, tenure distribution, service-level inspection and a responsive 3D customer charge/tenure landscape.

### 3. Model Lab
All five required models are compared using Accuracy, Precision, Recall, F1 and ROC-AUC. The page also includes selectable confusion matrices, Stratified 5-fold CV and a 3D Precision/Recall/F1 metric landscape.

### 4. Predictor
The left sidebar contains the **Model Control** from the beginning. You can choose:
- Auto — Recall-first
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting

The predictor then uses that selected model. It exposes the probability threshold, risk band, model probability, explanation, customer snapshot and threshold mechanics.

Two demo modes use real dataset records selected by the active model:
- Highest-risk demo
- Lowest-risk demo

This makes the viva demonstration reproducible without guessing a special combination of dropdown values.

### 5. Methodology
A concise viva reference covering dataset, preprocessing, algorithms, metrics and limitations.

## Dataset

IBM Telco Customer Churn sample dataset:
- 7,043 customers
- 21 raw columns
- `Churn` is the target
- `customerID` is excluded as an identifier
- `TotalCharges` is converted from text to numeric and median-imputed inside the preprocessing pipeline

The CSV is downloaded automatically if `data/Telco-Customer-Churn.csv` does not exist.

## Model-selection rule

The automatic model is selected by:
1. highest hold-out **Recall**
2. **F1-score** as the secondary tie-breaker
3. Stratified 5-fold cross-validation is shown as stability evidence

The sidebar still lets you deliberately switch to any required model for demonstration and prediction.

## Interpretation note

The project is educational. Predictive associations are not causal claims. Production use would require calibration, threshold tuning against business costs, privacy controls, fairness checks and drift monitoring.
