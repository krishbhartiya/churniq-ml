# Viva Script — 20 Marks

## Opening
“My project is Telecom Customer Churn Prediction using Machine Learning. It is a binary classification problem where the target is whether a telecom customer churns or stays.”

## Dataset
“I used the IBM Telco Customer Churn sample dataset containing 7,043 customers and 21 raw columns. It contains demographic, service, contract and billing information.”

## Preprocessing
“TotalCharges is stored as text, so I converted it to numeric and treated blank values as missing. Numeric values are median-imputed and standardized. Categorical variables are one-hot encoded using a ColumnTransformer and Pipeline.”

## Algorithms
“I trained Logistic Regression, KNN, Decision Tree, Random Forest and Gradient Boosting.”

## Metrics
“I compared Accuracy, Precision, Recall and F1-score and also inspected confusion matrices. I additionally used ROC-AUC.”

## Recall
“Recall is especially important because a false negative means a real churner was not identified. The case study explicitly asks us to emphasize this metric, so I use recall as the primary model-selection criterion.”

## Cross-validation
“I used Stratified 5-fold cross-validation to check whether performance is reasonably stable across multiple train-validation splits.”

## Feature analysis
“I examined contract, tenure, charges and service attributes for predictive associations. These are associations, not causal claims.”

## Deployment
“The selected workflow is exposed through a Streamlit application. A user enters customer attributes and receives High or Low risk, churn probability and an explanation of the important entered characteristics.”

## Limitation
“The dataset is a sample cross-sectional dataset, so it is not a temporal forecast. Production use would require real operational data, calibration, monitoring, privacy and business validation.”
