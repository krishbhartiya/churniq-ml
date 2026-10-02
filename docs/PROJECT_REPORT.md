# Telecom Customer Churn Prediction — Project Report

## 1. Introduction
Telecom churn prediction is a binary classification problem in which the model estimates whether a customer is likely to leave the service. The purpose is to support retention teams by prioritizing customers for further review.

## 2. Dataset
The project uses the IBM Telco Customer Churn sample dataset. It contains 7,043 customer records and 21 raw columns covering demographics, tenure, services, contract and billing attributes. The target is `Churn`.

## 3. Data Preprocessing
`customerID` is removed as an identifier. `TotalCharges` is converted from text to numeric, causing blank values to become missing. Numeric missing values are median-imputed; categorical values are most-frequent-imputed. Categorical features are one-hot encoded and numeric features are standardized. All transformations are inside a scikit-learn Pipeline.

## 4. Exploratory Data Analysis
The notebook examines churn distribution, contract type, tenure, monthly charges, internet service and payment method. Contract and tenure analyses directly address the case-study questions about churn relationships.

## 5. Machine Learning
Five required classifiers are trained:
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting

## 6. Evaluation
Accuracy, Precision, Recall, F1-score and confusion matrices are used. ROC-AUC is included as an additional metric. Stratified 5-fold cross-validation evaluates performance stability.

## 7. Model Selection
Recall is the primary selection criterion because missing a true churner is important for the retention use case. F1-score and cross-validation results are used as supporting evidence.

## 8. Deployment
A Streamlit application trains the same model-selection workflow and allows a user to enter customer subscription information. It displays High/Low churn risk, estimated probability, selected model and an interpretable explanation based on important profile characteristics.

## 9. Limitations
The IBM data is sample data for a fictional telecom company and is cross-sectional. It is not a live production dataset or a sequence of future observations. Predictions are probabilistic and associations do not prove causation. Production deployment would require real data, calibration, drift monitoring, privacy, fairness and business validation.

## 10. Conclusion
The project demonstrates an end-to-end ML workflow from data loading and EDA through preprocessing, model comparison, cross-validation, recall-focused selection and interactive deployment.
