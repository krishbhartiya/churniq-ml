DATASET
=======

The app expects the IBM Telco Customer Churn CSV at:

data/Telco-Customer-Churn.csv

If this file is not present, the Streamlit app attempts to download it automatically from the
IBM Telco Customer Churn sample repository. If automatic download is blocked, download the
CSV once and place it here with the exact filename above, then restart Streamlit.

Required columns include:
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn
