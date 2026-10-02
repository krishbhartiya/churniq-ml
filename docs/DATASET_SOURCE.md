# Dataset Source

**Source:** IBM Telco Customer Churn sample dataset.

IBM's archived repository:
`https://github.com/IBM/telco-customer-churn-on-icp4d`

Raw CSV:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

The public file contains 7,043 customer rows and 21 raw columns.

The dataset is sample data for a fictional telecommunications company. It contains a churn label indicating whether the customer left within the relevant observation period.

The notebook downloads the raw file automatically and converts `TotalCharges` to numeric so blank entries become missing values.
