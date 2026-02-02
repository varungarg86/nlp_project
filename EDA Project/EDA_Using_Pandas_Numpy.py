
# #Demo 1: Handling Missing Values and Outliers in a Dataset Using Pandas/Numpy
# 
# 
# #**Scenario: Customer Purchase Behavior Analysis**
# 
# A retail company is analyzing customer purchase data to understand spending patterns and predict future sales. However, the dataset contains missing values and outliers due to various reasons:
# 
# * Some customers did not provide their income details while registering.
# 
# * Certain purchases have extremely high or low values, possibly due to data entry errors or fraud.
# 
# To ensure accurate insights, handling missing values and outliers is crucial before proceeding with data analysis and machine learning models.
# 
# ##**Objective**
# Identify and handle missing values in important features like customer income and purchase amount.
# 
# * Detect and treat outliers in numerical columns such as purchase amount and transaction frequency.
# 
# * Ensure data quality and reliability for accurate decision-making and predictive modeling.


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("customer_purchase_behavior.csv")

# Display first few rows to understand the dataset
print("Initial Dataset:\n", df.head())

df.shape


df.describe

# Check for missing values
print("\nMissing Values Before Handling:\n", df.isnull().sum())


# Strategy 1: Fill missing Income with the median value (better for skewed data)
df['Income'].fillna(df['Income'].median(), inplace=True)

# Alternative: You could use df['Income'].fillna(df['Income'].mean(), inplace=True)


# Verify missing values are handled
print("\nMissing Values After Handling:\n", df.isnull().sum())


# Function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]


# Detect outliers in 'PurchaseAmount' and 'TransactionsPerMonth'
outliers_purchase = detect_outliers_iqr(df, 'PurchaseAmount')
outliers_transactions = detect_outliers_iqr(df, 'TransactionsPerMonth')

print("\nOutliers in PurchaseAmount:\n", outliers_purchase)
print("\nOutliers in TransactionsPerMonth:\n", outliers_transactions)


# Define a function to cap outliers using IQR limits
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])


# Apply capping on 'PurchaseAmount' and 'TransactionsPerMonth'
cap_outliers(df, 'PurchaseAmount')
cap_outliers(df, 'TransactionsPerMonth')


# Display dataset after handling outliers
print("\nDataset after Handling Outliers:\n", df.head())


# Save the cleaned dataset
df.to_csv("cleaned_customer_purchase_behavior.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_customer_purchase_behavior.csv'")



