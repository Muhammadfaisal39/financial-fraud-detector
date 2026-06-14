# Financial Fraud Detector
# Day 1 — Load and Explore the Data

import pandas as pd

# Load the dataset
df = pd.read_csv('creditcard.csv')

print("=== DATASET OVERVIEW ===")
print("Total transactions:", len(df))
print("Total columns:", len(df.columns))
print("\nColumn names:", df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())

# -----------------------------------------------
# Check how many transactions are fraud vs normal
# -----------------------------------------------

print("\n=== FRAUD vs NORMAL TRANSACTIONS ===")
print(df['Class'].value_counts())

fraud_count = df['Class'].sum()
total_count = len(df)
fraud_percentage = (fraud_count / total_count) * 100

print(f"\nTotal transactions: {total_count}")
print(f"Fraud transactions: {fraud_count}")
print(f"Fraud percentage: {fraud_percentage:.4f}%")