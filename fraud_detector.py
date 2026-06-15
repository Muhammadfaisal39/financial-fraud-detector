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

# -----------------------------------------------
# Day 2 — Understand and Clean the Data
# -----------------------------------------------

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sum(), "total missing values")

# -----------------------------------------------
# Look at the Amount column
# -----------------------------------------------

print("\n=== AMOUNT STATISTICS (ALL TRANSACTIONS) ===")
print(df['Amount'].describe())

# -----------------------------------------------
# Compare Normal vs Fraud transactions
# -----------------------------------------------

normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

print("\n=== AVERAGE TRANSACTION AMOUNT ===")
print(f"Normal transactions average: ${normal['Amount'].mean():.2f}")
print(f"Fraud transactions average:  ${fraud['Amount'].mean():.2f}")

print("\n=== MAX TRANSACTION AMOUNT ===")
print(f"Normal transactions max: ${normal['Amount'].max():.2f}")
print(f"Fraud transactions max:  ${fraud['Amount'].max():.2f}")

# -----------------------------------------------
# Scale the Amount and Time columns
# -----------------------------------------------

# V1-V28 are already on a similar scale (from PCA)
# But Amount and Time are on very different scales
# We need to bring them to the same scale for our model

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

print("\n=== SCALING DONE ===")
print(df[['Amount', 'Amount_scaled', 'Time', 'Time_scaled']].head())

print("\n✅ Data exploration and cleaning complete!")