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

# -----------------------------------------------
# Day 3 — Train the Isolation Forest Model
# -----------------------------------------------

from sklearn.ensemble import IsolationForest

# -----------------------------------------------
# Prepare the features (input) for the model
# -----------------------------------------------

# We use V1-V28 plus our scaled Amount and Time
# We drop the original Amount, Time, and Class
features = [col for col in df.columns if col not in ['Time', 'Amount', 'Class']]

X = df[features]

print("\n=== FEATURES USED FOR TRAINING ===")
print(features)
print("\nTotal features:", len(features))

# -----------------------------------------------
# Train the Isolation Forest
# -----------------------------------------------

# contamination = expected proportion of anomalies (fraud) in the data
# We know fraud is about 0.17% from Day 1, so we set it close to that
model = IsolationForest(
    n_estimators=100,
    contamination=0.0017,
    random_state=42
)

print("\n⏳ Training Isolation Forest... (this may take a minute)")
model.fit(X)

print("✅ Model trained successfully!")

# -----------------------------------------------
# Make predictions
# -----------------------------------------------

# The model returns: 1 = normal, -1 = anomaly (flagged as fraud)
predictions = model.predict(X)

# Convert to easier format: 0 = normal, 1 = flagged as anomaly
df['anomaly'] = [1 if p == -1 else 0 for p in predictions]

print("\n=== MODEL PREDICTIONS ===")
print(df['anomaly'].value_counts())

flagged_count = df['anomaly'].sum()
print(f"\nTotal transactions flagged as anomalies: {flagged_count}")