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

# -----------------------------------------------
# Day 4 — Evaluate the Model
# -----------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

print("\n=== MODEL EVALUATION ===")

# Compare model predictions against real fraud labels
y_true = df['Class']      # Real labels (0=normal, 1=fraud)
y_pred = df['anomaly']    # Model predictions (0=normal, 1=flagged)

# -----------------------------------------------
# Precision, Recall, F1
# -----------------------------------------------

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.1f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.1f}%)")

print("\nIn plain English:")
print(f"Out of transactions we flagged — {precision*100:.1f}% were real fraud")
print(f"Out of all real fraud — we caught {recall*100:.1f}% of them")

print("\n=== DETAILED REPORT ===")
print(classification_report(y_true, y_pred,
      target_names=['Normal', 'Fraud']))

# -----------------------------------------------
# Confusion Matrix
# -----------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actually Normal', 'Actually Fraud'])
plt.title('Confusion Matrix — Fraud Detection Results', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\n✅ Confusion matrix saved as confusion_matrix.png")

# -----------------------------------------------
# Summary in plain English
# -----------------------------------------------

tn = cm[0][0]  # Normal correctly identified
fp = cm[0][1]  # Normal wrongly flagged as fraud
fn = cm[1][0]  # Fraud missed by model
tp = cm[1][1]  # Fraud correctly caught

print("\n=== RESULTS SUMMARY ===")
print(f"✅ Real fraud caught:           {tp} out of {tp+fn}")
print(f"❌ Real fraud missed:           {fn} out of {tp+fn}")
print(f"⚠️  Innocent transactions flagged: {fp} out of {tn+fp}")
print(f"✅ Normal transactions cleared: {tn} out of {tn+fp}")