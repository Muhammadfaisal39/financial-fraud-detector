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

# -----------------------------------------------
# Day 5 — Fix Imbalance with SMOTE
# -----------------------------------------------

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score)

print("\n=== DAY 5: FIXING IMBALANCED DATASET WITH SMOTE ===")

# -----------------------------------------------
# Prepare data
# -----------------------------------------------

# Separate features and labels
X_data = df[features]
y_data = df['Class']

print(f"Before SMOTE — Normal: {sum(y_data==0)}, Fraud: {sum(y_data==1)}")

# -----------------------------------------------
# Split into train and test BEFORE applying SMOTE
# -----------------------------------------------

# Important: we apply SMOTE only on training data
# Test data must stay real — no synthetic samples
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data  # keeps fraud ratio same in both splits
)

print(f"\nTraining set — Normal: {sum(y_train==0)}, Fraud: {sum(y_train==1)}")
print(f"Test set     — Normal: {sum(y_test==0)},  Fraud: {sum(y_test==1)}")

# -----------------------------------------------
# Apply SMOTE to training data only
# -----------------------------------------------

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE — Normal: {sum(y_train_balanced==0)}, "
      f"Fraud: {sum(y_train_balanced==1)}")
print("✅ Dataset is now balanced!")

# -----------------------------------------------
# Train Random Forest on balanced data
# -----------------------------------------------

print("\n⏳ Training Random Forest on balanced data...")

rf_balanced = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_balanced.fit(X_train_balanced, y_train_balanced)

print("✅ Model trained successfully!")

# -----------------------------------------------
# Evaluate on REAL test data
# -----------------------------------------------

y_pred_balanced = rf_balanced.predict(X_test)

precision_b = precision_score(y_test, y_pred_balanced)
recall_b = recall_score(y_test, y_pred_balanced)
f1_b = f1_score(y_test, y_pred_balanced)

print("\n=== RESULTS AFTER SMOTE ===")
print(f"Precision: {precision_b:.4f} ({precision_b*100:.1f}%)")
print(f"Recall:    {recall_b:.4f} ({recall_b*100:.1f}%)")
print(f"F1 Score:  {f1_b:.4f} ({f1_b*100:.1f}%)")

print("\n=== DETAILED REPORT ===")
print(classification_report(y_test, y_pred_balanced,
      target_names=['Normal', 'Fraud']))

# -----------------------------------------------
# New Confusion Matrix
# -----------------------------------------------

cm_balanced = confusion_matrix(y_test, y_pred_balanced)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actually Normal', 'Actually Fraud'])
plt.title('Confusion Matrix After SMOTE — Improved Results', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_smote.png')
plt.show()

print("\n✅ Improved confusion matrix saved!")

# -----------------------------------------------
# Before vs After Comparison
# -----------------------------------------------

print("\n=== BEFORE vs AFTER SMOTE ===")
print(f"{'Metric':<12} {'Before SMOTE':>15} {'After SMOTE':>15}")
print("-" * 44)
print(f"{'Precision':<12} {'27.8%':>15} {f'{precision_b*100:.1f}%':>15}")
print(f"{'Recall':<12} {'27.4%':>15} {f'{recall_b*100:.1f}%':>15}")
print(f"{'F1 Score':<12} {'27.6%':>15} {f'{f1_b*100:.1f}%':>15}")

# -----------------------------------------------
# Day 6 — Explain Predictions with SHAP
# -----------------------------------------------

import shap

print("\n=== DAY 6: SHAP EXPLAINABILITY ===")
print("⏳ Generating SHAP explanations... this may take a minute")

# Create SHAP explainer for our balanced Random Forest model
explainer = shap.TreeExplainer(rf_balanced)

# Calculate SHAP values for a sample of test data
# (using a sample of 500 to keep it fast)
X_test_sample = X_test.sample(n=500, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# For binary classification, shap_values has 2 sets
# We want the "fraud" class explanations (index 1)
if isinstance(shap_values, list):
    fraud_shap_values = shap_values[1]
else:
    fraud_shap_values = shap_values[:, :, 1]

# -----------------------------------------------
# Overall feature importance — which features matter most?
# -----------------------------------------------

plt.figure(figsize=(10, 8))
shap.summary_plot(fraud_shap_values, X_test_sample,
                  show=False)
plt.title("SHAP — Which Features Matter Most for Fraud Detection?",
          fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()

print("✅ SHAP summary plot saved as shap_summary.png")

# -----------------------------------------------
# Explain ONE specific fraud prediction
# -----------------------------------------------

# Find one actual fraud case in our sample
fraud_indices = y_test.loc[X_test_sample.index][y_test.loc[X_test_sample.index] == 1].index

if len(fraud_indices) > 0:
    sample_fraud_idx = fraud_indices[0]
    position = X_test_sample.index.get_loc(sample_fraud_idx)

    print(f"\n=== EXPLAINING ONE FRAUD CASE ===")
    print(f"Transaction details:")
    print(X_test_sample.loc[sample_fraud_idx])

    # Force plot for this single prediction
    plt.figure(figsize=(12, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=fraud_shap_values[position],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test_sample.iloc[position],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False
    )
    plt.title("Why Was THIS Transaction Flagged as Fraud?", fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_single_explanation.png')
    plt.show()

    print("✅ Single prediction explanation saved as shap_single_explanation.png")

print("\n🎉 Project complete with full explainability!")