# Applied Machine Learning for Financial Systems: Prediction, Anomaly Detection, and Explainability

**A Research Summary**

**Muhammad Faisal**
CS Graduate, Hazara University Mansehra | June 2026
[GitHub](https://github.com/Muhammadfaisal39) · [LinkedIn](https://www.linkedin.com/in/muhammadfaisal39)

---

## Abstract

Financial systems generate vast amounts of transactional and market data, creating both opportunities and challenges for machine learning applications. This summary presents two applied research projects exploring machine learning in financial contexts: (1) a stock price prediction system comparing Linear Regression and Random Forest models, and (2) a credit card fraud detection system addressing the well-documented challenge of class imbalance through SMOTE-based resampling. Both projects incorporate SHAP (SHapley Additive exPlanations) to address the interpretability gap identified in current financial machine learning literature. The fraud detection model improved from an F1 score of 27.6% to 82.1% after balancing the training data, and feature importance analysis independently confirmed indicators (V14, V12, V4) consistent with prior published research on the same dataset. These findings motivate continued research into explainable, deployable machine learning systems for real-world financial decision-making.

---

## 1. Problem Statement

Machine learning has demonstrated strong potential in financial applications, including stock price forecasting, market analysis, and fraud detection. However, two persistent challenges limit real-world adoption.

First, financial datasets such as fraud and risk data are typically highly imbalanced, with the events of interest (fraud, default, anomalies) representing a tiny fraction of total observations. Standard models trained on such data tend to perform poorly on the minority class despite achieving misleadingly high overall accuracy.

Second, many high-performing models, particularly ensemble methods such as Random Forest, function as "black boxes," offering limited insight into why a particular prediction was made. In regulated financial environments, this lack of transparency restricts practical deployment, as analysts, regulators, and customers require justifiable, interpretable decisions.

This research summary investigates both challenges directly: building predictive and detection models on real financial data, identifying where they fail, applying established techniques to improve performance, and using explainability methods to make model decisions transparent.

---

## 2. Related Work

This work builds on two strands of existing literature.

The first concerns the application of machine learning to financial forecasting broadly. Falaiye et al. (2024) reviewed the advancements, challenges, and implications of AI-driven predictions in U.S. financial markets, highlighting that ensemble and deep learning methods consistently outperform traditional statistical approaches in stock price and market behaviour prediction, while also noting persistent challenges around model robustness and generalizability across market conditions.

The second strand concerns explainability in financial machine learning. A systematic review of explainable AI (XAI) techniques in finance identified SHAP, LIME, Counterfactual Explanations, and Partial Dependence Plots (PDPs) as the most widely applied methods across financial use cases including credit scoring, fraud detection, risk assessment, and portfolio management. The review emphasized that interpretability is not a secondary concern but a prerequisite for real-world adoption of ML systems in regulated financial environments. This finding directly informed the decision to incorporate SHAP into both projects in this summary, rather than treating model performance as the sole measure of success.

---

## 3. Methodology

### 3.1 Project 1 — Financial Data Predictor

Using historical S&P 500 data (619,040 records across 500 companies, 2013–2018), Apple Inc. (AAPL) closing prices were isolated and used to train two regression models: Linear Regression as a baseline, and Random Forest Regression as a higher-capacity alternative. Data was split 80/20 into training and test sets. Model performance was evaluated using R² score and Mean Absolute Error (MAE).

**Repository:** [financial-data-predictor](https://github.com/Muhammadfaisal39/financial-data-predictor)

### 3.2 Project 2 — Financial Fraud Detector

Using the publicly available credit card fraud dataset (284,807 transactions, 492 confirmed fraud cases, 0.17% positive class), two modelling approaches were compared. The first used Isolation Forest, an unsupervised anomaly detection algorithm, applied without access to ground-truth fraud labels during training. The second addressed the severe class imbalance directly: the training set was rebalanced using SMOTE (Synthetic Minority Oversampling Technique), generating synthetic minority-class examples, after which a Random Forest classifier was trained in a supervised setting. Crucially, SMOTE was applied only to the training split; the test set retained its original, real-world class distribution to ensure evaluation validity.

**Repository:** [financial-fraud-detector](https://github.com/Muhammadfaisal39/financial-fraud-detector)

### 3.3 Explainability

For both projects, SHAP (TreeExplainer) was applied post-hoc to the trained models. SHAP decomposes each individual prediction into additive contributions from each input feature, enabling both global feature importance ranking and local, per-prediction explanations.

---

## 4. Results

### 4.1 Stock Price Prediction

| Model | R² Score | Mean Absolute Error |
|---|---|---|
| Linear Regression | 0.7664 (76.6%) | $11.67 |
| **Random Forest** | **0.9984 (99.8%)** | **$0.89** |

Random Forest substantially outperformed Linear Regression, reducing average prediction error from $11.67 to $0.89 and improving R² from 0.766 to 0.998. This confirms that Apple's price trajectory over the study period, while broadly directional, contains non-linear patterns that ensemble tree-based methods capture more effectively than a single linear function.

### 4.2 Fraud Detection

| Metric | Isolation Forest (Unsupervised) | Random Forest + SMOTE (Supervised, Balanced) |
|---|---|---|
| Precision | 27.8% | **82.5%** |
| Recall | 27.4% | **81.6%** |
| F1 Score | 27.6% | **82.1%** |

The unsupervised Isolation Forest model, trained without fraud labels, correctly identified 135 of 492 fraud cases (27.4% recall). While this demonstrates that anomaly-based detection carries some genuine signal, performance was insufficient for practical deployment. Addressing the class imbalance through SMOTE and moving to a supervised Random Forest classifier improved every metric by approximately 55 percentage points, demonstrating that combining synthetic balancing with supervised learning is substantially more effective than unsupervised anomaly detection alone on this severely imbalanced dataset (0.17% positive class).

### 4.3 Explainability Findings

SHAP analysis of the fraud detection model identified V14, V12, and V4 as the three strongest predictors of fraudulent transactions, with low values of V14 and V12 and high values of V4 most strongly associated with fraud. These findings independently align with feature importance patterns reported in prior published work using the same dataset, providing external validation that the model is learning genuine underlying fraud signals rather than dataset-specific noise.

**Figure 1.** SHAP summary plot showing global feature importance for fraud prediction (see [financial-fraud-detector](https://github.com/Muhammadfaisal39/financial-fraud-detector) repository for the full chart). Red indicates high feature values; blue indicates low values. Position relative to zero indicates direction of impact on the model's fraud prediction.

---

## 5. Discussion

Three findings stand out across both projects.

First, ensemble tree-based methods (Random Forest) consistently outperformed simpler baselines in both regression and classification settings, consistent with broader literature on financial ML.

Second, class imbalance is not a minor technical detail but a central determinant of model usefulness in fraud detection; a model with 99.8% raw accuracy caught barely a quarter of actual fraud cases, illustrating why accuracy alone is an inadequate evaluation metric for imbalanced problems.

Third, explainability methods such as SHAP are not merely diagnostic add-ons — in this work, they served as a form of independent validation, confirming that model behaviour aligned with patterns already established in the literature, which increases confidence that the models learned meaningful structure rather than overfitting to noise.

---

## 6. Future Work

- Extend stock prediction beyond a single equity to a portfolio of stocks, and incorporate time-series-specific architectures (e.g., LSTM) to capture temporal dependencies that tree-based models do not explicitly model.
- Evaluate additional resampling strategies (e.g., ADASYN, cost-sensitive learning) against SMOTE to assess robustness of the fraud detection improvement.
- Apply LIME and Counterfactual Explanations alongside SHAP to compare interpretability outputs and assess consistency across explanation methods, as recommended in the XAI literature reviewed in Section 2.
- Investigate deployment-oriented constraints, including inference latency and model monitoring for concept drift in live financial data streams.

---

## References

1. Falaiye, T., et al. (2024). *Machine learning in financial forecasting: A US review — Exploring the advancements, challenges, and implications of AI-driven predictions in financial markets.* ResearchGate.
2. Systematic review of Explainable AI (XAI) techniques in financial applications, identifying SHAP, LIME, Counterfactual Explanations, and Partial Dependence Plots as predominant methods across credit scoring, fraud detection, risk assessment, and portfolio management use cases.
3. Project repositories: [financial-data-predictor](https://github.com/Muhammadfaisal39/financial-data-predictor) · [financial-fraud-detector](https://github.com/Muhammadfaisal39/financial-fraud-detector)

---

*This research summary was also presented in part as preliminary work at the HEC National Conference, "Emerging Trends in Natural Sciences," Pakistan, 2023.*
