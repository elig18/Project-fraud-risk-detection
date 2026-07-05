# Project-fraud-risk-detection

# Credit Card Fraud Detection — Risk Scoring & Audit Reporting
 
> Detecting fraudulent transactions on a highly imbalanced dataset, with automated risk scoring and audit-ready reporting. 
---
 
## Overview
 
This project goes beyond a standard fraud classifier. It addresses a real-world challenge in **digital risk & compliance**: how do you detect anomalies in a dataset where 99.83% of transactions are legitimate?
 
The pipeline covers:
- Exploratory data analysis on a severely imbalanced dataset
- Comparison of undersampling (NearMiss) vs oversampling (SMOTE) strategies
- Training and evaluation of multiple classifiers (Logistic Regression, KNN, SVC, Decision Tree)
- Neural network testing on both resampling strategies
- **Risk scoring** — each transaction is assigned a risk level (Low / Medium / High / Critical)
- **Automated audit report** — a CSV log of flagged transactions, ready for compliance review
- **Interactive Streamlit dashboard** — upload a transaction file and get instant fraud predictions with risk levels
---
 
## Project Structure
 
```
fraud-risk-detection/
│
├── notebook.ipynb          # Full analysis: EDA, resampling, classifiers, neural network
├── risk_report.py          # Generates an audit-ready CSV report with risk scoring
├── app.py                  # Streamlit dashboard — interactive fraud detection interface
├── requirements.txt        # Dependencies
└── README.md
```
 
---
 
## Why This Matters for Digital Risk & Compliance
 
In regulated environments, detecting anomalies is only half the challenge. The other half is **explainability and auditability**:
 
- A model that flags every transaction as fraud is useless — **precision matters**
- A model that misses fraud is dangerous — **recall matters even more**
- Compliance teams don't just need predictions — they need **traceable, documented outputs**
This is why this project includes a **risk scoring layer** and an **audit report generator** on top of the ML pipeline — not just a classifier.
 
---
 
## Key Results
 
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 94% | 0.9799 |
| KNN | 93% | 0.9246 |
| Support Vector Classifier | 93% | 0.9747 |
| Decision Tree | 93% | 0.9174 |
 
> ⚠️ Accuracy alone is misleading on imbalanced data. This project evaluates models using **Precision-Recall AUC** and **F1-score**, as recommended for fraud detection tasks.
 
---
 
## Methodology
 
### 1. Exploratory Data Analysis
- Class distribution visualization (0.17% fraud)
- Distribution of `Amount` and `Time` features
- Correlation matrices (imbalanced vs. subsampled)
- Boxplots of most correlated features (V14, V12, V10, V4, V11...)
### 2. Preprocessing
- RobustScaler applied to `Amount` and `Time` (less sensitive to outliers)
- Outlier removal using IQR method on fraud-correlated features
### 3. Resampling Strategies
- **Random UnderSampling** — 492 fraud vs 492 non-fraud
- **SMOTE (OverSampling)** — synthetic minority points generated during cross-validation
### 4. Classification
- 4 classifiers compared with GridSearchCV + StratifiedKFold
- ROC curves and Precision-Recall curves for model selection
- Learning curves to detect overfitting
### 5. Neural Network
- Simple Sequential model (Keras) tested on both resampled datasets
- Confusion matrices compared against 100% accuracy baseline
### 6. Risk Scoring *(added layer)*
Each transaction gets a risk level based on the model's predicted probability:
 
| Probability | Risk Level |
|---|---|
| < 0.3 | 🟢 Low |
| 0.3 – 0.6 | 🟡 Medium |
| 0.6 – 0.85 | 🟠 High |
| > 0.85 | 🔴 Critical |
 
### 7. Audit Report *(added layer)*
`risk_report.py` generates a structured CSV log of all flagged transactions with:
- Transaction index
- Predicted probability
- Risk level
- Recommended action (Monitor / Investigate / Block)
---
 
## Streamlit Dashboard
 
```bash
streamlit run app.py
```
 
Upload a CSV file of transactions → get instant predictions, risk levels, and a downloadable audit report.
 
---
 
## Getting Started
 
```bash
# Clone the repo
git clone https://github.com/elig118/fraud-risk-detection.git
cd fraud-risk-detection
 
# Install dependencies
pip install -r requirements.txt
 
# Run the notebook
jupyter notebook notebook.ipynb
 
# Generate audit report
python risk_report.py
 
# Launch dashboard
streamlit run app.py
```
 
---
 
## Requirements
 
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
tensorflow
keras
streamlit
scipy
```
 
---
 
## What I Learned
 
- Why accuracy is a **misleading metric** on imbalanced datasets and how to choose the right evaluation framework
- The tradeoff between **undersampling** (information loss) and **oversampling** (synthetic data risk)
- Why SMOTE must happen **during** cross-validation, not before — to avoid data leakage
- How to build a **compliance-oriented output layer** on top of a standard ML pipeline
---
 
*Built as part of my personal projects with a focus on practical applications in digital risk management.*

---

**Author:** Elisabeth Gil, Kaggle
**Stack:** Python · Scikit-learn · Imbalanced-learn · Keras · Streamlit  

