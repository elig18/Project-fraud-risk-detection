# imports
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# page configuration
st.set_page_config(
    page_title="Fraud Risk Detection",
    page_icon="🔍",
    layout="wide"
)


# 1. Help functions

def assign_risk_level(probability):
    """Assign a risk level based on predicted fraud probability."""
    if probability < 0.30:
        return "🟢 Low"
    elif probability < 0.60:
        return "🟡 Medium"
    elif probability < 0.85:
        return "🟠 High"
    else:
        return "🔴 Critical"


def assign_action(risk_level):
    """Map each risk level to a recommended compliance action."""
    actions = {
        "🟢 Low": "No action required",
        "🟡 Medium": "Monitor",
        "🟠 High": "Investigate",
        "🔴 Critical": "Block immediately"
    }
    return actions.get(risk_level, "Unknown")


def preprocess(df):
    """
    Apply the same preprocessing as in the notebook:
    - RobustScaler on Amount and Time
    """
    df = df.copy()
    rob_scaler = RobustScaler()
    df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]
    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)
    return df


@st.cache_resource(show_spinner=False)
def train_model(df):
    """
    Train a Logistic Regression model using SMOTE.
    Cached so it only runs once per session.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sm, y_train_sm)

    return model, X_test, y_test


def score_transactions(model, X):
    """Score a set of transactions and return a results DataFrame."""
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "transaction_index": X.index,
        "fraud_probability": np.round(probabilities, 4),
    })
    results["risk_level"] = results["fraud_probability"].apply(assign_risk_level)
    results["recommended_action"] = results["risk_level"].apply(assign_action)
    results["report_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return results.sort_values("fraud_probability", ascending=False).reset_index(drop=True)


# 2. sidebar

with st.sidebar:
    st.title("Fraud Risk Detection")
    st.markdown("---")
    st.markdown(
        "This dashboard detects fraudulent credit card transactions "
        "using a Logistic Regression model trained with SMOTE oversampling."
    )
    st.markdown("---")

    st.subheader("Settings")
    threshold = st.slider(
        "Risk threshold (flag transactions above)",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Only transactions with fraud probability above this threshold will be flagged."
    )

    st.markdown("---")
    st.markdown(
        "**Risk levels:**\n"
        "- 🟢 Low < 0.30\n"
        "- 🟡 Medium 0.30 to 0.60\n"
        "- 🟠 High 0.60 to 0.85\n"
        "- 🔴 Critical > 0.85"
    )
    st.markdown("---")
    st.caption("Built by Elisabeth Gil · [@elig18](https://github.com/elig18)")


# 3. main file upload and model training

st.title("Credit Card Fraud Detection")
st.markdown("**Risk Scoring & Audit Reporting Dashboard**")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your transaction CSV file (creditcard.csv format)",
    type=["csv"]
)

if uploaded_file is not None:

    # --- Load data ---
    with st.spinner("Loading and preprocessing data..."):
        raw_df = pd.read_csv(uploaded_file)
        df = preprocess(raw_df.copy())

    st.success(f"Dataset loaded — **{len(df):,} transactions** | **{int(df['Class'].sum())} fraud cases** ({round(df['Class'].mean()*100, 2)}%)")

    # --- Train model (with detailed spinner) ---
    with st.spinner("Training model with SMOTE — this takes 1-2 min on first load, cached after that..."):
        model, X_test, y_test = train_model(df)

    st.success("Model trained successfully.")
    st.markdown("---")


    # 4. tabs

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset Overview",
        "🤖 Model Performance",
        "🚨 Flagged Transactions",
        "📋 Audit Report"
    ])


    # Tab 1 : dataset overview

    with tab1:
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Cases", f"{int(df['Class'].sum())}")
        col3.metric("Fraud Rate", f"{round(df['Class'].mean()*100, 3)}%")

        st.markdown("#### Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")
        colors = ["#7C3AED", "#06B6D4"]
        df["Class"].value_counts().plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Class Distribution (0: No Fraud | 1: Fraud)", color="#F1F5F9")
        ax.set_xlabel("Class", color="#F1F5F9")
        ax.set_ylabel("Count", color="#F1F5F9")
        ax.tick_params(colors="#F1F5F9")
        ax.set_xticklabels(["No Fraud", "Fraud"], rotation=0, color="#F1F5F9")
        st.pyplot(fig)

        st.markdown("#### Transaction Amount Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor("#0F172A")
        ax2.set_facecolor("#1E293B")
        ax2.hist(raw_df["Amount"], bins=100, color="#7C3AED", alpha=0.8)
        ax2.set_title("Distribution of Transaction Amount", color="#F1F5F9")
        ax2.set_xlabel("Amount (€)", color="#F1F5F9")
        ax2.set_ylabel("Count", color="#F1F5F9")
        ax2.tick_params(colors="#F1F5F9")
        st.pyplot(fig2)


    # Tab 2 : model performance

    with tab2:
        st.subheader("Model Performance on Test Set")

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{round(report['accuracy']*100, 2)}%")
        col2.metric("Fraud Recall", f"{round(report['1']['recall']*100, 2)}%")
        col3.metric("Fraud Precision", f"{round(report['1']['precision']*100, 2)}%")

        st.markdown("#### Classification Report")
        st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor("#0F172A")
        ax3.set_facecolor("#1E293B")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="BuPu", ax=ax3,
            xticklabels=["No Fraud", "Fraud"],
            yticklabels=["No Fraud", "Fraud"]
        )
        ax3.set_title("Confusion Matrix", color="#F1F5F9")
        ax3.set_ylabel("True Label", color="#F1F5F9")
        ax3.set_xlabel("Predicted Label", color="#F1F5F9")
        ax3.tick_params(colors="#F1F5F9")
        st.pyplot(fig3)


    # Tab 3 : flagged transactions

    with tab3:
        st.subheader("Flagged Transactions")
        st.markdown(f"Showing transactions with fraud probability **≥ {threshold}** (adjust in sidebar)")

        # Score all test transactions
        scored = score_transactions(model, X_test)
        flagged = scored[scored["fraud_probability"] >= threshold].copy()

        # Risk level counts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Critical", len(flagged[flagged["risk_level"] == "🔴 Critical"]))
        col2.metric("🟠 High", len(flagged[flagged["risk_level"] == "🟠 High"]))
        col3.metric("🟡 Medium", len(flagged[flagged["risk_level"] == "🟡 Medium"]))
        col4.metric("Total Flagged", len(flagged))

        st.markdown("#### Top Flagged Transactions")
        st.dataframe(
            flagged[["transaction_index", "fraud_probability", "risk_level", "recommended_action"]].head(50),
            use_container_width=True
        )

        st.markdown("#### Fraud Probability Distribution (flagged only)")
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        fig4.patch.set_facecolor("#0F172A")
        ax4.set_facecolor("#1E293B")
        ax4.hist(flagged["fraud_probability"], bins=30, color="#06B6D4", alpha=0.8)
        ax4.set_title("Distribution of Fraud Probability — Flagged Transactions", color="#F1F5F9")
        ax4.set_xlabel("Fraud Probability", color="#F1F5F9")
        ax4.set_ylabel("Count", color="#F1F5F9")
        ax4.tick_params(colors="#F1F5F9")
        st.pyplot(fig4)


    # Tab 4 : report

    with tab4:
        st.subheader("Audit Report")
        st.markdown(
            "This report contains all flagged transactions with their risk level "
            "and recommended compliance action. Download it as CSV for audit purposes."
        )

        scored_full = score_transactions(model, X_test)
        flagged_full = scored_full[scored_full["fraud_probability"] >= threshold].copy()

        st.dataframe(flagged_full, use_container_width=True)

        # Download button
        csv = flagged_full.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Audit Report (CSV)",
            data=csv,
            file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        st.info(
            f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            f"{len(flagged_full)} transactions flagged · threshold = {threshold}"
        )

else:
    # --- Empty state ---
    st.info("Upload a CSV file to get started. Use the **creditcard.csv** dataset from Kaggle.")
    st.markdown(
        """
        **Expected format:** the file should follow the structure of the 
        [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 
        columns `Time`, `V1`–`V28`, `Amount`, and `Class`.
        """
    )