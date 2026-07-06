# imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from imblearn.over_sampling import SMOTE
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# 1. pre process, loading of data

def load_and_preprocess(filepath="creditcard.csv"):
    """
    Load the dataset and apply the same preprocessing as in the notebook:
    - RobustScaler on Amount and Time
    - Drop original Amount and Time columns
    """
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)

    # Scale Amount and Time with RobustScaler (less sensitive to outliers)
    rob_scaler = RobustScaler()
    df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    # Reorder columns
    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]
    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

    print(f"[INFO] Dataset loaded — {len(df)} transactions ({df['Class'].sum()} fraud cases)")
    return df


# 2. model training

def train_model(df):
    """
    Train a Logistic Regression model using SMOTE oversampling.
    Returns the trained model, test features, and test labels.
    """
    print("[INFO] Training model with SMOTE oversampling...")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train/test split : stratified to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training data (never on test set, avoid data leakage)
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    print(f"[INFO] After SMOTE — Training set: {len(X_train_sm)} samples")

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sm, y_train_sm)

    print("[INFO] Model trained successfully.")
    return model, X_test, y_test


# 3. risk scoring

def assign_risk_level(probability):
    """
    Assign a risk level based on the predicted fraud probability.

    Thresholds:
        < 0.30    → Low      (no immediate action needed)
        0.30–0.60 → Medium   (monitor the transaction)
        0.60–0.85 → High     (investigate)
        > 0.85    → Critical (block immediately)
    """
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    elif probability < 0.85:
        return "High"
    else:
        return "Critical"


def assign_action(risk_level):
    """
    Map each risk level to a recommended compliance action.
    """
    actions = {
        "Low": "No action required",
        "Medium": "Monitor",
        "High": "Investigate",
        "Critical": "Block immediately"
    }
    return actions[risk_level]


# 4. audit report generated

def generate_audit_report(model, X_test, y_test, threshold=0.30, output_file="audit_report.csv"):
    """
    Generate a compliance-ready audit report of flagged transactions.

    Only transactions with predicted fraud probability >= threshold are included.
    The report contains:
        - Transaction index
        - Predicted fraud probability
        - Risk level (Low / Medium / High / Critical)
        - Recommended action
        - True label (for validation purposes)

    Args:
        model       : trained classifier with predict_proba support
        X_test      : test features (DataFrame)
        y_test      : true labels (Series)
        threshold   : minimum probability to flag a transaction (default: 0.30)
        output_file : name of the output CSV file
    """
    print(f"\n[INFO] Generating audit report (threshold = {threshold})...")

    # Get predicted probabilities for the fraud class (class 1)
    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Build results DataFrame
    results = pd.DataFrame({
        "transaction_index": X_test.index,
        "fraud_probability": np.round(probabilities, 4),
        "true_label": y_test.values
    })

    # Apply risk scoring
    results["risk_level"] = results["fraud_probability"].apply(assign_risk_level)
    results["recommended_action"] = results["risk_level"].apply(assign_action)

    # Filter : only flag transactions above threshold
    flagged = results[results["fraud_probability"] >= threshold].copy()
    flagged = flagged.sort_values("fraud_probability", ascending=False).reset_index(drop=True)

    # Add report metadata
    flagged["report_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Export to CSV
    flagged.to_csv(output_file, index=False)

    # Compute fraud-specific metrics (most important for compliance)
    fraud_recall = recall_score(y_test, y_pred)
    fraud_precision = precision_score(y_test, y_pred)

    # Summary in console
    print(f"\n{'='*55}")
    print(f"  AUDIT REPORT SUMMARY")
    print(f"{'='*55}")
    print(f"  Total transactions scored  : {len(results)}")
    print(f"  Transactions flagged       : {len(flagged)}")
    print(f"  Threshold applied          : {threshold}")
    print(f"\n  Model performance on fraud:")
    print(f"    Recall (fraud detected)  : {round(fraud_recall * 100, 2)}%")
    print(f"    Precision                : {round(fraud_precision * 100, 2)}%")
    print(f"\n  Risk level breakdown:")
    for level in ["Critical", "High", "Medium", "Low"]:
        count = len(flagged[flagged["risk_level"] == level])
        print(f"    🔴 {level:<10} : {count}" if level == "Critical"
              else f"    🟠 {level:<10} : {count}" if level == "High"
              else f"    🟡 {level:<10} : {count}" if level == "Medium"
              else f"    🟢 {level:<10} : {count}")
    print(f"\n  Report saved to: {output_file}")
    print(f"{'='*55}\n")

    return flagged


# 5. Main

if __name__ == "__main__":

    # --- Load and preprocess ---
    df = load_and_preprocess("creditcard.csv")

    # --- Train model ---
    model, X_test, y_test = train_model(df)

    # --- Generate audit report ---
    report = generate_audit_report(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=0.30,
        output_file="audit_report.csv"
    )

    # --- Preview top flagged transactions ---
    print("[INFO] Top 10 highest-risk transactions:\n")
    print(report[["transaction_index", "fraud_probability", "risk_level", "recommended_action"]].head(10).to_string(index=False))