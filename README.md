"""
INTERNSHIP PROJECT: Customer Churn Prediction using Data Science & Machine Learning
Author: Vinay Kumar Gupt
Purpose: End-to-end demonstration of data preprocessing, EDA, feature engineering,
model training, evaluation, comparison, and prediction.

The script is self-contained: if customer_churn.csv is not found, it generates a
reproducible synthetic telecom-style dataset and saves it locally.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42
DATA_FILE = "customer_churn.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dataset(n=1500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    tenure = rng.integers(1, 73, n)
    monthly_charges = np.round(rng.normal(70, 25, n).clip(20, 150), 2)
    total_charges = np.round((monthly_charges * tenure * rng.uniform(0.80, 1.10, n)), 2)
    support_calls = rng.poisson(2.0, n).clip(0, 10)
    late_payments = rng.poisson(0.8, n).clip(0, 6)
    contract_months = rng.choice([1, 12, 24], n, p=[0.50, 0.30, 0.20])
    internet_service = rng.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.50, 0.15])
    payment_method = rng.choice(["Electronic", "Bank Transfer", "Card", "Mailed Check"], n)
    senior = (age >= 60).astype(int)
    has_partner = rng.choice(["Yes", "No"], n)
    tech_support = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])

    # Synthetic but realistic churn signal for educational purposes.
    score = (
        -1.7
        - 0.035 * tenure
        + 0.020 * monthly_charges
        + 0.42 * support_calls
        + 0.50 * late_payments
        - 0.85 * (contract_months == 24)
        - 0.45 * (contract_months == 12)
        + 0.55 * (internet_service == "Fiber")
        + 0.30 * senior
        - 0.35 * (tech_support == "Yes")
        + rng.normal(0, 0.65, n)
    )
    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "CustomerID": [f"C{i:05d}" for i in range(1, n + 1)],
        "Age": age,
        "TenureMonths": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportCalls": support_calls,
        "LatePayments": late_payments,
        "ContractMonths": contract_months,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "SeniorCitizen": senior,
        "Partner": has_partner,
        "TechSupport": tech_support,
        "Churn": churn
    })
    # Add a small amount of missingness to demonstrate preprocessing.
    for col in ["MonthlyCharges", "TotalCharges", "TechSupport"]:
        idx = rng.choice(n, size=max(1, int(0.02*n)), replace=False)
        df.loc[idx, col] = np.nan
    return df


def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded existing dataset: {DATA_FILE}")
    else:
        df = generate_dataset()
        df.to_csv(DATA_FILE, index=False)
        print(f"Generated educational dataset: {DATA_FILE}")
    return df


def basic_eda(df):
    print("\n--- Dataset shape ---")
    print(df.shape)
    print("\n--- Missing values ---")
    print(df.isna().sum())
    print("\n--- Class distribution ---")
    print(df["Churn"].value_counts(normalize=True).rename("proportion"))

    # Churn distribution
    ax = df["Churn"].value_counts().sort_index().plot(kind="bar")
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn (0=No, 1=Yes)")
    ax.set_ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/churn_distribution.png", dpi=160)
    plt.close()

    # Monthly charges by churn
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.suptitle("")
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_charges_by_churn.png", dpi=160)
    plt.close()


def prepare_data(df):
    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", __import__("sklearn").preprocessing.OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
    ])

    return X, y, preprocessor


def train_and_evaluate(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, proba)
        }
        results.append(row)
        fitted[name] = (pipe, pred, proba)

        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred))

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    # ROC curves
    plt.figure(figsize=(7, 5))
    for name, (pipe, pred, proba) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name)
    plt.title("ROC Curves - Model Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=160)
    plt.close()

    best_name = results_df.iloc[0]["Model"]
    best_pipe, best_pred, best_proba = fitted[best_name]
    cm = confusion_matrix(y_test, best_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/best_confusion_matrix.png", dpi=160)
    plt.close()

    return results_df, best_name, best_pipe


def predict_customer(model, customer):
    new_df = pd.DataFrame([customer])
    probability = model.predict_proba(new_df)[0, 1]
    prediction = int(probability >= 0.50)
    return prediction, probability


def main():
    df = load_data()
    basic_eda(df)

    X, y, preprocessor = prepare_data(df)
    results, best_name, best_model = train_and_evaluate(X, y, preprocessor)

    print("\n--- Model comparison ---")
    print(results.to_string(index=False))
    print(f"\nRecommended model by ROC-AUC: {best_name}")

    sample_customer = {
        "Age": 35,
        "TenureMonths": 8,
        "MonthlyCharges": 105.0,
        "TotalCharges": 840.0,
        "SupportCalls": 4,
        "LatePayments": 2,
        "ContractMonths": 1,
        "InternetService": "Fiber",
        "PaymentMethod": "Electronic",
        "SeniorCitizen": 0,
        "Partner": "No",
        "TechSupport": "No"
    }
    pred, prob = predict_customer(best_model, sample_customer)
    print("\n--- Example prediction ---")
    print(f"Churn prediction: {'Yes' if pred else 'No'}")
    print(f"Churn probability: {prob:.2%}")

    print("\nProject completed. Check the 'outputs' folder for charts and metrics.")


if __name__ == "__main__":
    main()

