import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODEL_PATH = os.path.join("models", "baseline.pkl")


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Encode target before dropping non-numerics
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset")
    y = (df["Churn"] == "Yes").astype(int).rename("Churn")

    # Keep only numeric columns (drops customerID, Contract, etc.)
    df = df.select_dtypes(include="number")

    # Remove target from features if it was already numeric
    df = df.drop(columns=["Churn"], errors="ignore")

    return df, y


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    y_prob   = model.predict_proba(X_scaled)[:, 1]

    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred),
        "auc_roc"  : roc_auc_score(y_test, y_prob),
    }


def save_model(model: LogisticRegression, scaler: StandardScaler) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)


def print_metrics(metrics: dict[str, float], n_train: int, n_test: int) -> None:
    sep = "-" * 38
    print(f"\n{sep}")
    print(f"  Baseline - Logistic Regression")
    print(sep)
    print(f"  Train / Test split : {n_train} / {n_test}")
    print(sep)
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  F1 Score           : {metrics['f1']:.4f}")
    print(f"  AUC-ROC            : {metrics['auc_roc']:.4f}")
    print(f"{sep}\n")


def run(data_dir: str = "data") -> dict[str, float]:
    src = os.path.join(data_dir, "features.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found — run run_features.py first")

    X, y = load_data(src)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, scaler = train(X_train, y_train)
    metrics = evaluate(model, scaler, X_test, y_test)

    save_model(model, scaler)
    print(f"Model saved to {MODEL_PATH}")

    print_metrics(metrics, len(X_train), len(X_test))
    return metrics


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run(data_dir)
