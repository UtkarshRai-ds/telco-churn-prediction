import os
import sys

import joblib
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODEL_PATH = os.path.join("models", "production_model.pkl")
DATA_PATH  = os.path.join("data", "features.csv")
OUTPUT_PATH = os.path.join("data", "predictions.csv")


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def load_data(path: str = DATA_PATH) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found at {path}")
    df = pd.read_csv(path)

    customer_ids = df["customerID"]
    y = (df["Churn"] == "Yes").astype(int) if "Churn" in df.columns else None

    X = df.select_dtypes(include="number").drop(columns=["Churn"], errors="ignore")
    return customer_ids, X, y


def predict(bundle: dict, X: pd.DataFrame) -> tuple[list[int], list[float]]:
    model         = bundle["model"]
    scaler        = bundle["scaler"]
    feature_names = bundle.get("feature_names")

    if feature_names is not None:
        X = X[feature_names]

    X_scaled     = scaler.transform(X)
    predictions  = model.predict(X_scaled).tolist()
    probabilities = model.predict_proba(X_scaled)[:, 1].tolist()
    return predictions, probabilities


def save_predictions(
    customer_ids: pd.Series,
    actual: pd.Series | None,
    predictions: list[int],
    probabilities: list[float],
    path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    results = pd.DataFrame({
        "customerID":        customer_ids.values,
        "actual_churn":      actual.values if actual is not None else None,
        "predicted_churn":   predictions,
        "churn_probability": probabilities,
    })
    results.to_csv(path, index=False)
    return results


if __name__ == "__main__":
    bundle = load_model()
    customer_ids, X, y = load_data()

    predictions, probabilities = predict(bundle, X)
    results = save_predictions(customer_ids, y, predictions, probabilities)

    n_churners = sum(predictions)
    print(f"Predictions saved to {OUTPUT_PATH}")
    print(f"Total customers  : {len(results)}")
    print(f"Predicted churners: {n_churners} ({n_churners / len(results):.1%})")
