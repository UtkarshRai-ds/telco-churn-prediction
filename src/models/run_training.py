import json
import os
import sys
import tempfile

import joblib
import mlflow
import mlflow.sklearn
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

EXPERIMENT_NAME  = "telco-churn"
PRODUCTION_PATH  = os.path.join("models", "production_model.pkl")
BEST_PARAMS_PATH = os.path.join("models", "best_params.json")

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

def _load_best_params() -> dict:
    if not os.path.exists(BEST_PARAMS_PATH):
        raise FileNotFoundError(
            f"{BEST_PARAMS_PATH} not found — run tuning.py first"
        )
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


def build_model_configs() -> list[dict]:
    best = _load_best_params()
    return [
        {
            "name": "baseline",
            "params": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        {
            "name": "tuned_best",
            "params": {**best, "max_iter": 1000, "random_state": 42},
        },
    ]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found")
    y = (df["Churn"] == "Yes").astype(int).rename("Churn")
    X = df.select_dtypes(include="number").drop(columns=["Churn"], errors="ignore")
    return X, y


def compute_metrics(model, scaler, X, y) -> dict[str, float]:
    Xs     = scaler.transform(X)
    y_pred = model.predict(Xs)
    y_prob = model.predict_proba(Xs)[:, 1]
    return {
        "accuracy" : accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall"   : recall_score(y, y_pred),
        "f1"       : f1_score(y, y_pred),
        "auc_roc"  : roc_auc_score(y, y_prob),
    }


# ---------------------------------------------------------------------------
# MLflow training loop
# ---------------------------------------------------------------------------

def train_and_log(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
) -> tuple[LogisticRegression, dict]:

    with mlflow.start_run(run_name=config["name"]):
        params = config["params"]
        model  = LogisticRegression(**params)
        model.fit(scaler.transform(X_train), y_train)

        train_metrics = compute_metrics(model, scaler, X_train, y_train)
        test_metrics  = compute_metrics(model, scaler, X_test,  y_test)

        # Log hyperparameters
        mlflow.log_param("model_name", config["name"])
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log train metrics with prefix
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        # Log test metrics with prefix
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Log model artifact via joblib (scaler bundled)
        bundle = {"model": model, "scaler": scaler, "feature_names": list(X_train.columns)}
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, f"{config['name']}.pkl")
            joblib.dump(bundle, artifact_path)
            mlflow.log_artifact(artifact_path, artifact_path="models")

        run_id = mlflow.active_run().info.run_id
        print(f"  [{config['name']}]  run_id={run_id[:8]}...")
        print(f"    train AUC={train_metrics['auc_roc']:.4f}  "
              f"test AUC={test_metrics['auc_roc']:.4f}  "
              f"F1={test_metrics['f1']:.4f}")

    return model, test_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(data_dir: str = "data") -> None:
    src = os.path.join(data_dir, "features.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found — run run_features.py first")

    X, y = load_data(src)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)

    mlflow.set_tracking_uri("file:///" + _ROOT.replace("\\", "/") + "/mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    configs      = build_model_configs()
    best_model   = None
    best_metrics = None
    best_auc     = -1.0

    print(f"MLflow experiment: '{EXPERIMENT_NAME}'")
    print(f"Tracking URI      : {mlflow.get_tracking_uri()}\n")
    print(f"Training {len(configs)} model(s)...")

    for config in configs:
        model, metrics = train_and_log(
            config, X_train, y_train, X_test, y_test, scaler
        )
        if metrics["auc_roc"] > best_auc:
            best_auc     = metrics["auc_roc"]
            best_model   = model
            best_metrics = metrics
            best_name    = config["name"]

    # Save best model as production artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": best_model, "scaler": scaler, "feature_names": list(X_train.columns)},
        PRODUCTION_PATH,
    )

    sep = "-" * 44
    print(f"\n{sep}")
    print(f"  Production model  : {best_name}")
    print(f"  Saved to          : {PRODUCTION_PATH}")
    print(sep)
    print(f"  Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f}")
    print(f"  F1        : {best_metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {best_metrics['auc_roc']:.4f}")
    print(f"{sep}\n")
    print(f"View runs: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run(data_dir)
