import json
import os
import sys

import joblib
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

N_TRIALS   = 30
N_CV_FOLDS = 5
BEST_PARAMS_PATH = os.path.join("models", "best_params.json")
TUNED_MODEL_PATH = os.path.join("models", "tuned_model.pkl")


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found")
    y = (df["Churn"] == "Yes").astype(int).rename("Churn")
    X = df.select_dtypes(include="number").drop(columns=["Churn"], errors="ignore")
    return X, y


def _make_objective(X_train, y_train, scaler: StandardScaler):
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        C            = trial.suggest_float("C", 1e-3, 100, log=True)
        penalty      = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver       = trial.suggest_categorical("solver", ["liblinear", "saga"])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "none"])

        # liblinear supports both l1/l2; saga supports l1/l2 — all combos valid here
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=None if class_weight == "none" else class_weight,
            max_iter=1000,
            random_state=42,
        )

        scores = cross_val_score(
            model, scaler.transform(X_train), y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        auc = scores.mean()

        trial.set_user_attr("cv_std", scores.std())
        print(
            f"  Trial {trial.number:>2} | "
            f"C={C:.4f}  penalty={penalty}  solver={solver}  "
            f"class_weight={class_weight:<10} | "
            f"AUC={auc:.4f} (+/-{scores.std():.4f})"
        )
        return auc

    return objective


def tune(X_train, y_train, scaler: StandardScaler) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    print(f"Running Optuna — {N_TRIALS} trials, {N_CV_FOLDS}-fold CV (scoring=roc_auc)\n")
    study.optimize(_make_objective(X_train, y_train, scaler), n_trials=N_TRIALS)

    best = study.best_params.copy()
    # restore None from the "none" string used in the categorical space
    if best.get("class_weight") == "none":
        best["class_weight"] = None

    print(f"\nBest trial  : #{study.best_trial.number}")
    print(f"Best CV AUC : {study.best_value:.4f}")
    print(f"Best params : {best}")
    return best


def train_final(
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler: StandardScaler,
) -> LogisticRegression:
    model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    return model


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


def print_metrics(metrics: dict[str, float], best_params: dict) -> None:
    sep = "-" * 42
    print(f"\n{sep}")
    print(f"  Tuned Logistic Regression - Test Set")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(sep)
    print(f"  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k:<16}: {v}")
    print(f"{sep}\n")


def run(data_dir: str = "data") -> dict[str, float]:
    src = os.path.join(data_dir, "features.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found — run run_features.py first")

    X, y = load_data(src)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    best_params = tune(X_train, y_train, scaler)

    os.makedirs("models", exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params saved to {BEST_PARAMS_PATH}")

    model = train_final(best_params, X_train, y_train, scaler)
    metrics = evaluate(model, scaler, X_test, y_test)

    joblib.dump({"model": model, "scaler": scaler}, TUNED_MODEL_PATH)
    print(f"Tuned model saved to {TUNED_MODEL_PATH}")

    print_metrics(metrics, best_params)
    return metrics


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run(data_dir)
