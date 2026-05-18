import os
import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Model definitions
# Why each model fits churn prediction:
#
# LogisticRegression — interpretable linear baseline; coefficients map directly
#   to feature importance for marketing stakeholders; fast to train; works well
#   with class_weight='balanced' on imbalanced churn labels.
#
# XGBClassifier — gradient boosting captures non-linear interactions (e.g.
#   senior + fiber optic risk); scale_pos_weight compensates for ~26% churn
#   rate; strong out-of-the-box AUC on tabular data.
#
# RandomForestClassifier — ensemble of trees reduces variance; robust to
#   correlated features and outliers; feature_importances_ provides intuitive
#   explanation for business teams; class_weight='balanced' handles imbalance.
# ---------------------------------------------------------------------------

MODELS = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "XGBClassifier": XGBClassifier(
        scale_pos_weight=3, n_estimators=100, random_state=42,
        eval_metric="logloss", verbosity=0,
    ),
    "RandomForest": RandomForestClassifier(
        class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1
    ),
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found")
    y = (df["Churn"] == "Yes").astype(int).rename("Churn")
    X = df.select_dtypes(include="number").drop(columns=["Churn"], errors="ignore")
    return X, y


def run(data_dir: str = "data") -> pd.DataFrame:
    src = os.path.join(data_dir, "features.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found — run run_features.py first")

    X, y = load_data(src)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)

    results = []

    for name, model in MODELS.items():
        print(f"Training {name}...", end=" ", flush=True)
        t0 = time.time()

        cv_scores = cross_val_score(model, X_train_s, y_train, cv=CV, scoring="roc_auc")
        model.fit(X_train_s, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        results.append({
            "model"        : name,
            "cv_auc_mean"  : cv_scores.mean(),
            "cv_auc_std"   : cv_scores.std(),
            "test_auc"     : roc_auc_score(y_test, y_prob),
            "test_f1"      : f1_score(y_test, y_pred),
            "train_time_s" : elapsed,
        })

        model_path = os.path.join("models", f"{name.lower()}.pkl")
        joblib.dump({"model": model, "scaler": scaler}, model_path)
        print(f"done ({elapsed:.1f}s)  saved to {model_path}")

    comparison = pd.DataFrame(results).set_index("model")
    comparison = comparison.sort_values("test_auc", ascending=False)

    _print_comparison(comparison)
    _print_recommendation(comparison)

    return comparison


def _print_comparison(df: pd.DataFrame) -> None:
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  Model Comparison")
    print(sep)
    header = f"  {'Model':<22} {'CV AUC':>9} {'CV Std':>8} {'Test AUC':>10} {'F1':>8} {'Time(s)':>9}"
    print(header)
    print(sep)
    for name, row in df.iterrows():
        print(
            f"  {name:<22}"
            f"  {row['cv_auc_mean']:.4f}"
            f"  {row['cv_auc_std']:.4f}"
            f"  {row['test_auc']:.4f}"
            f"  {row['test_f1']:.4f}"
            f"  {row['train_time_s']:>7.1f}s"
        )
    print(sep)


def _print_recommendation(df: pd.DataFrame) -> None:
    best = df.index[0]
    best_row = df.iloc[0]

    runners_up = df.iloc[1:]
    auc_gap = best_row["test_auc"] - runners_up["test_auc"].max()

    print(f"\n  Best model: {best}")
    print(f"  Test AUC {best_row['test_auc']:.4f}  |  F1 {best_row['test_f1']:.4f}"
          f"  |  AUC gap over runner-up: +{auc_gap:.4f}")

    rationale = {
        "LogisticRegression": (
            "Coefficients translate directly into feature importance scores that "
            "marketing can act on (e.g. 'month-to-month customers with high charges "
            "are X times more likely to churn'). Fast to retrain as new data arrives."
        ),
        "XGBClassifier": (
            "Captures non-linear interactions between features (e.g. senior citizens "
            "on fiber optic with short tenure) that logistic regression misses. "
            "Strong AUC with minimal tuning — good starting point for production."
        ),
        "RandomForest": (
            "Feature importances give marketing a ranked list of churn drivers. "
            "Robust to correlated features and outliers, with low variance across "
            "CV folds — reliable performance on unseen customer cohorts."
        ),
    }

    print(f"\n  Recommendation for marketing stakeholders:")
    print(f"  {rationale.get(best, '')}")
    print()


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run(data_dir)
