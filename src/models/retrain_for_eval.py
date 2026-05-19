"""
retrain_for_eval.py — Retrain XGBoost and Random Forest on the SAME
feature-engineered dataset used by the tuned Logistic Regression.

Why this exists
---------------
The original XGBoost and Random Forest models were trained on the raw
categorical columns (Contract, Dependents, InternetService, ...), while
the tuned LR was trained on the 36-column engineered features.csv.

That makes the three models incomparable: same dataset, different inputs.
This script fixes it by retraining XGB and RF on the exact same
post-engineering feature matrix as the tuned LR.

What it does
------------
1. Loads data/features.csv
2. Recreates the same train/test split (random_state=42, test_size=0.2,
   stratified) used by tuning.py
3. Trains a default-ish XGBoost and Random Forest on the training set
4. Reports test-set metrics so we can confirm they're sensible
5. Saves fresh models to:
       models/xgbclassifier.pkl   (overwrites the old one)
       models/randomforest.pkl    (overwrites the old one)
6. Also writes a small status JSON so we have a record of what changed

After this runs, re-run:  python src/models/evaluate_for_app.py

Run from the project root:
    python src/models/retrain_for_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed in this environment. "
        "Install it with:  pip install xgboost"
    ) from exc


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

FEATURES_CSV = DATA_DIR / "features.csv"
STATUS_JSON = MODELS_DIR / "retrain_status.json"

# Must match the split used in tuning.py and evaluate_for_app.py
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_xy() -> tuple[pd.DataFrame, np.ndarray]:
    """Load features.csv and split into X (features) and y (binary target)."""
    df = pd.read_csv(FEATURES_CSV)

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected '{TARGET}' column in {FEATURES_CSV}. "
            f"Got: {list(df.columns)[:10]}..."
        )

    y_raw = df[TARGET]
    if y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0}).astype(int).values
    else:
        y = y_raw.astype(int).values

    X = df.drop(columns=[TARGET])

    # XGBoost and RF need fully numeric input. If any non-numeric columns
    # slipped through feature engineering, drop them with a warning so the
    # script doesn't fail silently.
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"  [warn] dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    return X, y


def _evaluate(name: str, model, X_test, y_test) -> dict:
    """Compute and pretty-print the standard metric set."""
    y_pred  = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall":    float(recall_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred)),
        "auc_roc":   float(roc_auc_score(y_test, y_score)),
    }
    print(f"  {name}:")
    for k, v in metrics.items():
        print(f"    {k:<10s} {v:.4f}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading {FEATURES_CSV.name} ...")
    X, y = _load_xy()
    print(f"  X: {X.shape}   churn rate: {y.mean():.3f}")
    print(f"  feature columns: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  train: {X_train.shape}   test: {X_test.shape}")

    # ── Baseline Logistic Regression ──────────────────────────────────────────
    # An untuned, no-class-weight LR. This is the "what would we get with
    # zero tuning?" reference point. The tuned LR's improvement over this
    # baseline is the real measure of what hyperparameter search added.
    print("\nTraining Baseline Logistic Regression ...")
    baseline = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = _evaluate("Baseline LR", baseline, X_test, y_test)

    baseline_path = MODELS_DIR / "baseline.pkl"
    joblib.dump(baseline, baseline_path)
    print(f"  saved -> {baseline_path.relative_to(ROOT)}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost ...")
    # Handle class imbalance via scale_pos_weight = (negatives / positives)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_metrics = _evaluate("XGBoost", xgb, X_test, y_test)

    xgb_path = MODELS_DIR / "xgbclassifier.pkl"
    joblib.dump(xgb, xgb_path)
    print(f"  saved -> {xgb_path.relative_to(ROOT)}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\nTraining Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = _evaluate("Random Forest", rf, X_test, y_test)

    rf_path = MODELS_DIR / "randomforest.pkl"
    joblib.dump(rf, rf_path)
    print(f"  saved -> {rf_path.relative_to(ROOT)}")

    # ── Status file ───────────────────────────────────────────────────────────
    status = {
        "note": (
            "Baseline LR, XGBoost, and Random Forest retrained on features.csv "
            "(engineered feature matrix) so they're directly comparable to the "
            "tuned LR. Baseline is an untuned LR with no class balancing — "
            "the reference point that shows what tuning actually added."
        ),
        "feature_count": int(X.shape[1]),
        "test_size":     TEST_SIZE,
        "random_state":  RANDOM_STATE,
        "baseline":      baseline_metrics,
        "xgboost":       xgb_metrics,
        "random_forest": rf_metrics,
    }
    with open(STATUS_JSON, "w") as f:
        json.dump(status, f, indent=2)
    print(f"\nWrote {STATUS_JSON.relative_to(ROOT)}")
    print("\nDone. Next step:")
    print("    python src/models/evaluate_for_app.py")


if __name__ == "__main__":
    main()
