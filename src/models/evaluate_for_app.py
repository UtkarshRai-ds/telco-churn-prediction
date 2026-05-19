"""
evaluate_for_app.py — Pre-compute evaluation artifacts for the Streamlit dashboard.

Loads the three saved models (baseline LR, XGBoost, Random Forest) plus the
tuned production model, runs them on the same held-out test split used during
training, and writes ROC curve points, PR curve points, and predicted
probability arrays to data/eval_artifacts.json.

Run once after training:
    python src/models/evaluate_for_app.py

The Streamlit app reads the JSON output — no .pkl files are loaded by the
dashboard for the comparison charts. This keeps cold starts fast and avoids
sklearn/xgboost version-pinning headaches when deploying to Streamlit Cloud.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = DATA_DIR / "eval_artifacts.json"

FEATURES_CSV = DATA_DIR / "features.csv"

# Must match the split used in src/models/train.py and tuning.py
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> tuple[list, list]:
    """ROC/PR curves can have thousands of points. Downsample for a smaller JSON
    payload without visibly changing the curve. Keeps endpoints intact."""
    if len(x) <= max_points:
        return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx].tolist(), y[idx].tolist()


def _load_test_split() -> tuple[pd.DataFrame, np.ndarray]:
    """Recreate the exact test split used during training."""
    df = pd.read_csv(FEATURES_CSV)

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected '{TARGET}' column in {FEATURES_CSV}. "
            f"Got columns: {list(df.columns)[:10]}..."
        )

    # Coerce target to 0/1 regardless of how it was saved
    y_raw = df[TARGET]
    if y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0}).astype(int).values
    else:
        y = y_raw.astype(int).values

    X = df.drop(columns=[TARGET])

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_test, y_test


def _predict_proba(bundle_or_model, X: pd.DataFrame) -> np.ndarray:
    """Handle both the production bundle ({'model','scaler','feature_names'})
    and bare model objects saved during the comparison run."""
    if isinstance(bundle_or_model, dict) and "model" in bundle_or_model:
        feature_names = bundle_or_model.get("feature_names", X.columns.tolist())
        scaler = bundle_or_model.get("scaler")
        X_in = X[feature_names]
        if scaler is not None:
            X_in = scaler.transform(X_in)
        return bundle_or_model["model"].predict_proba(X_in)[:, 1]

    # Bare model — figure out what columns it was trained on.
    # sklearn models store this in `feature_names_in_` after fitting.
    # XGBoost stores it in `feature_names_in_` too (recent versions) or
    # we fall back to selecting all numeric columns, which matches how
    # retrain_for_eval.py prepared the training set.
    model = bundle_or_model

    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        X_in = X[list(expected)]
    else:
        # Fall back: drop non-numeric columns (same as retrain_for_eval.py)
        X_in = X.select_dtypes(include="number")

    return model.predict_proba(X_in)[:, 1]


# ── Model registry ────────────────────────────────────────────────────────────
# (key in JSON, display name, pickle filename)
MODELS = [
    ("baseline",      "Logistic Regression (Baseline)", "baseline.pkl"),
    ("xgboost",       "XGBoost",                        "xgbclassifier.pkl"),
    ("random_forest", "Random Forest",                  "randomforest.pkl"),
    ("tuned_best",    "Tuned Logistic Regression",      "production_model.pkl"),
]


def main() -> None:
    print(f"Loading test split from {FEATURES_CSV.name} ...")
    X_test, y_test = _load_test_split()
    print(f"  X_test: {X_test.shape}   churn rate in test: {y_test.mean():.3f}")

    artifacts: dict = {
        "test_set": {
            "n_samples":   int(len(y_test)),
            "n_positives": int(y_test.sum()),
            "churn_rate":  float(y_test.mean()),
        },
        "models": {},
    }

    for key, display_name, filename in MODELS:
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename} not found")
            continue

        print(f"Evaluating {display_name} ({filename}) ...")
        obj = joblib.load(path)

        try:
            y_score = _predict_proba(obj, X_test)
        except Exception as exc:
            print(f"  [error] could not score {filename}: {exc}")
            continue

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        fpr_ds, tpr_ds = _downsample_curve(fpr, tpr)

        # Precision-Recall curve
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        avg_prec = average_precision_score(y_test, y_score)
        # PR curve order is recall-descending — reverse to recall-ascending for plotting
        rec_ds, prec_ds = _downsample_curve(rec[::-1], prec[::-1])

        artifacts["models"][key] = {
            "display_name": display_name,
            "roc": {
                "fpr": fpr_ds,
                "tpr": tpr_ds,
                "auc": float(roc_auc),
            },
            "pr": {
                "recall":     rec_ds,
                "precision":  prec_ds,
                "avg_precision": float(avg_prec),
            },
            # Full probability array — needed for the distribution chart on
            # the production model. Keeping it for every model is cheap
            # (~7k floats per model) and useful for future analysis.
            "y_score":  [round(float(p), 4) for p in y_score],
            "y_true":   [int(v) for v in y_test],
        }
        print(f"  AUC={roc_auc:.4f}   AP={avg_prec:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifacts, f)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}  ({size_kb:.1f} KB)")
    print(f"Models included: {list(artifacts['models'].keys())}")


if __name__ == "__main__":
    main()
