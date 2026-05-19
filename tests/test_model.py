import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope="module")
def model_bundle() -> dict:
    path = os.path.join(_ROOT, "models", "production_model.pkl")
    return joblib.load(path)


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    path = os.path.join(_ROOT, "data", "features.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def X_test(model_bundle, features_df) -> np.ndarray:
    feature_names = model_bundle["feature_names"]
    X = features_df[feature_names]
    return model_bundle["scaler"].transform(X)


# ── Bundle structure ──────────────────────────────────────────────────────────

class TestModelLoad:
    def test_loads_as_dict(self, model_bundle):
        assert isinstance(model_bundle, dict)

    def test_bundle_has_required_keys(self, model_bundle):
        assert "model" in model_bundle
        assert "scaler" in model_bundle
        assert "feature_names" in model_bundle

    def test_feature_names_is_nonempty_list(self, model_bundle):
        feature_names = model_bundle["feature_names"]
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_model_has_predict_proba(self, model_bundle):
        assert hasattr(model_bundle["model"], "predict_proba")

    def test_model_has_predict(self, model_bundle):
        assert hasattr(model_bundle["model"], "predict")

    def test_scaler_has_transform(self, model_bundle):
        assert hasattr(model_bundle["scaler"], "transform")


# ── Prediction correctness ────────────────────────────────────────────────────

class TestModelPrediction:
    def test_single_row_probability_in_range(self, model_bundle, X_test):
        prob = model_bundle["model"].predict_proba(X_test[:1])[0, 1]
        assert 0.0 <= prob <= 1.0, f"Single-row probability out of range: {prob}"

    def test_batch_probabilities_in_range(self, model_bundle, X_test):
        probs = model_bundle["model"].predict_proba(X_test)[:, 1]
        assert np.all(probs >= 0.0), "Some probabilities are negative"
        assert np.all(probs <= 1.0), "Some probabilities exceed 1.0"

    def test_predict_proba_output_shape(self, model_bundle, X_test, features_df):
        probs = model_bundle["model"].predict_proba(X_test)
        assert probs.shape == (len(features_df), 2), (
            f"Expected shape ({len(features_df)}, 2), got {probs.shape}"
        )

    def test_class_probabilities_sum_to_one(self, model_bundle, X_test):
        probs = model_bundle["model"].predict_proba(X_test[:20])
        np.testing.assert_allclose(
            probs.sum(axis=1), 1.0, atol=1e-6,
            err_msg="Class probabilities do not sum to 1.0"
        )

    def test_binary_predictions_are_zero_or_one(self, model_bundle, X_test):
        preds = model_bundle["model"].predict(X_test)
        assert set(preds).issubset({0, 1}), f"Unexpected prediction values: {set(preds)}"

    def test_feature_names_match_features_csv(self, model_bundle, features_df):
        # Every model feature must be a column in features.csv
        missing = [f for f in model_bundle["feature_names"] if f not in features_df.columns]
        assert missing == [], f"Feature(s) in model not found in features.csv: {missing}"
