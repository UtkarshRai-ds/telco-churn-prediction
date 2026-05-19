import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features.engineer import create_features, select_features

# 21 original cleaned.csv columns + 14 engineered = 35 total
EXPECTED_TOTAL_COLS = 35

BINARY_FEATURES = [
    "contract_flexibility_risk",
    "is_long_term_customer",
    "high_data_user",
    "senior_internet_risk",
    "month_to_month_new_customer",
    "long_tenure_low_engagement",
    "contract_spend_risk",
]


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    path = os.path.join(_ROOT, "data", "cleaned.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def engineered_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    return create_features(cleaned_df)


@pytest.fixture(scope="module")
def selected_df(engineered_df: pd.DataFrame) -> pd.DataFrame:
    df, _ = select_features(engineered_df)
    return df


# ── create_features() ─────────────────────────────────────────────────────────

class TestCreateFeatures:
    def test_total_column_count(self, engineered_df):
        assert engineered_df.shape[1] == EXPECTED_TOTAL_COLS, (
            f"Expected {EXPECTED_TOTAL_COLS} columns, got {engineered_df.shape[1]}"
        )

    def test_row_count_unchanged(self, cleaned_df, engineered_df):
        assert engineered_df.shape[0] == cleaned_df.shape[0]

    def test_no_nan_in_numeric_columns(self, engineered_df):
        numeric = engineered_df.select_dtypes(include="number")
        null_counts = numeric.isnull().sum()
        assert null_counts.sum() == 0, (
            f"NaN found in numeric columns: {null_counts[null_counts > 0].to_dict()}"
        )

    def test_total_services_range(self, engineered_df):
        col = engineered_df["total_services"]
        assert col.min() >= 0, f"total_services min is {col.min()}, expected >= 0"
        assert col.max() <= 8, f"total_services max is {col.max()}, expected <= 8"

    def test_monthly_spending_trend_is_finite(self, engineered_df):
        col = engineered_df["monthly_spending_trend"]
        assert np.isfinite(col).all(), "monthly_spending_trend contains inf or -inf"

    def test_binary_features_are_zero_or_one(self, engineered_df):
        for col in BINARY_FEATURES:
            unique = set(engineered_df[col].unique())
            assert unique.issubset({0, 1}), (
                f"'{col}' has non-binary values: {unique - {0, 1}}"
            )

    def test_support_adoption_range(self, engineered_df):
        # TechSupport + OnlineBackup + DeviceProtection — max is 3
        col = engineered_df["support_adoption"]
        assert col.min() >= 0
        assert col.max() <= 3

    def test_tenure_quartile_is_categorical(self, engineered_df):
        assert hasattr(engineered_df["tenure_quartile"], "cat"), (
            "tenure_quartile should be a Categorical column"
        )
        assert set(engineered_df["tenure_quartile"].cat.categories) == {"Q1", "Q2", "Q3", "Q4"}


# ── select_features() ────────────────────────────────────────────────────────

class TestSelectFeatures:
    def test_tenure_quartile_raw_column_removed(self, selected_df):
        # select_features one-hot encodes tenure_quartile and drops the original
        assert "tenure_quartile" not in selected_df.columns

    def test_tenure_quartile_dummies_present(self, selected_df):
        dummy_cols = [c for c in selected_df.columns if c.startswith("tenure_quartile_")]
        assert len(dummy_cols) == 4, (
            f"Expected 4 tenure_quartile dummy columns, got {len(dummy_cols)}: {dummy_cols}"
        )

    def test_tenure_quartile_dummies_are_binary(self, selected_df):
        dummy_cols = [c for c in selected_df.columns if c.startswith("tenure_quartile_")]
        for col in dummy_cols:
            # pandas get_dummies returns bool in pandas >= 1.5; cast to int for check
            unique = set(selected_df[col].astype(int).unique())
            assert unique.issubset({0, 1}), f"'{col}' has non-binary values: {unique}"

    def test_row_count_unchanged(self, engineered_df, selected_df):
        assert selected_df.shape[0] == engineered_df.shape[0]

    def test_returns_removal_report(self, engineered_df):
        _, removed = select_features(engineered_df)
        assert "high_correlation" in removed
        assert "low_variance" in removed
