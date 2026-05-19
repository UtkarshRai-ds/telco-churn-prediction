import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.quality import EXPECTED_SCHEMA, check_data_quality


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    path = os.path.join(_ROOT, "data", "cleaned.csv")
    return pd.read_csv(path)


# ── Real data: all 5 checks must pass ────────────────────────────────────────

class TestQualityOnRealData:
    def test_overall_success(self, cleaned_df):
        result = check_data_quality(cleaned_df)
        assert result["success"] is True, f"Quality gate failed: {result['failures']}"

    def test_no_failures(self, cleaned_df):
        result = check_data_quality(cleaned_df)
        assert result["failures"] == []

    def test_result_has_required_keys(self, cleaned_df):
        result = check_data_quality(cleaned_df)
        assert set(result.keys()) == {"success", "failures", "warnings", "statistics"}

    def test_statistics_row_and_column_counts(self, cleaned_df):
        result = check_data_quality(cleaned_df)
        stats = result["statistics"]
        assert stats["total_rows"] == len(cleaned_df)
        assert stats["total_columns"] == len(cleaned_df.columns)

    def test_statistics_null_count_matches(self, cleaned_df):
        result = check_data_quality(cleaned_df)
        expected_nulls = int(cleaned_df.isnull().sum().sum())
        assert result["statistics"]["total_nulls"] == expected_nulls


# ── Broken data: gate should catch schema and null violations ─────────────────

class TestQualityOnBrokenData:
    def test_missing_required_column_fails(self, cleaned_df):
        # Drop a required numeric column — _check_schema should fire
        broken = cleaned_df.drop(columns=["tenure"])
        result = check_data_quality(broken)
        assert result["success"] is False
        assert any("tenure" in msg for msg in result["failures"])

    def test_all_null_column_fails(self, cleaned_df):
        # Set MonthlyCharges entirely to NaN — null rate (100%) > CRITICAL_NULL_RATE (50%)
        broken = cleaned_df.copy()
        broken["MonthlyCharges"] = None
        result = check_data_quality(broken)
        assert result["success"] is False
        assert any("MonthlyCharges" in msg for msg in result["failures"])

    def test_duplicate_rows_are_not_caught(self, cleaned_df):
        # The quality gate has no duplicate-row check — this documents that gap.
        with_dup = pd.concat([cleaned_df, cleaned_df.head(1)], ignore_index=True)
        result = check_data_quality(with_dup)
        assert result["success"] is True

    def test_too_few_rows_fails(self):
        # Build a tiny but schema-valid dataframe — _check_row_count should fire
        row = {col: (0 if dtype in ("int64", "float64") else "No")
               for col, dtype in EXPECTED_SCHEMA.items()}
        row["customerID"] = "test-001"
        row["Churn"] = "No"
        row["TotalCharges"] = "0.0"
        tiny = pd.DataFrame([row] * 5)  # 5 rows < MIN_ROWS_CRITICAL (100)
        result = check_data_quality(tiny)
        assert result["success"] is False
        assert any("rows" in msg for msg in result["failures"])
