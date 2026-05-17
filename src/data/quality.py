import glob
import os
import sys
from typing import Any

import pandas as pd

# Expected schema: column -> dtype string
EXPECTED_SCHEMA: dict[str, str] = {
    "customerID": "object",
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "object",
    "Churn": "object",
}

# Inclusive (min, max) bounds for numeric columns
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 1000),
    "MonthlyCharges": (0, 10_000),
}

TARGET_COLUMN = "Churn"
CRITICAL_NULL_RATE = 0.50
WARN_NULL_RATE = 0.20
MIN_ROWS_CRITICAL = 100
MIN_ROWS_WARN = 1_000
MIN_CLASS_PCT = 0.05
IMBALANCE_THRESHOLD = 0.80


# --- individual checks ---

def _check_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    missing = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    if missing:
        failures.append(f"Missing required columns: {missing}")
    for col, expected in EXPECTED_SCHEMA.items():
        if col in df.columns and str(df[col].dtype) != expected:
            warnings.append(
                f"'{col}': expected dtype '{expected}', got '{df[col].dtype}'"
            )
    return failures, warnings


def _check_row_count(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    n = len(df)
    if n < MIN_ROWS_CRITICAL:
        return [f"Only {n} rows — minimum required is {MIN_ROWS_CRITICAL}"], []
    if n < MIN_ROWS_WARN:
        return [], [f"Only {n} rows — fewer than {MIN_ROWS_WARN} may reduce model reliability"]
    return [], []


def _check_null_rates(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    rates = df.isnull().mean()
    for col, rate in rates.items():
        pct = rate * 100
        if rate > CRITICAL_NULL_RATE:
            failures.append(f"'{col}' has {pct:.1f}% nulls (> {CRITICAL_NULL_RATE*100:.0f}% threshold)")
        elif rate > WARN_NULL_RATE:
            warnings.append(f"'{col}' has {pct:.1f}% nulls (> {WARN_NULL_RATE*100:.0f}%)")
    return failures, warnings


def _check_value_ranges(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        below = int((series < lo).sum())
        above = int((series > hi).sum())
        if below:
            failures.append(f"'{col}': {below} value(s) below minimum ({lo})")
        if above:
            warnings.append(f"'{col}': {above} value(s) above expected maximum ({hi})")
    return failures, warnings


def _check_target_distribution(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    if TARGET_COLUMN not in df.columns:
        warnings.append(f"Target column '{TARGET_COLUMN}' not found — skipping distribution check")
        return failures, warnings

    dist = df[TARGET_COLUMN].value_counts(normalize=True)
    if len(dist) < 2:
        failures.append(f"Target '{TARGET_COLUMN}' has only {len(dist)} class — need at least 2")

    for cls, pct in dist.items():
        if pct < MIN_CLASS_PCT:
            failures.append(
                f"Target class '{cls}' is only {pct*100:.1f}% of data (< {MIN_CLASS_PCT*100:.0f}% minimum)"
            )
        elif pct > IMBALANCE_THRESHOLD:
            warnings.append(
                f"Target class '{cls}' is {pct*100:.1f}% of data (> {IMBALANCE_THRESHOLD*100:.0f}%) — imbalanced"
            )
    return failures, warnings


# --- public API ---

def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    all_failures: list[str] = []
    all_warnings: list[str] = []

    for check in (
        _check_schema,
        _check_row_count,
        _check_null_rates,
        _check_value_ranges,
        _check_target_distribution,
    ):
        f, w = check(df)
        all_failures.extend(f)
        all_warnings.extend(w)

    return {
        "success": len(all_failures) == 0,
        "failures": all_failures,
        "warnings": all_warnings,
        "statistics": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_nulls": int(df.isnull().sum().sum()),
            "total_nulls_by_column": df.isnull().sum().to_dict(),
            "numeric_columns": list(df.select_dtypes(include="number").columns),
            "object_columns": list(df.select_dtypes(include="object").columns),
        },
    }


def _print_report(result: dict[str, Any]) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"Data quality gate: {status}\n")

    if result["failures"]:
        print("FAILURES (critical):")
        for msg in result["failures"]:
            print(f"  [FAIL] {msg}")
        print()

    if result["warnings"]:
        print("WARNINGS:")
        for msg in result["warnings"]:
            print(f"  [WARN] {msg}")
        print()

    stats = result["statistics"]
    print("Statistics:")
    print(f"  total_rows      : {stats['total_rows']}")
    print(f"  total_columns   : {stats['total_columns']}")
    print(f"  total_nulls     : {stats['total_nulls']}")
    print(f"  numeric_columns : {stats['numeric_columns']}")
    print(f"  object_columns  : {stats['object_columns']}")
    nulls_by_col = {k: v for k, v in stats["total_nulls_by_column"].items() if v > 0}
    if nulls_by_col:
        print(f"  nulls_by_column : {nulls_by_col}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    pattern = os.path.join(data_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No CSV files found in '{data_dir}'")
        sys.exit(1)

    df = pd.read_csv(files[0])
    result = check_data_quality(df)
    _print_report(result)
    sys.exit(0 if result["success"] else 1)
