import glob
import os
import sys
from typing import Any

import pandas as pd

# allow running as `python src/data/cleaner.py` from project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.quality import (  # noqa: E402
    CRITICAL_NULL_RATE,
    EXPECTED_SCHEMA,
    TARGET_COLUMN,
    check_data_quality,
)

OUTPUT_PATH = os.path.join("data", "cleaned.csv")
_NUMERIC_DTYPES = {"int64", "float64"}


def _drop_high_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    rates = df.isnull().mean()
    cols = rates[rates > CRITICAL_NULL_RATE].index.tolist()
    if cols:
        print(f"  Dropping columns with > {CRITICAL_NULL_RATE*100:.0f}% nulls: {cols}")
        df = df.drop(columns=cols)
    return df


def _drop_null_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        return df
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    n = before - len(df)
    if n:
        print(f"  Dropped {n} rows with null target ('{TARGET_COLUMN}')")
    return df


def _handle_remaining_nulls(df: pd.DataFrame, is_time_series: bool) -> pd.DataFrame:
    if is_time_series:
        filled = df.isnull().sum().sum()
        df = df.ffill()
        if filled:
            print(f"  Forward-filled {filled} null(s) (time-series mode)")
    else:
        before = len(df)
        df = df.dropna()
        n = before - len(df)
        if n:
            print(f"  Dropped {n} rows with remaining nulls")
    return df


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        expected = EXPECTED_SCHEMA.get(col)
        if expected in _NUMERIC_DTYPES:
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = int((coerced.isna() & df[col].notna()).sum())
            if bad:
                print(f"  '{col}': {bad} unparseable value(s) set to NaN")
            df[col] = coerced
        elif expected == "object":
            df[col] = df[col].astype(str)
        else:
            # column not in schema: coerce if >= 90% parseable, else stringify
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() >= 0.9:
                df[col] = coerced
            else:
                df[col] = df[col].astype(str)
    return df


def _drop_coercion_nulls(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in df.columns
        if EXPECTED_SCHEMA.get(c) in _NUMERIC_DTYPES
    ]
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    n = before - len(df)
    if n:
        print(f"  Dropped {n} rows with unparseable numeric values after coercion")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first")
    n = before - len(df)
    if n:
        print(f"  Dropped {n} exact duplicate rows")
    return df


def clean_data(
    df: pd.DataFrame, is_time_series: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    print("Cleaning steps:")

    df = _drop_high_null_columns(df)
    df = _drop_null_target_rows(df)
    df = _handle_remaining_nulls(df, is_time_series)
    df = _convert_dtypes(df)
    df = _drop_coercion_nulls(df)
    df = _drop_duplicates(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    quality_result = check_data_quality(df)
    return df, quality_result


def _print_quality(result: dict[str, Any]) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"\nQuality gate: {status}")
    for msg in result["failures"]:
        print(f"  [FAIL] {msg}")
    for msg in result["warnings"]:
        print(f"  [WARN] {msg}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    raw_files = [f for f in files if "cleaned" not in os.path.basename(f)]
    if not raw_files:
        print(f"No raw CSV files found in '{data_dir}'")
        sys.exit(1)

    raw_df = pd.read_csv(raw_files[0])
    print(f"Loaded: {raw_files[0]}")
    print(f"Before: {len(raw_df)} rows x {len(raw_df.columns)} columns\n")

    cleaned_df, quality_result = clean_data(raw_df)

    print(f"\nAfter:  {len(cleaned_df)} rows x {len(cleaned_df.columns)} columns")
    _print_quality(quality_result)
    sys.exit(0 if quality_result["success"] else 1)
