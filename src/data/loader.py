import pandas as pd
import glob
import os
import sys


def find_csv(data_dir: str) -> str:
    pattern = os.path.join(data_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
    if len(files) > 1:
        print(f"Multiple CSVs found; loading: {files[0]}")
    return files[0]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {path}\n")
    return df


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Shape: {rows} rows x {cols} columns\n")


def print_column_info(df: pd.DataFrame) -> None:
    print("Column names and data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    print()


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("No numeric columns found.\n")
        return
    stats = numeric.agg(["mean", "std", "min", "max"])
    print("Summary statistics (numeric columns):")
    print(stats.to_string())
    print()


def print_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0]
    if report.empty:
        print("No missing values found.\n")
    else:
        print("Missing value counts and percentages:")
        print(report.to_string())
        print()


def run(data_dir: str = "data") -> pd.DataFrame:
    path = find_csv(data_dir)
    df = load_csv(path)
    print_shape(df)
    print_column_info(df)
    print_summary_stats(df)
    print_missing_values(df)
    return df


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run(data_dir)
