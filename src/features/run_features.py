import os
import sys
import time

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features.engineer import create_features, select_features


def run(data_dir: str = "data") -> pd.DataFrame:
    t0 = time.time()

    src = os.path.join(data_dir, "cleaned.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found — run cleaner.py first")

    df = pd.read_csv(src)
    print(f"Loaded:           {src}")
    print(f"Raw shape:        {df.shape[0]} rows x {df.shape[1]} columns")

    df = create_features(df)
    print(f"After engineer:   {df.shape[0]} rows x {df.shape[1]} columns")

    df, removed = select_features(df)
    n_corr = len(removed["high_correlation"])
    n_var  = len(removed["low_variance"])
    print(f"After selection:  {df.shape[0]} rows x {df.shape[1]} columns"
          f"  (-{n_corr} high-corr, -{n_var} low-var, +4 OHE)")

    out = os.path.join(data_dir, "features.csv")
    df.to_csv(out, index=False)
    print(f"Saved:            {out}")

    numeric_kept = df.select_dtypes(include="number").columns.tolist()
    print(f"\nKept features ({len(numeric_kept)}):")
    for col in numeric_kept:
        print(f"  {col}")

    print(f"\nElapsed: {time.time() - t0:.2f}s")
    return df


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run(data_dir)
