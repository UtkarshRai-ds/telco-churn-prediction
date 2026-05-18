import glob
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def _yn(series: pd.Series) -> pd.Series:
    """Map Yes->1, everything else (No / No internet service / etc.) -> 0."""
    return (series == "Yes").astype(int)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges arrives as object in raw/cleaned data; coerce once here
    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    monthly = df["MonthlyCharges"]
    tenure  = df["tenure"]
    total   = df["TotalCharges"]
    contract = df["Contract"]

    # ── Domain-specific features ──────────────────────────────────────────────

    # High CLV churners represent maximum revenue loss; flags customers worth retaining
    df["customer_lifetime_value"] = tenure * monthly

    # Month-to-month carries no exit penalty, giving customers full churn freedom
    df["contract_flexibility_risk"] = (contract == "Month-to-month").astype(int)

    # Customers past 24 months have demonstrated sustained commitment to the provider
    df["is_long_term_customer"] = (tenure >= 24).astype(int)

    # Each additional service increases switching cost and deepens ecosystem lock-in
    df["total_services"] = sum(_yn(df[col]) for col in _SERVICE_COLS)

    # Fiber optic users depend on high speeds; ISP-switching is disruptive and costly
    df["high_data_user"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Investment in support products signals active platform reliance and satisfaction
    df["support_adoption"] = (
        _yn(df["TechSupport"]) + _yn(df["OnlineBackup"]) + _yn(df["DeviceProtection"])
    )

    # ── Statistical features ──────────────────────────────────────────────────

    # High ratio = large monthly bill relative to tenure; signals potential price sensitivity
    df["monthly_spending_trend"] = monthly / (tenure + 1)

    # Normalises spend against the customer base; values > 1 are above-average spenders
    df["relative_monthly_spend"] = monthly / monthly.mean()

    # Divides customers into lifecycle quartiles (Q1 = newest, Q4 = most tenured)
    df["tenure_quartile"] = pd.qcut(tenure, q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Large deviation from tenure * monthly may indicate credits, promos, or mid-cycle changes
    df["charge_consistency"] = (total - monthly * tenure).abs()

    # ── Interaction features ──────────────────────────────────────────────────

    # Seniors on high-bandwidth services face both higher switching friction and tech frustration
    df["senior_internet_risk"] = df["SeniorCitizen"] * df["high_data_user"]

    # No contract lock-in + still in trial window = highest observed churn probability
    df["month_to_month_new_customer"] = (
        (contract == "Month-to-month").astype(int) * (tenure < 6).astype(int)
    )

    # Long-tenure, low-engagement customers are inertial and vulnerable to competitive offers
    df["long_tenure_low_engagement"] = (
        (tenure >= 24).astype(int) * (df["total_services"] < 3).astype(int)
    )

    # High bill + no lock-in = strongest financial motivation to shop around and leave
    df["contract_spend_risk"] = (
        (contract == "Month-to-month").astype(int) * (monthly > monthly.median()).astype(int)
    )

    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    removed: dict[str, list[str]] = {"high_correlation": [], "low_variance": []}

    # ── One-hot encode tenure_quartile ────────────────────────────────────────
    if "tenure_quartile" in df.columns:
        dummies = pd.get_dummies(df["tenure_quartile"], prefix="tenure_quartile")
        df = df.drop(columns=["tenure_quartile"]).join(dummies)

    # ── Work only on numeric columns; preserve non-numeric passthrough ────────
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    numeric = df.select_dtypes(include="number").copy()

    # ── Drop high-correlation features (> 0.95) ───────────────────────────────
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr: list[str] = []
    for col in upper.columns:
        partners = upper.index[upper[col] > 0.95].tolist()
        if partners:
            to_drop_corr.append(col)
            for partner in partners:
                removed["high_correlation"].append(
                    f"{col} (corr={corr.loc[col, partner]:.3f} with '{partner}')"
                )
    numeric = numeric.drop(columns=to_drop_corr)

    # ── Drop low-variance features (< 1% of max variance) ────────────────────
    # Normalize to [0, 1] before comparing so large-scale features (e.g.
    # customer_lifetime_value) don't inflate the threshold and wipe out binary
    # or small-range features that still carry genuine signal.
    scale = numeric.max() - numeric.min()
    normalized = numeric / (scale.replace(0, 1))  # avoid div-by-zero for constants
    variances = normalized.var()
    threshold = 0.01 * variances.max()
    to_drop_var = variances[variances < threshold].index.tolist()
    for col in to_drop_var:
        removed["low_variance"].append(
            f"{col} (normalized_var={variances[col]:.6f}, threshold={threshold:.6f})"
        )
    numeric = numeric.drop(columns=to_drop_var)

    # ── Reassemble dataframe ──────────────────────────────────────────────────
    df_selected = pd.concat([df[non_numeric], numeric], axis=1)

    total_removed = len(to_drop_corr) + len(to_drop_var)
    return df_selected, removed


def _print_selection_log(removed: dict[str, list[str]]) -> None:
    corr_drops = removed["high_correlation"]
    var_drops   = removed["low_variance"]

    if not corr_drops and not var_drops:
        print("  No features removed.")
        return

    if corr_drops:
        print(f"  Removed {len(corr_drops)} high-correlation feature(s):")
        for msg in corr_drops:
            print(f"    [corr] {msg}")

    if var_drops:
        print(f"  Removed {len(var_drops)} low-variance feature(s):")
        for msg in var_drops:
            print(f"    [var]  {msg}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    src = os.path.join(data_dir, "cleaned.csv")
    if not os.path.exists(src):
        print(f"cleaned.csv not found in '{data_dir}' — run cleaner.py first")
        sys.exit(1)

    df_raw = pd.read_csv(src)
    print(f"Loaded:   {src}")
    print(f"Raw:      {df_raw.shape[0]} rows x {df_raw.shape[1]} columns\n")

    df_feat = create_features(df_raw)
    new_cols = [c for c in df_feat.columns if c not in df_raw.columns]
    print(f"After create_features: {df_feat.shape[1]} columns (+{len(new_cols)} engineered)\n")

    print("Running select_features...")
    df_sel, removed = select_features(df_feat)
    _print_selection_log(removed)
    print(f"\nFinal:    {df_sel.shape[0]} rows x {df_sel.shape[1]} columns")

    out = os.path.join(data_dir, "featured.csv")
    df_sel.to_csv(out, index=False)
    print(f"Saved to  {out}")
