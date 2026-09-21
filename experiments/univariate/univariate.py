"""
FIAM 2026 - Univariate feature check.

For each of the 147 characteristics individually (no other predictors, no
interactions), measures how much standalone information it carries about
next-month excess return (ret_exc_lead1m), two ways:

  1. Information Coefficient (IC): the monthly cross-sectional Pearson
     correlation between the rank-transformed characteristic and next-month
     return, averaged over the OOS window (2021-01 to 2026-08) with its
     t-stat (mean / std * sqrt(n_months), same IR-style construction used
     elsewhere in this project). Because each characteristic is already
     rank-transformed to [-1, 1] (a monotonic transform), Pearson correlation
     on the transformed value equals Spearman rank correlation on the raw
     value. No model is fit for this -- it is a descriptive statistic on
     already-known values and their realized next-month return, so there is
     no look-ahead risk to manage.

  2. Single-factor walk-forward OOS R^2: fit `target ~ var` alone (one
     univariate OLS per characteristic per fold, closed-form), using the same
     expanding-window / annual-refit walk-forward schedule as ols.py /
     xgb.py, pooled over all OOS predictions (2021-01 to 2026-08). Directly
     comparable in magnitude to the 147-feature OOS R^2 already reported in
     docs/OLS.md (-0.0086%) and docs/XGB.md (-0.1265%).

Self-contained: data loading, rank transform, and the walk-forward fold
schedule are ported from ols.py (not imported), matching the project
convention that each analysis script stands alone.

Run: .venv/bin/python univariate.py
Output: output/univariate_results.csv, output/univariate_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FIAM_DIR = BASE / "fiam"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"

OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")

# ---------------------------------------------------------------------------
# 1. Load data and build the model table (ported from ols.py)
# ---------------------------------------------------------------------------


def load_model_table() -> tuple[pd.DataFrame, list[str]]:
    stock_vars = pd.read_csv(FACTOR_LIST)["variable"].tolist()
    keep_cols = stock_vars + ["permno", "eom", TARGET_COL]
    df = pd.read_parquet(CHARS_FILE, columns=keep_cols)
    df["eom"] = pd.to_datetime(df["eom"])

    df["target_month"] = (df["eom"] + pd.offsets.MonthBegin(1)).values.astype(
        "datetime64[M]"
    )
    df["target_month"] = pd.to_datetime(df["target_month"]) + pd.offsets.MonthEnd(0)

    df = df[df[TARGET_COL].notna()].copy()
    return df, stock_vars


def cross_sectional_rank_transform(df: pd.DataFrame, stock_vars: list[str]) -> pd.DataFrame:
    """Median-fill then rank-transform each predictor to [-1, 1], per
    characteristic month (eom), matching ols.py / xgb.py's convention."""
    out = df.copy()
    g = out.groupby("eom")
    for var in stock_vars:
        med = g[var].transform("median")
        filled = out[var].fillna(med)
        filled = filled.fillna(0.0)
        out[var] = filled

    g = out.groupby("eom")
    for var in stock_vars:
        r = g[var].rank(method="dense")
        rmax = r.groupby(out["eom"]).transform("max")
        out[var] = np.where(rmax > 0, (r / rmax) * 2 - 1, 0.0)
    return out


# ---------------------------------------------------------------------------
# 2. Information Coefficient (no model fitting, purely descriptive)
# ---------------------------------------------------------------------------


def compute_ic(df: pd.DataFrame, stock_vars: list[str]) -> pd.DataFrame:
    sub = df[(df["target_month"] >= OOS_START) & (df["target_month"] <= OOS_END)]
    months = sorted(sub["target_month"].unique())
    ic = np.full((len(months), len(stock_vars)), np.nan)

    for i, month in enumerate(months):
        grp = sub[sub["target_month"] == month]
        X = grp[stock_vars].values
        y = grp[TARGET_COL].values
        xm = X - X.mean(axis=0)
        ym = y - y.mean()
        num = (xm * ym[:, None]).sum(axis=0)
        denom = np.sqrt((xm**2).sum(axis=0) * (ym**2).sum())
        ic[i] = np.where(denom > 0, num / denom, 0.0)

    icdf = pd.DataFrame(ic, index=pd.to_datetime(months), columns=stock_vars)
    ic_mean = icdf.mean()
    ic_std = icdf.std()
    n_months = len(icdf)
    ic_tstat = ic_mean / ic_std * np.sqrt(n_months)
    return pd.DataFrame(
        {"ic_mean": ic_mean, "ic_std": ic_std, "ic_tstat": ic_tstat, "ic_n_months": n_months}
    )


# ---------------------------------------------------------------------------
# 3. Single-factor walk-forward OOS R^2 (closed-form univariate OLS per fold)
# ---------------------------------------------------------------------------


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def univariate_walk_forward_r2(df: pd.DataFrame, stock_vars: list[str]) -> pd.Series:
    test_years = range(2021, 2027)
    n_vars = len(stock_vars)
    sse = np.zeros(n_vars)
    sst = 0.0

    for test_year in test_years:
        test_start, test_end = year_bounds(test_year)
        test_end = min(test_end, OOS_END)
        val_start, _ = year_bounds(test_year - 2)

        train_mask = df["target_month"] < val_start
        test_mask = (df["target_month"] >= test_start) & (df["target_month"] <= test_end)

        train = df[train_mask]
        test = df[test_mask]
        if train.empty or test.empty:
            continue

        Xtr = train[stock_vars].values
        ytr = train[TARGET_COL].values
        Xte = test[stock_vars].values
        yte = test[TARGET_COL].values

        # closed-form univariate OLS per column: b = cov(x,y)/var(x), a = ybar - b*xbar
        mean_x = Xtr.mean(axis=0)
        mean_y = ytr.mean()
        cov = ((Xtr - mean_x) * (ytr - mean_y)[:, None]).mean(axis=0)
        var_x = Xtr.var(axis=0)
        b = np.where(var_x > 0, cov / var_x, 0.0)
        a = mean_y - b * mean_x

        pred = a[None, :] + b[None, :] * Xte
        errors = yte[:, None] - pred
        sse += (errors**2).sum(axis=0)
        sst += (yte**2).sum()

        print(
            f"[fold {test_year}] train={len(train):,} rows, test={len(test):,} rows"
        )

    r2 = 1.0 - sse / sst
    return pd.Series(r2, index=stock_vars, name="oos_r2_univariate")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors")

    print("Cross-sectional preprocessing (median fill + rank to [-1,1] per month)...")
    df = cross_sectional_rank_transform(df, stock_vars)

    print("\nComputing monthly Information Coefficient per characteristic (2021-01 to 2026-08)...")
    ic_df = compute_ic(df, stock_vars)

    print("Computing single-factor walk-forward OOS R^2 per characteristic...")
    r2_series = univariate_walk_forward_r2(df, stock_vars)

    results = ic_df.join(r2_series).sort_values("ic_tstat", ascending=False)
    results.index.name = "variable"
    results.to_csv(OUTPUT / "univariate_results.csv")

    with open(OUTPUT / "univariate_results.json", "w") as f:
        json.dump(
            {
                "oos_window": [str(OOS_START.date()), str(OOS_END.date())],
                "n_characteristics": len(stock_vars),
                "results": results.reset_index().to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print(f"\nTop 15 by |IC t-stat| (strongest standalone monthly rank correlation):")
    by_abs_t = results.reindex(results["ic_tstat"].abs().sort_values(ascending=False).index)
    print(by_abs_t.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nTop 15 by single-factor OOS R^2:")
    by_r2 = results.sort_values("oos_r2_univariate", ascending=False)
    print(by_r2.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

    n_positive_r2 = (results["oos_r2_univariate"] > 0).sum()
    n_sig_ic = (results["ic_tstat"].abs() > 2).sum()
    print(f"\n{n_positive_r2} / {len(stock_vars)} characteristics have positive single-factor OOS R^2")
    print(f"{n_sig_ic} / {len(stock_vars)} characteristics have |IC t-stat| > 2")

    print("\nFull results written to output/univariate_results.csv and .json")
