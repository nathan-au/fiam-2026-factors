"""
FIAM 2026 - PCA stability diagnostic (operationalizes "Principal component
error in high-dimensional factor models", Bernstein, Goldberg, Gunther,
Kercheval, Lan, Lin & Yao, arXiv 2609.20550, Sep 2026).

Unlike the other 7 scripts in this batch, this is NOT a return-prediction or
portfolio-construction model -- per docs/PAPERS.md §3, the paper itself is a
diagnostic tool, not a strategy: it decomposes PCA-based factor-estimation
error into an "out-of-subspace" component (how far an estimated principal
subspace is from the true one) and an "in-subspace" component (finite-sample
noise within the right subspace). It only matters here if a PCA/shrinkage
-based prediction model (e.g. Kozak-Nagel-Santosh, docs/PAPERS.md §1) is
built on top of the 147 characteristics -- this script answers the
prerequisite question: **are the principal components of this panel stable
enough to trust a shrinkage prior built on them?**

This does NOT reproduce the paper's exact asymptotic error bounds. Their
verbatim abstract (read directly) illustrates the "out-of-subspace error
dominates" finding via a three-factor SIMULATION of the US equity market
(synthetic, low-dimensional) -- not on a real high-dimensional panel like
this project's 147 characteristics, so their specific numeric finding isn't
directly transferable to this setting; only their error-decomposition
FRAMEWORK is. What IS implemented, a standard and
directly interpretable stand-in for their "out-of-subspace" error concept:
**principal-angle stability between independently estimated subspaces.**
For each walk-forward fold's training data, the training months are split in
half chronologically; the top-k principal components of the 147
rank-transformed characteristics' cross-sectional covariance are estimated
separately on each half; the cosine of the principal angles between the two
half-period k-dimensional subspaces (via SVD of the cross-product of the two
orthonormal loading matrices) measures how much the ESTIMATED subspace would
have differed had history split differently -- a direct, computable proxy
for "how reliable is this PCA estimate," for k = 1..15.

Self-contained: data loading and rank transform are ported from ols.py (not
imported). No target variable, no walk-forward test/val split, no LP, no
beta-neutral portfolio, no OOS R^2 -- this script produces a diagnostic
table, not predictions or holdings.

Run: .venv/bin/python experiments/pca_diagnostic/pca_diagnostic.py
Outputs (all in output/): pca_stability_by_k.csv, pca_variance_explained.csv,
    pca_diagnostic_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent  # experiments/<name>/
BASE = HERE.parents[1]  # project root: fiam/ data and cache/ are shared
FIAM_DIR = BASE / "fiam"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"
OOS_END = pd.Timestamp("2026-08-31")
MAX_K = 15

# ---------------------------------------------------------------------------
# 1. Load data and build the model table (ported from ols.py, target kept
#    only to define the same walk-forward training windows other scripts
#    use -- never used as a PCA input)
# ---------------------------------------------------------------------------


def load_model_table() -> tuple[pd.DataFrame, list[str]]:
    stock_vars = pd.read_csv(FACTOR_LIST)["variable"].tolist()
    keep_cols = stock_vars + ["permno", "eom", TARGET_COL]
    df = pd.read_parquet(CHARS_FILE, columns=keep_cols)
    df["eom"] = pd.to_datetime(df["eom"])
    df["target_month"] = (df["eom"] + pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    df["target_month"] = pd.to_datetime(df["target_month"]) + pd.offsets.MonthEnd(0)
    df = df[df[TARGET_COL].notna()].copy()
    return df, stock_vars


def cross_sectional_rank_transform(df: pd.DataFrame, stock_vars: list[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("eom")
    for var in stock_vars:
        med = g[var].transform("median")
        out[var] = out[var].fillna(med).fillna(0.0)
    g = out.groupby("eom")
    for var in stock_vars:
        r = g[var].rank(method="dense")
        rmax = r.groupby(out["eom"]).transform("max")
        out[var] = np.where(rmax > 0, (r / rmax) * 2 - 1, 0.0)
    return out


# ---------------------------------------------------------------------------
# 2. PCA + principal-angle subspace stability
# ---------------------------------------------------------------------------


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def top_k_loadings(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of X's covariance matrix (columns already
    standardized by construction -- rank-transformed to [-1, 1]), as a
    (n_features, k) orthonormal matrix."""
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:k]], eigvals[order]


def principal_angle_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spaces of two
    orthonormal matrices A and B (both n_features x k), via SVD of A^T B.
    All 1.0 = identical subspaces (maximally stable); values near 0 for any
    angle = the two estimates diverge in that direction (unstable)."""
    M = A.T @ B
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, -1.0, 1.0)


def run_diagnostic(df: pd.DataFrame, stock_vars: list[str]):
    test_years = range(2021, 2027)
    stability_rows = []
    variance_rows = []

    for test_year in test_years:
        val_start, _ = year_bounds(test_year - 2)
        train = df[df["target_month"] < val_start]
        if train.empty:
            continue

        months = sorted(train["target_month"].unique())
        mid = len(months) // 2
        months_a, months_b = months[:mid], months[mid:]
        if len(months_a) < 12 or len(months_b) < 12:
            continue  # need enough months per half for a stable covariance estimate

        X_a = train[train["target_month"].isin(months_a)][stock_vars].values
        X_b = train[train["target_month"].isin(months_b)][stock_vars].values

        loadings_a, eigvals_a = top_k_loadings(X_a, MAX_K)
        loadings_b, eigvals_b = top_k_loadings(X_b, MAX_K)

        total_var_a = eigvals_a[eigvals_a > 0].sum()
        cum_var_a = np.cumsum(eigvals_a[:MAX_K]) / total_var_a
        for k in range(1, MAX_K + 1):
            variance_rows.append({
                "test_year": test_year, "k": k,
                "cumulative_variance_explained_half_a": float(cum_var_a[k - 1]),
            })

        for k in range(1, MAX_K + 1):
            cosines = principal_angle_cosines(loadings_a[:, :k], loadings_b[:, :k])
            stability_rows.append({
                "test_year": test_year, "k": k,
                "mean_principal_angle_cosine": float(cosines.mean()),
                "min_principal_angle_cosine": float(cosines.min()),
                "n_train_months_half_a": len(months_a), "n_train_months_half_b": len(months_b),
            })

        print(
            f"[fold {test_year}] train months={len(months)} (split {len(months_a)}/{len(months_b)}), "
            f"k=1 stability cos={stability_rows[-MAX_K]['mean_principal_angle_cosine']:.3f}, "
            f"k={MAX_K} stability cos={stability_rows[-1]['mean_principal_angle_cosine']:.3f}, "
            f"var explained by top {MAX_K} PCs (half A)={cum_var_a[-1]*100:.1f}%"
        )

    return pd.DataFrame(stability_rows), pd.DataFrame(variance_rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} characteristics")

    print("Cross-sectional preprocessing...")
    df = cross_sectional_rank_transform(df, stock_vars)

    print("\nSplit-half PCA subspace stability, per walk-forward training fold, k=1..15...")
    stability_df, variance_df = run_diagnostic(df, stock_vars)
    stability_df.to_csv(OUTPUT / "pca_stability_by_k.csv", index=False)
    variance_df.to_csv(OUTPUT / "pca_variance_explained.csv", index=False)

    summary_by_k = stability_df.groupby("k")["mean_principal_angle_cosine"].mean()
    print("\nAverage subspace stability (mean principal-angle cosine) across all folds, by k:")
    for k, cos in summary_by_k.items():
        print(f"  k={k:2d}: {cos:.4f}")

    variance_summary_by_k = variance_df.groupby("k")["cumulative_variance_explained_half_a"].mean()
    k_for_50pct = int((variance_summary_by_k >= 0.5).idxmax()) if (variance_summary_by_k >= 0.5).any() else None
    k_for_90pct = int((variance_summary_by_k >= 0.9).idxmax()) if (variance_summary_by_k >= 0.9).any() else None

    results = {
        "max_k": MAX_K,
        "avg_stability_cosine_by_k": {int(k): float(v) for k, v in summary_by_k.items()},
        "avg_cumulative_variance_explained_by_k": {int(k): float(v) for k, v in variance_summary_by_k.items()},
        "k_needed_for_50pct_variance_avg": k_for_50pct,
        "k_needed_for_90pct_variance_within_top_15_avg": k_for_90pct,
        "n_folds": int(stability_df["test_year"].nunique()),
    }
    with open(OUTPUT / "pca_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nk needed for 50% cumulative variance (avg across folds): {k_for_50pct}")
    print(f"k needed for 90% cumulative variance within top {MAX_K} (avg across folds): {k_for_90pct}")
    print("\nFull results written to experiments/pca_diagnostic/output/pca_diagnostic_results.json")
