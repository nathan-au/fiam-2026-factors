"""
FIAM 2026 - Sparse-in-expanded-space prediction model (operationalizes "The
Virtue of Sparsity in Complexity", arXiv 2604.17166, Apr 2026).

The paper's actual method, per its verbatim abstract (read directly -- see
docs/PAPERS.md sec 1): NONLINEAR FEATURE EXPANSIONS combined with BASIS
PURSUIT, benchmarked against "ridgeless" (near-zero-regularization)
high-dimensional ridge -- their claim is that sparsity EMERGES FROM
expanding the feature space first, not from penalizing the original small
feature set directly. This version implements that two-step design:

  1. NONLINEAR EXPANSION: random Fourier features (RFF, Rahimi & Recht 2007
     -- the standard random-feature construction used throughout this
     literature, including Kelly-Malamud-Zhou's own "Virtue of Complexity").
     The 147 rank-transformed characteristics are projected through a FIXED
     random Gaussian map into P=600 nonlinear features:
         Z = sqrt(2/P) * cos(X @ Omega + b),  Omega ~ N(0,1)^(147xP), b ~ U(0,2pi)^P
     Omega/b are drawn once (fixed seed) and reused across every fold, so
     all folds share the same feature map. Z is then standardized to unit
     variance (fit on training data only, applied to val/test) -- RFF
     output is tiny in scale (~+-0.058 for P=600), and fitting Lasso
     directly on that scale made even the smallest alpha in the grid
     collapse every coefficient to zero (caught during a first run: 0/600
     nonzero at every fold -- a real bug, not the paper's finding).
  2. SPARSE RECOVERY IN THAT SPACE: LASSO on the P=600 expanded features.
     LASSO is not an approximation of basis pursuit here -- for a noisy
     regression target, L1-penalized least squares (LASSO) IS "basis
     pursuit denoising" (Chen, Donoho & Saunders 1998), the textbook noisy
     generalization of pure basis pursuit. This is a legitimate match to
     the paper's stated method, not an analogy.
  3. RIDGELESS COMPARATOR, same space: Ridge with a near-zero alpha (1e-6)
     on the SAME 600 RFF features -- the paper's actual "dense" benchmark,
     not plain OLS on the original 147 characteristics as the first version
     of this script used.

Honest deviation from the paper: their "beyond a critical complexity
threshold" regime is typically reached with P approaching or exceeding the
estimation sample size n. This project's walk-forward pools many months
into one training fold (n up to ~400K rows), so reaching P > n is
computationally infeasible; P=600 is a tractable nonlinear expansion that
tests the qualitative comparison (sparse-in-expanded-space vs.
ridgeless-in-expanded-space) without reaching their literal
overparameterized-beyond-n regime.

Self-contained: data loading, rank transform, walk-forward schedule,
investability screen, beta-neutral LP, and evaluation code are ported from
ols.py / xgb.py (not imported).

Run: .venv/bin/python experiments/sparse/sparse.py
Outputs (all in output/): oos_predictions_sparse.csv,
    portfolio_holdings_beta_neutral_sparse.csv,
    portfolio_returns_beta_neutral_sparse.csv, sparse_rff_coefficient_norms.csv,
    sparse_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from sklearn.linear_model import Lasso, Ridge

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent  # experiments/<name>/
BASE = HERE.parents[1]  # project root: fiam/ data and cache/ are shared
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"
OUTPUT = HERE / "output"
CACHE.mkdir(exist_ok=True)
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
    keep_cols = stock_vars + ["permno", "eom", "ticker", "company_name", "me", TARGET_COL]
    df = pd.read_parquet(CHARS_FILE, columns=keep_cols)
    df["eom"] = pd.to_datetime(df["eom"])
    df["raw_prc"] = df["prc"]
    df["raw_dolvol_126d"] = df["dolvol_126d"]
    df["raw_me"] = df["me"]
    df["raw_beta_60m"] = df["beta_60m"]
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
# 2. Random Fourier Feature expansion + sparse (LASSO) vs. ridgeless (Ridge)
#    walk-forward comparison, per the paper's actual two-step design
# ---------------------------------------------------------------------------

N_RFF = 600
RFF_SEED = 1234

LASSO_ALPHA_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
RIDGELESS_ALPHA = 1e-6  # near-zero regularization = "dense/ridgeless" comparator


def standardize(Z_train, Z_val, Z_test):
    """RFF features are bounded ~[-sqrt(2/P), sqrt(2/P)] (tiny scale, e.g.
    +-0.058 for P=600) -- fitting Lasso directly on that scale makes even
    the grid's smallest alpha collapse every coefficient to zero (verified:
    a first run hit 0/600 nonzero at every fold). Standardizing to unit
    variance (fit on train only, applied to val/test) is the standard fix
    and keeps alpha values comparable to typical usage."""
    mean = Z_train.mean(axis=0)
    std = Z_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (Z_train - mean) / std, (Z_val - mean) / std, (Z_test - mean) / std


def build_rff_map(n_features: int, n_rff: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed random Gaussian map (Omega, b) for random Fourier features,
    drawn once and reused across every fold -- Rahimi & Recht (2007),
    the standard random-feature construction in this literature."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(size=(n_features, n_rff))
    b = rng.uniform(0, 2 * np.pi, size=n_rff)
    return omega, b


def apply_rff(X: np.ndarray, omega: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_rff = omega.shape[1]
    return np.sqrt(2.0 / n_rff) * np.cos(X @ omega + b)


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def fit_best_lasso(X_train, y_train, X_val, y_val):
    best_model, best_alpha, best_val_mse = None, None, np.inf
    for alpha in LASSO_ALPHA_GRID:
        model = Lasso(alpha=alpha, max_iter=5000, random_state=42)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = float(np.mean((y_val - val_pred) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
            best_alpha = alpha
    return best_model, best_alpha, best_val_mse


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str]):
    test_years = range(2021, 2027)
    omega, b = build_rff_map(len(stock_vars), N_RFF, RFF_SEED)
    print(f"RFF map: {len(stock_vars)} characteristics -> {N_RFF} nonlinear random features (fixed seed={RFF_SEED})")

    preds_sparse, preds_ridgeless, fold_coef_norms, fold_params = [], [], [], []

    for test_year in test_years:
        test_start, test_end = year_bounds(test_year)
        test_end = min(test_end, OOS_END)
        val_start, _ = year_bounds(test_year - 2)
        _, val_end = year_bounds(test_year - 1)

        train_mask = df["target_month"] < val_start
        val_mask = (df["target_month"] >= val_start) & (df["target_month"] <= val_end)
        test_mask = (df["target_month"] >= test_start) & (df["target_month"] <= test_end)

        train, val, test = df[train_mask], df[val_mask], df[test_mask]
        if train.empty or val.empty or test.empty:
            continue

        Z_train = apply_rff(train[stock_vars].values, omega, b)
        Z_val = apply_rff(val[stock_vars].values, omega, b)
        Z_test = apply_rff(test[stock_vars].values, omega, b)
        Z_train, Z_val, Z_test = standardize(Z_train, Z_val, Z_test)
        y_train, y_val = train[TARGET_COL].values, val[TARGET_COL].values

        # sparse (basis-pursuit-denoising / LASSO) in RFF space, alpha tuned on validation
        sparse_model, alpha, val_mse = fit_best_lasso(Z_train, y_train, Z_val, y_val)
        n_nonzero = int(np.sum(np.abs(sparse_model.coef_) > 1e-10))

        # ridgeless (near-zero-alpha Ridge) in the SAME RFF space -- the paper's dense comparator
        ridgeless_model = Ridge(alpha=RIDGELESS_ALPHA)
        ridgeless_model.fit(Z_train, y_train)
        ridgeless_val_mse = float(np.mean((y_val - ridgeless_model.predict(Z_val)) ** 2))

        fold_params.append({
            "test_year": test_year, "sparse_alpha": alpha, "sparse_val_mse": val_mse,
            "n_nonzero_of_rff": n_nonzero, "n_total_rff": N_RFF,
            "ridgeless_val_mse": ridgeless_val_mse,
        })
        fold_coef_norms.append({
            "test_year": test_year,
            "sparse_coef_l1_norm": float(np.abs(sparse_model.coef_).sum()),
            "ridgeless_coef_l1_norm": float(np.abs(ridgeless_model.coef_).sum()),
            "sparse_coef_l2_norm": float(np.linalg.norm(sparse_model.coef_)),
            "ridgeless_coef_l2_norm": float(np.linalg.norm(ridgeless_model.coef_)),
        })

        fold_meta_cols = ["permno", "target_month", "ticker", "company_name", TARGET_COL,
                           "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m"]

        fold_sp = test[fold_meta_cols].copy()
        fold_sp["pred"] = sparse_model.predict(Z_test)
        fold_sp["test_year"] = test_year
        preds_sparse.append(fold_sp)

        fold_rl = test[fold_meta_cols].copy()
        fold_rl["pred"] = ridgeless_model.predict(Z_test)
        fold_rl["test_year"] = test_year
        preds_ridgeless.append(fold_rl)

        print(
            f"[fold {test_year}] train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"sparse: alpha={alpha:g} nonzero={n_nonzero}/{N_RFF} val_mse={val_mse:.6f} | "
            f"ridgeless: val_mse={ridgeless_val_mse:.6f}"
        )

    return (
        pd.concat(preds_sparse, ignore_index=True),
        pd.concat(preds_ridgeless, ignore_index=True),
        pd.DataFrame(fold_coef_norms),
        fold_params,
    )


# ---------------------------------------------------------------------------
# 3. Evaluation, portfolio construction (ported from ols.py / xgb.py, unchanged)
# ---------------------------------------------------------------------------


def oos_r2(actual, predicted):
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum(actual**2)


MIN_DOLLAR_VOLUME = 10_000_000.0
MAX_WEIGHT = 0.01


def build_beta_neutral_portfolio(preds):
    holdings, monthly_stats = [], []
    for month, grp in preds.groupby("target_month"):
        grp = grp[grp["raw_dolvol_126d"] >= MIN_DOLLAR_VOLUME]
        grp = grp.dropna(subset=["raw_beta_60m"]).reset_index(drop=True)
        n = len(grp)
        if n < 2 * int(1.0 / MAX_WEIGHT):
            continue
        pred = grp["pred"].values
        beta = grp["raw_beta_60m"].values
        c = np.concatenate([-pred, pred])
        A_eq = np.array([
            np.concatenate([np.ones(n), -np.ones(n)]),
            np.concatenate([beta, -beta]),
            np.concatenate([np.ones(n), np.ones(n)]),
        ])
        b_eq = [0.0, 0.0, 2.0]
        bounds = [(0.0, MAX_WEIGHT)] * (2 * n)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            print(f"  [warn] LP infeasible for {month.date()}, skipping month")
            continue
        w = res.x[:n] - res.x[n:]
        keep = np.abs(w) > 1e-8
        both = grp.loc[keep].copy()
        both["weight"] = w[keep]
        both["target_month"] = month
        holdings.append(both[["target_month", "permno", "ticker", "company_name", "weight", TARGET_COL, "pred", "raw_beta_60m"]])
        port_ret = (both["weight"] * both[TARGET_COL]).sum()
        long_mask = both["weight"] > 0
        long_ret = (both.loc[long_mask, "weight"] * both.loc[long_mask, TARGET_COL]).sum()
        short_ret = (both.loc[~long_mask, "weight"] * both.loc[~long_mask, TARGET_COL]).sum()
        gross = both["weight"].abs().sum()
        net = both["weight"].sum()
        beta_exposure = (both["weight"] * both["raw_beta_60m"]).sum()
        monthly_stats.append({
            "target_month": month, "port_excess_ret": port_ret, "long_leg_ret": long_ret,
            "short_leg_ret": short_ret, "gross_exposure": gross, "net_exposure": net,
            "beta_exposure": beta_exposure, "n_long": int((both["weight"] > 0).sum()),
            "n_short": int((both["weight"] < 0).sum()), "n_positions": int(keep.sum()),
        })
    holdings_df = pd.concat(holdings, ignore_index=True)
    stats_df = pd.DataFrame(monthly_stats).sort_values("target_month").reset_index(drop=True)
    return holdings_df, stats_df


def load_fred_series():
    tb3ms = pd.read_csv(CACHE / "TB3MS.csv", parse_dates=["observation_date"])
    tb3ms = tb3ms.rename(columns={"observation_date": "month", "TB3MS": "tb3ms_annual_pct"})
    tb3ms["month"] = tb3ms["month"].values.astype("datetime64[M]")
    tb3ms["month"] = pd.to_datetime(tb3ms["month"]) + pd.offsets.MonthEnd(0)
    tb3ms["rf_monthly"] = tb3ms["tb3ms_annual_pct"] / 100.0 / 12.0
    sp = pd.read_csv(CACHE / "SP500.csv", parse_dates=["observation_date"])
    sp = sp.rename(columns={"observation_date": "date", "SP500": "close"})
    sp = sp.dropna(subset=["close"])
    sp["month"] = sp["date"].values.astype("datetime64[M]")
    sp["month"] = pd.to_datetime(sp["month"]) + pd.offsets.MonthEnd(0)
    sp_monthly = sp.sort_values("date").groupby("month")["close"].last().reset_index()
    sp_monthly["sp500_ret"] = sp_monthly["close"].pct_change()
    return tb3ms[["month", "tb3ms_annual_pct", "rf_monthly"]], sp_monthly[["month", "sp500_ret"]]


def compute_turnover_and_concentration(holdings_df):
    piv = holdings_df.pivot_table(index="target_month", columns="permno", values="weight", fill_value=0.0).sort_index()
    diffs = piv.diff().abs().sum(axis=1, min_count=1) / 2.0
    turnover = (diffs.iloc[1:] / 2.0).rename("turnover")
    abs_w = holdings_df["weight"].abs()
    top10_share_by_month = holdings_df.groupby("target_month")["weight"].apply(lambda w: w.abs().nlargest(10).sum() / 2.0)
    return {
        "avg_monthly_turnover": float(turnover.mean()), "min_monthly_turnover": float(turnover.min()),
        "max_monthly_turnover": float(turnover.max()), "avg_position_weight_abs": float(abs_w.mean()),
        "max_position_weight_abs": float(abs_w.max()), "avg_top10_share_of_gross": float(top10_share_by_month.mean()),
    }


def compute_short_book_characteristics(holdings_df):
    chars = pd.read_parquet(CHARS_FILE, columns=["permno", "eom", "me", "dolvol_126d"])
    chars["eom"] = pd.to_datetime(chars["eom"])
    h = holdings_df.copy()
    h["target_month"] = pd.to_datetime(h["target_month"])
    h["char_month"] = (h["target_month"] - pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    h["char_month"] = pd.to_datetime(h["char_month"]) + pd.offsets.MonthEnd(0)
    m = h.merge(chars, left_on=["permno", "char_month"], right_on=["permno", "eom"], how="left")
    short = m[m["weight"] < 0]
    return {
        "short_book_avg_market_cap_musd": float(short["me"].mean()), "short_book_median_market_cap_musd": float(short["me"].median()),
        "short_book_avg_dollar_volume_usd": float(short["dolvol_126d"].mean()),
        "short_book_share_below_2000m_cap": float((short["me"] < 2000).mean()),
        "short_book_share_below_1000m_cap": float((short["me"] < 1000).mean()),
    }


def drawdown_series(rets):
    wealth = (1 + rets).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def compute_drawdown_stats(rets, dates, cagr):
    rets = rets.reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    dd = drawdown_series(rets)
    trough_i = int(dd.idxmin())
    max_dd = float(dd.iloc[trough_i])
    if max_dd < 0:
        at_high = dd.iloc[: trough_i + 1] >= 0
        peak_i = int(at_high[at_high].index.max()) if at_high.any() else -1
        recovered = dd.iloc[trough_i + 1:] >= 0
        recovery_i = int(recovered[recovered].index.min()) if recovered.any() else None
    else:
        peak_i, recovery_i = trough_i, trough_i

    def date_at(i):
        if i is None:
            return None
        if i < 0:
            return str((dates.iloc[0] - pd.offsets.MonthEnd(1)).date())
        return str(dates.iloc[i].date())

    underwater = (dd < 0).astype(int)
    runs = underwater.groupby((underwater == 0).cumsum()).sum()
    longest_underwater = int(runs.max()) if len(runs) else 0
    return {
        "max_drawdown": max_dd, "max_drawdown_peak_date": date_at(peak_i),
        "max_drawdown_trough_date": date_at(trough_i), "max_drawdown_recovery_date": date_at(recovery_i),
        "max_drawdown_peak_to_trough_months": int(trough_i - peak_i),
        "max_drawdown_recovery_months": None if recovery_i is None else int(recovery_i - trough_i),
        "longest_underwater_months": longest_underwater, "current_drawdown": float(dd.iloc[-1]),
        "avg_drawdown_when_underwater": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else None,
    }


def compute_performance(stats_df, tag):
    tb3ms, sp500 = load_fred_series()
    perf = stats_df.merge(tb3ms, left_on="target_month", right_on="month", how="left")
    perf = perf.merge(sp500, left_on="target_month", right_on="month", how="left", suffixes=("", "_sp"))
    perf["benchmark_monthly"] = perf["rf_monthly"] + 0.04 / 12.0
    perf["active_ret"] = perf["port_excess_ret"] - 0.04 / 12.0
    ir = np.sqrt(12) * perf["active_ret"].mean() / perf["active_ret"].std()
    sharpe = np.sqrt(12) * perf["port_excess_ret"].mean() / perf["port_excess_ret"].std()
    reg_df = perf.dropna(subset=["sp500_ret", "rf_monthly"]).copy()
    y = perf.loc[reg_df.index, "port_excess_ret"].values
    x = (reg_df["sp500_ret"] - reg_df["rf_monthly"]).values
    X = sm.add_constant(x)
    ols_res = sm.OLS(y, X).fit()
    alpha_monthly, beta = ols_res.params
    alpha_se, beta_se = ols_res.bse
    alpha_t, beta_t = ols_res.tvalues
    cum_ret = (1 + perf["port_excess_ret"]).prod() - 1
    n_months = len(perf)
    ann_ret_arith = perf["port_excess_ret"].mean() * 12
    ann_ret_geo = (1 + perf["port_excess_ret"]).prod() ** (12 / n_months) - 1
    hit_rate = (perf["active_ret"] > 0).mean()
    best_month = perf.loc[perf["port_excess_ret"].idxmax()]
    worst_month = perf.loc[perf["port_excess_ret"].idxmin()]

    def calendar_year_of(col):
        d = perf.dropna(subset=[col]).assign(year=lambda x: x["target_month"].dt.year)
        return d.groupby("year").apply(lambda g: (1 + g[col]).prod() - 1, include_groups=False).to_dict()

    calendar_year = calendar_year_of("port_excess_ret")
    calendar_year_benchmark = calendar_year_of("benchmark_monthly")
    calendar_year_sp500 = calendar_year_of("sp500_ret")
    long_leg_cagr = (1 + perf["long_leg_ret"]).prod() ** (12 / n_months) - 1
    short_leg_cagr = (1 + perf["short_leg_ret"]).prod() ** (12 / n_months) - 1
    perf["drawdown"] = drawdown_series(perf["port_excess_ret"])
    perf["sp500_drawdown"] = drawdown_series(perf["sp500_ret"].fillna(0.0))
    sp500_cagr = (1 + perf["sp500_ret"].fillna(0.0)).prod() ** (12 / n_months) - 1
    dd_stats = compute_drawdown_stats(perf["port_excess_ret"], perf["target_month"], ann_ret_geo)
    sp500_dd_stats = compute_drawdown_stats(perf["sp500_ret"].fillna(0.0), perf["target_month"], sp500_cagr)

    results = {
        "n_months": int(n_months), "avg_monthly_return": float(perf["port_excess_ret"].mean()),
        "annualized_return_arith": float(ann_ret_arith), "annualized_return_geo_cagr": float(ann_ret_geo),
        "cumulative_return": float(cum_ret), "hit_rate_active": float(hit_rate),
        "information_ratio": float(ir), "sharpe_ratio": float(sharpe),
        "alpha_monthly": float(alpha_monthly), "alpha_annualized": float(alpha_monthly * 12),
        "alpha_tstat": float(alpha_t), "alpha_se_monthly": float(alpha_se),
        "beta": float(beta), "beta_se": float(beta_se), "beta_tstat": float(beta_t),
        "corr_with_sp500": float(perf["port_excess_ret"].corr(perf["sp500_ret"])),
        "avg_gross_exposure": float(stats_df["gross_exposure"].mean()),
        "max_gross_exposure": float(stats_df["gross_exposure"].max()),
        "avg_net_exposure": float(stats_df["net_exposure"].mean()),
        "min_net_exposure": float(stats_df["net_exposure"].min()),
        "max_net_exposure": float(stats_df["net_exposure"].max()),
        "avg_n_positions": float(stats_df["n_positions"].mean()),
        "avg_n_long": float(stats_df["n_long"].mean()), "avg_n_short": float(stats_df["n_short"].mean()),
        "best_month": {"date": str(best_month["target_month"].date()), "ret": float(best_month["port_excess_ret"])},
        "worst_month": {"date": str(worst_month["target_month"].date()), "ret": float(worst_month["port_excess_ret"])},
        "calendar_year_returns": {int(k): float(v) for k, v in calendar_year.items()},
        "calendar_year_returns_benchmark": {int(k): float(v) for k, v in calendar_year_benchmark.items()},
        "calendar_year_returns_sp500": {int(k): float(v) for k, v in calendar_year_sp500.items()},
        "long_leg_avg_monthly_ret": float(perf["long_leg_ret"].mean()), "long_leg_cagr": float(long_leg_cagr),
        "short_leg_avg_monthly_ret": float(perf["short_leg_ret"].mean()), "short_leg_cagr": float(short_leg_cagr),
        **dd_stats, "sp500_drawdown": sp500_dd_stats,
    }
    perf.to_csv(OUTPUT / f"portfolio_returns_{tag}.csv", index=False)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def evaluate_arm(preds: pd.DataFrame, tag: str) -> dict:
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / f"oos_predictions_{tag}.csv", index=False)
    r2 = oos_r2(preds[TARGET_COL].values, preds["pred"].values)

    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / f"portfolio_holdings_beta_neutral_{tag}.csv", index=False)
    bn_results = compute_performance(bn_stats, tag=f"beta_neutral_{tag}")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    return {"oos_r2": float(r2), "oos_prediction_rows": int(len(preds)), "beta_neutral": bn_results}


if __name__ == "__main__":
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors")

    print("Cross-sectional preprocessing...")
    df = cross_sectional_rank_transform(df, stock_vars)

    print("Walk-forward: RFF expansion (147->600) + sparse LASSO vs. ridgeless Ridge, same space...")
    preds_sparse, preds_ridgeless, coef_norms_df, fold_params = run_walk_forward(df, stock_vars)
    coef_norms_df.to_csv(OUTPUT / "sparse_rff_coefficient_norms.csv", index=False)

    avg_nonzero_frac = np.mean([p["n_nonzero_of_rff"] / p["n_total_rff"] for p in fold_params])
    print(f"\nAverage sparsity (LASSO in RFF space) across folds: {avg_nonzero_frac*100:.1f}% of {N_RFF} random features nonzero")

    print("\nEvaluating sparse (primary) arm...")
    sparse_results = evaluate_arm(preds_sparse, "sparse")
    print(f"  Sparse OOS R^2: {sparse_results['oos_r2']*100:.4f}%  "
          f"IR: {sparse_results['beta_neutral']['information_ratio']:.4f}  "
          f"Sharpe: {sparse_results['beta_neutral']['sharpe_ratio']:.4f}")

    print("\nEvaluating ridgeless (dense comparator) arm...")
    ridgeless_results = evaluate_arm(preds_ridgeless, "sparse_ridgeless")
    print(f"  Ridgeless OOS R^2: {ridgeless_results['oos_r2']*100:.4f}%  "
          f"IR: {ridgeless_results['beta_neutral']['information_ratio']:.4f}  "
          f"Sharpe: {ridgeless_results['beta_neutral']['sharpe_ratio']:.4f}")

    all_results = {
        "n_original_characteristics": len(stock_vars), "n_rff": N_RFF, "rff_seed": RFF_SEED,
        "fold_hyperparameters": fold_params, "avg_nonzero_fraction_of_rff": float(avg_nonzero_frac),
        "sparse": sparse_results, "ridgeless": ridgeless_results,
    }

    print("\n=== Summary: sparse (LASSO-in-RFF-space) vs. ridgeless (Ridge-in-RFF-space) ===")
    print(f"  Sparse:    R2={sparse_results['oos_r2']*100:+.4f}%  IR={sparse_results['beta_neutral']['information_ratio']:.4f}  "
          f"Sharpe={sparse_results['beta_neutral']['sharpe_ratio']:.4f}  "
          f"CAGR={sparse_results['beta_neutral']['annualized_return_geo_cagr']*100:.2f}%")
    print(f"  Ridgeless: R2={ridgeless_results['oos_r2']*100:+.4f}%  IR={ridgeless_results['beta_neutral']['information_ratio']:.4f}  "
          f"Sharpe={ridgeless_results['beta_neutral']['sharpe_ratio']:.4f}  "
          f"CAGR={ridgeless_results['beta_neutral']['annualized_return_geo_cagr']*100:.2f}%")
    print(f"  Paper's claim: sparse should DOMINATE ridgeless in Sharpe. "
          f"Observed: {'CONFIRMED' if sparse_results['beta_neutral']['sharpe_ratio'] > ridgeless_results['beta_neutral']['sharpe_ratio'] else 'NOT CONFIRMED'} in this run.")

    with open(OUTPUT / "sparse_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to experiments/sparse/output/sparse_results.json")
