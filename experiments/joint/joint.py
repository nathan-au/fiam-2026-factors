"""
FIAM 2026 - Portfolio-aware weighted regression (operationalizes "Machine
Learning Meets Markowitz", NBER WP 34861, Feb 2026).

The paper's argument (per docs/RESEARCH.md Part III §2): the standard two-stage
pipeline -- forecast returns with uniform weight on every observation's
error, THEN separately solve a portfolio optimizer -- is "deeply
problematic" because the optimizer only cares about prediction error in the
stocks that end up mattering to the final portfolio, not all of them
equally. `ols.py` and `xgb.py` both use exactly this two-stage architecture.

This script does NOT reproduce the paper's actual joint estimation machinery
(not available beyond the abstract/summary -- their method unifies the
return-generation and portfolio-optimization steps directly, likely via some
form of decision-focused/implicit-differentiation training). What it DOES
implement is a practical, honestly-labeled approximation of the same
underlying idea: **weight each training observation's squared-error loss by
how much it matters to a decile-based long/short portfolio** -- i.e. give
more weight to stocks whose realized target return is cross-sectionally
extreme (the ones most likely to end up in the traded top/bottom decile) and
less weight to stocks near the cross-sectional median (whose prediction
accuracy barely affects which names get selected). This is a portfolio-aware
weighted least squares, not the paper's literal joint-optimization method --
call it "decision weighting" rather than "joint optimization."

Weight scheme: w_i = |rank_target_i|^p, where rank_target_i is the
cross-sectional rank of the REALIZED target return within its month, rescaled
to [-1, 1] (same convention as the rank-transformed predictors), and p is a
tuning exponent selected on the validation fold. Using the REALIZED target's
own rank as the weight is legitimate here (not a leakage route) because it
is computed from training-fold data only, to weight training-fold loss -- it
never touches predictors, and no information about the test period's
targets is used.

FIX (see experiments/joint/README.md for the full before/after): the first version of this
script selected p by UNIFORM validation MSE, which structurally always
prefers p=0 (unweighted OLS minimizes unweighted MSE by construction) --
every fold degenerated to plain OLS, never actually testing the weighting.
This version selects p by running the REAL beta-neutral LP on the
validation period's predictions for each candidate p and picking whichever
p produces the best validation-period PORTFOLIO SHARPE RATIO
(validation_portfolio_sharpe()) -- a portfolio-level criterion for a
portfolio-level design choice, directly matching the paper's own argument
that pointwise error is the wrong thing to optimize when the ultimate
target is portfolio performance.

Self-contained: data loading, rank transform, walk-forward schedule,
investability screen, beta-neutral LP, and evaluation code are ported from
ols.py / xgb.py (not imported).

Run: .venv/bin/python experiments/joint/joint.py
Outputs (all in output/): oos_predictions_joint.csv,
    portfolio_holdings_beta_neutral_joint.csv,
    portfolio_returns_beta_neutral_joint.csv, joint_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression

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


def add_target_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of the REALIZED target, per characteristic month,
    rescaled to [-1, 1] -- used only as a training-loss weight (see module
    docstring), never as a predictor."""
    out = df.copy()
    g = out.groupby("eom")
    r = g[TARGET_COL].rank(method="dense")
    rmax = r.groupby(out["eom"]).transform("max")
    out["target_rank"] = np.where(rmax > 0, (r / rmax) * 2 - 1, 0.0)
    return out


# ---------------------------------------------------------------------------
# 2. Expanding-window walk-forward portfolio-aware weighted OLS
# ---------------------------------------------------------------------------

EXPONENT_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]  # 0.0 = plain OLS (uniform weight)
MIN_WEIGHT = 0.05  # floor so no training row gets literally zero weight


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def validation_portfolio_sharpe(val_df: pd.DataFrame, val_pred: np.ndarray) -> float:
    """Actually run the beta-neutral LP on the validation period's
    predictions and compute the resulting portfolio's (raw, not
    benchmark-relative) monthly Sharpe ratio. This is the fix for the
    degenerate-p=0 problem documented in this script's earlier run (see
    experiments/joint/README.md): the paper's whole argument is that pointwise error
    should NOT be the model-selection criterion for a portfolio-aware
    model -- so the criterion here is a portfolio-level statistic, computed
    from the SAME LP construction used for final OOS evaluation, not MSE."""
    scored = val_df.copy()
    scored["pred"] = val_pred
    try:
        _, stats_df = build_beta_neutral_portfolio(scored)
    except ValueError:
        return -np.inf  # LP infeasible for every month in this candidate -- worst possible score
    if len(stats_df) < 6 or stats_df["port_excess_ret"].std() == 0:
        return -np.inf
    return float(np.sqrt(12) * stats_df["port_excess_ret"].mean() / stats_df["port_excess_ret"].std())


def fit_best_weighted_ols(X_train, y_train, w_train, val_df, stock_vars):
    X_val = val_df[stock_vars].values
    best_model, best_p, best_val_sharpe = None, None, -np.inf
    candidate_sharpes = {}
    for p in EXPONENT_GRID:
        sample_weight = np.maximum(np.abs(w_train) ** p, MIN_WEIGHT) if p > 0 else None
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        val_pred = model.predict(X_val)
        val_sharpe = validation_portfolio_sharpe(val_df, val_pred)
        candidate_sharpes[p] = val_sharpe
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_model = model
            best_p = p
    return best_model, best_p, best_val_sharpe, candidate_sharpes


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str]):
    test_years = range(2021, 2027)
    preds, fold_params = [], []

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

        X_train, y_train = train[stock_vars].values, train[TARGET_COL].values
        w_train = train["target_rank"].values

        model, p, val_sharpe, candidate_sharpes = fit_best_weighted_ols(X_train, y_train, w_train, val, stock_vars)
        fold_params.append({
            "test_year": test_year, "weight_exponent_p": p,
            "val_portfolio_sharpe": val_sharpe,
            "candidate_val_sharpes": {str(k): v for k, v in candidate_sharpes.items()},
        })

        X_test = test[stock_vars].values
        y_pred = model.predict(X_test)

        fold = test[
            ["permno", "target_month", "ticker", "company_name", TARGET_COL,
             "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m"]
        ].copy()
        fold["pred"] = y_pred
        fold["test_year"] = test_year
        fold["n_train"] = len(train)
        fold["n_val"] = len(val)
        preds.append(fold)

        print(
            f"[fold {test_year}] train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"selected p={p} (val portfolio Sharpe={val_sharpe:.4f}) | all candidates: "
            + ", ".join(f"p={k}:{v:.3f}" for k, v in candidate_sharpes.items())
        )

    return pd.concat(preds, ignore_index=True), fold_params


def run_walk_forward_forced(df: pd.DataFrame, stock_vars: list[str], p: float):
    """Same walk-forward loop, but skip validation selection and force the
    weighting exponent to a fixed value -- run when the validation-selected
    result degenerates to p=0 every fold (see module docstring addendum),
    to show what the weighting scheme actually does when it's applied,
    rather than leaving the "joint" idea untested in the OOS numbers."""
    test_years = range(2021, 2027)
    preds = []
    for test_year in test_years:
        test_start, test_end = year_bounds(test_year)
        test_end = min(test_end, OOS_END)
        val_start, _ = year_bounds(test_year - 2)

        train_mask = df["target_month"] < val_start
        test_mask = (df["target_month"] >= test_start) & (df["target_month"] <= test_end)
        train, test = df[train_mask], df[test_mask]
        if train.empty or test.empty:
            continue

        X_train, y_train = train[stock_vars].values, train[TARGET_COL].values
        w_train = train["target_rank"].values
        sample_weight = np.maximum(np.abs(w_train) ** p, MIN_WEIGHT)
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=sample_weight)

        X_test = test[stock_vars].values
        y_pred = model.predict(X_test)
        fold = test[
            ["permno", "target_month", "ticker", "company_name", TARGET_COL,
             "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m"]
        ].copy()
        fold["pred"] = y_pred
        preds.append(fold)
    return pd.concat(preds, ignore_index=True)


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

if __name__ == "__main__":
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors")

    print("Cross-sectional preprocessing...")
    df = cross_sectional_rank_transform(df, stock_vars)
    df = add_target_rank(df)

    print("Walk-forward expanding-window portfolio-aware weighted OLS (p tuned on validation)...")
    preds, fold_params = run_walk_forward(df, stock_vars)
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / "oos_predictions_joint.csv", index=False)

    r2 = oos_r2(preds[TARGET_COL].values, preds["pred"].values)
    print(f"\nOOS R^2 (full 2021-01 to 2026-08 sample): {r2 * 100:.4f}%")

    all_results = {"oos_r2": float(r2), "n_predictor_columns": len(stock_vars),
                    "oos_prediction_rows": int(len(preds)), "fold_hyperparameters": fold_params}

    print("\nBuilding portfolio: Beta-neutral...")
    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / "portfolio_holdings_beta_neutral_joint.csv", index=False)

    print("Computing performance...")
    bn_results = compute_performance(bn_stats, tag="beta_neutral_joint")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    all_results["beta_neutral"] = bn_results

    print("\n=== Summary: Beta-neutral (Portfolio-Aware Weighted OLS) ===")
    for k, v in bn_results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    selected_ps = [p["weight_exponent_p"] for p in fold_params]
    print(f"\nSelected p per fold (portfolio-Sharpe validation): {selected_ps}")
    print("Running a forced p=2 supplementary pass regardless, as a fixed reference point")
    print("independent of what validation happened to pick this run (see experiments/joint/README.md).")
    forced_preds = run_walk_forward_forced(df, stock_vars, p=2.0)
    forced_preds = forced_preds[(forced_preds["target_month"] >= OOS_START) & (forced_preds["target_month"] <= OOS_END)]
    forced_r2 = oos_r2(forced_preds[TARGET_COL].values, forced_preds["pred"].values)
    forced_holdings, forced_stats = build_beta_neutral_portfolio(forced_preds)
    forced_results = compute_performance(forced_stats, tag="beta_neutral_joint_forced_p2")
    all_results["forced_p2_supplementary"] = {
        "oos_r2": float(forced_r2),
        "information_ratio": forced_results["information_ratio"],
        "sharpe_ratio": forced_results["sharpe_ratio"],
        "annualized_return_geo_cagr": forced_results["annualized_return_geo_cagr"],
        "beta": forced_results["beta"],
        "beta_tstat": forced_results["beta_tstat"],
        "hit_rate_active": forced_results["hit_rate_active"],
    }
    print(f"  forced p=2 OOS R^2: {forced_r2*100:.4f}%  IR: {forced_results['information_ratio']:.4f}  "
          f"Sharpe: {forced_results['sharpe_ratio']:.4f}  CAGR: {forced_results['annualized_return_geo_cagr']*100:.2f}%")

    with open(OUTPUT / "joint_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to experiments/joint/output/joint_results.json")
