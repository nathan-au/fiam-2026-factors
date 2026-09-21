"""
FIAM 2026 - Quantity-conditioned prediction model (operationalizes
"Quantity, Risk, and Return" -- the Beta-Times-Quantity (BTQ) model, arXiv
2609.05162, Sep 2026).

The paper's actual concept, per its verbatim abstract (read directly -- see
docs/PAPERS.md sec 1): "quantity" (q) is a FACTOR-level construct -- "the
factor's quantity fluctuations ... induced by trading flows" -- how much
noise-trading flow a given factor's exposure has absorbed. "Sophisticated
investors should demand a higher factor premium when they have absorbed
noise trading flows of stocks with high loadings to that factor." This does
NOT reproduce the paper's actual BTQ estimator (this panel has no
trading-flow-by-investor-type data). What IS implemented now is a genuinely
FACTOR-level construction (not the first version's blanket per-stock
liquidity proxy interacted with everything):

  1. NAMED FACTORS: seven well-known priced factors, each represented by
     one characteristic already in this panel -- value (be_me), momentum
     (ret_12_1), size (market_equity), quality (qmj), profitability
     (gp_at), investment (at_gr1), betting-against-beta (betabab_1260d).
  2. FACTOR-LEVEL QUANTITY: for each named factor F and each month t,
         quantity_F,t = sum_i( |F_i,t| * turnover_126d_i,t ) / sum_i( |F_i,t| )
     -- a loading-weighted cross-sectional average of trading intensity
     (turnover, the most direct "quantity of trading" measure) across
     stocks exposed to F that month. This is ONE NUMBER PER FACTOR PER
     MONTH (not per stock), directly operationalizing "the factor's
     quantity fluctuations induced by trading flows."
  3. BTQ INTERACTION: for each named factor, F_i,t * quantity_F,t -- a
     stock's own loading on F, times F's own factor-level quantity that
     month (not the stock's own liquidity as before) -- literally "beta
     times quantity" at the factor level.
  4. Model: [147 levels + 7 BTQ interaction terms] = 154 predictors (much
     smaller than the first version's blanket 294), fit via Ridge
     (regularization still needed -- BTQ terms are correlated with their
     base factor), walk-forward, alpha tuned on validation.

**Ablation, not just a standalone score:** the whole point of testing BTQ is
whether the quantity interactions add anything. Alongside the primary
154-column model, this script also fits a 147-column levels-only Ridge (same
alpha grid, same walk-forward folds) as a controlled baseline -- same
regularization family, only the 7 BTQ terms differ -- and reports both
OOS R^2 so the interaction terms' marginal contribution is visible, not
just implied by comparing to a differently-regularized model (plain OLS).

Self-contained: data loading, rank transform, walk-forward schedule,
investability screen, beta-neutral LP, and evaluation code are ported from
ols.py / xgb.py (not imported).

Run: .venv/bin/python btq.py
Outputs (all in output/): oos_predictions_btq.csv,
    portfolio_holdings_beta_neutral_btq.csv,
    portfolio_returns_beta_neutral_btq.csv, btq_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"
OUTPUT = BASE / "output"
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"
OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")

NAMED_FACTORS = ["be_me", "ret_12_1", "market_equity", "qmj", "gp_at", "at_gr1", "betabab_1260d"]
TRADING_INTENSITY_COL = "turnover_126d"

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


def add_btq_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Factor-level quantity per docs/PAPERS.md's verbatim-abstract
    correction: one quantity_F value per factor per MONTH (loading-weighted
    average trading intensity across stocks exposed to F), not a per-stock
    proxy. Computed via groupby-apply, once, before the walk-forward split
    (it only uses each month's own cross-section -- no forward information,
    safe under FIAM.md's look-ahead rules exactly like the rank transform)."""
    out = df.copy()
    interaction_cols = []

    for factor in NAMED_FACTORS:
        abs_loading = out[factor].abs()
        weighted = abs_loading * out[TRADING_INTENSITY_COL]
        monthly_num = weighted.groupby(out["eom"]).transform("sum")
        monthly_den = abs_loading.groupby(out["eom"]).transform("sum")
        quantity_f = np.where(monthly_den > 0, monthly_num / monthly_den, 0.0)

        col = f"btq__{factor}"
        out[col] = out[factor] * quantity_f
        interaction_cols.append(col)

    return out, interaction_cols


# ---------------------------------------------------------------------------
# 2. Expanding-window walk-forward Ridge, alpha tuned on validation
# ---------------------------------------------------------------------------

ALPHA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def fit_best_ridge(X_train, y_train, X_val, y_val):
    best_model, best_alpha, best_val_mse = None, None, np.inf
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = float(np.mean((y_val - val_pred) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
            best_alpha = alpha
    return best_model, best_alpha, best_val_mse


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str], interaction_cols: list[str]):
    """Fits BOTH the primary (levels + quantity interactions) and the
    ablation (levels-only) Ridge model at each fold, same alpha grid."""
    test_years = range(2021, 2027)
    full_cols = stock_vars + interaction_cols
    preds_primary, preds_ablation, fold_params = [], [], []

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

        y_train, y_val = train[TARGET_COL].values, val[TARGET_COL].values

        # primary: levels + quantity interactions (294 cols)
        model_p, alpha_p, mse_p = fit_best_ridge(train[full_cols].values, y_train, val[full_cols].values, y_val)
        # ablation: levels only (147 cols), same alpha grid/family
        model_a, alpha_a, mse_a = fit_best_ridge(train[stock_vars].values, y_train, val[stock_vars].values, y_val)

        fold_params.append({
            "test_year": test_year,
            "primary_alpha": alpha_p, "primary_val_mse": mse_p,
            "ablation_alpha": alpha_a, "ablation_val_mse": mse_a,
        })

        fold_meta_cols = ["permno", "target_month", "ticker", "company_name", TARGET_COL,
                           "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m"]

        fold_p = test[fold_meta_cols].copy()
        fold_p["pred"] = model_p.predict(test[full_cols].values)
        fold_p["test_year"] = test_year
        preds_primary.append(fold_p)

        fold_a = test[fold_meta_cols].copy()
        fold_a["pred"] = model_a.predict(test[stock_vars].values)
        fold_a["test_year"] = test_year
        preds_ablation.append(fold_a)

        print(
            f"[fold {test_year}] train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"primary_alpha={alpha_p:g} (mse={mse_p:.6f}) vs ablation_alpha={alpha_a:g} (mse={mse_a:.6f})"
        )

    return pd.concat(preds_primary, ignore_index=True), pd.concat(preds_ablation, ignore_index=True), fold_params


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

    print(f"Building factor-level BTQ interaction terms for {len(NAMED_FACTORS)} named factors "
          f"(154 total predictors)...")
    df, interaction_cols = add_btq_interactions(df)

    print("Walk-forward expanding-window Ridge: primary (levels+BTQ) vs ablation (levels-only)...")
    preds, preds_ablation, fold_params = run_walk_forward(df, stock_vars, interaction_cols)
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds_ablation = preds_ablation[(preds_ablation["target_month"] >= OOS_START) & (preds_ablation["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / "oos_predictions_btq.csv", index=False)

    r2_primary = oos_r2(preds[TARGET_COL].values, preds["pred"].values)
    r2_ablation = oos_r2(preds_ablation[TARGET_COL].values, preds_ablation["pred"].values)
    print(f"\nOOS R^2 -- primary (levels+quantity interactions): {r2_primary * 100:.4f}%")
    print(f"OOS R^2 -- ablation (levels only, same Ridge family): {r2_ablation * 100:.4f}%")

    all_results = {
        "oos_r2_primary": float(r2_primary), "oos_r2_ablation_levels_only": float(r2_ablation),
        "n_predictor_columns_primary": len(stock_vars) + len(interaction_cols),
        "n_predictor_columns_ablation": len(stock_vars),
        "oos_prediction_rows": int(len(preds)), "fold_hyperparameters": fold_params,
    }

    print("\nBuilding portfolio: Beta-neutral (primary model)...")
    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / "portfolio_holdings_beta_neutral_btq.csv", index=False)

    print("Computing performance...")
    bn_results = compute_performance(bn_stats, tag="beta_neutral_btq")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    all_results["beta_neutral"] = bn_results

    print("\n=== Summary: Beta-neutral (BTQ / Quantity-conditioned Ridge) ===")
    for k, v in bn_results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    with open(OUTPUT / "btq_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to output/btq_results.json")
