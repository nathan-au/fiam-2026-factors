"""
FIAM 2026 - Interpretable polynomial model, loosely inspired by
"AlphaPortfolio: Goal-Oriented Investment Management Through Deep
Reinforcement Learning" (NBER WP 35195, May 2026, Cong-Tang-Wang).

CITATION-ACCURACY CORRECTION (see experiments/poly/README.md for the full story): this
script was originally described as operationalizing a "polynomial-network
layer for economic distillation" allegedly part of NBER WP 35195. Reading
the actual paper (not a search summary) showed that framing is wrong: WP
35195's core architecture is a Transformer encoder + cross-asset attention
network trained via RL; "polynomial" in that paper refers to a POST-HOC
sensitivity-analysis tool applied after training that model, not a
standalone predictive architecture. The "polynomial network"/"economic
distillation" language actually belongs to a different, older, pre-2026
working paper by an overlapping author set -- citing it here contradicted
this project's own 2026-only filter. No Transformer, attention, or RL is
built here, and this script should not be read as testing WP 35195's
actual method.

What IS implemented, honestly stated: an explicit **degree-2 polynomial
regression on a curated small subset of characteristics**, directly and
exactly attributable by construction (every term IS a named characteristic,
a named characteristic squared, or a named pair's product -- no black box
to distill afterward) -- inspired by the general shape of "use polynomial
structure for attribution," not a reproduction of any specific paper's
architecture.

Feature selection (the "curation" step): rather than use all 147
characteristics (which would make a degree-2 expansion ~10,000+ columns),
this script selects the **top 20 characteristics by |IC t-stat|**, using the
same univariate Information-Coefficient methodology as univariate.py --
but NOT by reading univariate.py's saved experiments/univariate/output/univariate_results.csv
directly. That file's IC was computed over the 2021-01..2026-08 OOS window,
the exact window this script is evaluated on, so using it to pick features
here would be look-ahead bias in the selection step (caught during
development -- see experiments/poly/README.md). Instead, feature selection is redone from
scratch INSIDE each walk-forward fold, using only that fold's TRAINING data,
so the selected-20 for the 2021 test fold, say, only ever saw characteristic
months through 2018-12. A degree-2 polynomial expansion of the selected 20
(20 levels + 20 squares + 190 pairwise interactions + 1 intercept = 231
terms) is fit via Ridge on the same fold's training data, alpha tuned on
validation. After each fold, the largest-magnitude coefficients are reported
by name -- an attribution output for this standalone model.

ADDED (closer to what NBER WP 35195 actually describes): a genuine
post-hoc "polynomial-feature-sensitivity analysis," per the verbatim
abstract's own phrase, applied to an already-trained complex model's
predictions rather than used as a standalone predictor. Since this project
has no Transformer/RL model to analyze, xgb.py's already-computed,
already-honest OOS predictions (experiments/xgb/output/oos_predictions_xgb.csv) stand in as
"the complex model" -- for each test year, a degree-2 polynomial surrogate
is fit IN-SAMPLE on that year's realized characteristics to explain (not
forecast) XGBoost's own predictions that year, reporting the surrogate R^2
("how much of XGBoost's cross-sectional ranking can a smooth polynomial in
a handful of named characteristics explain") and the dominant terms -- this
is an attribution analysis of a different model, contemporaneous by
construction (no forecasting, hence no walk-forward split needed for this
part), not a new trading strategy. See run_sensitivity_analysis() and
experiments/poly/README.md.

Self-contained: data loading, rank transform, walk-forward schedule,
investability screen, beta-neutral LP, and evaluation code are ported from
ols.py / xgb.py (not imported).

Run: .venv/bin/python experiments/poly/poly.py
Requires experiments/xgb/output/oos_predictions_xgb.csv to already exist (from xgb.py) for
the sensitivity-analysis section; the primary poly model runs regardless.
Outputs (all in output/): oos_predictions_poly.csv,
    portfolio_holdings_beta_neutral_poly.csv,
    portfolio_returns_beta_neutral_poly.csv, poly_top_terms.csv,
    poly_xgb_sensitivity_analysis.csv, poly_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

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
N_SELECTED = 20

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


def select_top_characteristics_from_train(train_df: pd.DataFrame, stock_vars: list[str], n: int) -> list[str]:
    """Per-fold feature selection using ONLY the training-fold months --
    same monthly-IC-t-stat methodology as univariate.py, recomputed here
    rather than reusing its saved (OOS-window) output, to avoid look-ahead
    bias in the selection step (see module docstring)."""
    months = sorted(train_df["target_month"].unique())
    ic = np.full((len(months), len(stock_vars)), np.nan)
    for i, month in enumerate(months):
        grp = train_df[train_df["target_month"] == month]
        X = grp[stock_vars].values
        y = grp[TARGET_COL].values
        xm = X - X.mean(axis=0)
        ym = y - y.mean()
        num = (xm * ym[:, None]).sum(axis=0)
        denom = np.sqrt((xm**2).sum(axis=0) * (ym**2).sum())
        ic[i] = np.where(denom > 0, num / denom, 0.0)
    icdf = pd.DataFrame(ic, columns=stock_vars)
    ic_tstat = icdf.mean() / icdf.std() * np.sqrt(len(icdf))
    top = ic_tstat.abs().sort_values(ascending=False).head(n).index.tolist()
    return top


# ---------------------------------------------------------------------------
# 2. Expanding-window walk-forward degree-2 polynomial Ridge
# ---------------------------------------------------------------------------

ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


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


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str]):
    test_years = range(2021, 2027)
    preds, fold_params, fold_top_terms = [], [], []

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

        selected_vars = select_top_characteristics_from_train(train, stock_vars, N_SELECTED)
        poly = PolynomialFeatures(degree=2, include_bias=False)

        X_train_poly = poly.fit_transform(train[selected_vars].values)
        poly_feature_names = poly.get_feature_names_out(selected_vars)
        X_val_poly = poly.transform(val[selected_vars].values)
        X_test_poly = poly.transform(test[selected_vars].values)

        y_train, y_val = train[TARGET_COL].values, val[TARGET_COL].values
        model, alpha, val_mse = fit_best_ridge(X_train_poly, y_train, X_val_poly, y_val)
        fold_params.append({
            "test_year": test_year, "alpha": alpha, "val_mse": val_mse,
            "n_poly_terms": X_train_poly.shape[1], "selected_characteristics": selected_vars,
        })

        coef_series = pd.Series(model.coef_, index=poly_feature_names).sort_values(key=np.abs, ascending=False)
        top10 = coef_series.head(10)
        for term, coef in top10.items():
            fold_top_terms.append({"test_year": test_year, "term": term, "coefficient": float(coef)})

        y_pred = model.predict(X_test_poly)
        fold = test[
            ["permno", "target_month", "ticker", "company_name", TARGET_COL,
             "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m"]
        ].copy()
        fold["pred"] = y_pred
        fold["test_year"] = test_year
        preds.append(fold)

        print(
            f"[fold {test_year}] train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"alpha={alpha:g} val_mse={val_mse:.6f} | top term: {top10.index[0]} ({top10.iloc[0]:+.5f})"
        )
        print(f"  selected (train-only): {selected_vars}")

    return pd.concat(preds, ignore_index=True), fold_params, pd.DataFrame(fold_top_terms)


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
# 4. Post-hoc polynomial-feature-sensitivity analysis of xgb.py's predictions
#    (the technique NBER WP 35195 actually describes -- see module docstring)
# ---------------------------------------------------------------------------

XGB_PREDICTIONS_FILE = HERE.parent / "xgb" / "output" / "oos_predictions_xgb.csv"  # from xgb.py


def run_sensitivity_analysis(df: pd.DataFrame, stock_vars: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not XGB_PREDICTIONS_FILE.exists():
        print(f"  [skip] {XGB_PREDICTIONS_FILE} not found -- run xgb.py first. Skipping sensitivity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    xgb_preds = pd.read_csv(XGB_PREDICTIONS_FILE, usecols=["permno", "target_month", "pred"])
    xgb_preds["target_month"] = pd.to_datetime(xgb_preds["target_month"])
    xgb_preds = xgb_preds.rename(columns={"pred": "xgb_pred"})

    merged = df.merge(xgb_preds, on=["permno", "target_month"], how="inner")
    print(f"  Matched {len(merged):,} rows between this panel and xgb.py's OOS predictions.")

    summary_rows, term_rows = [], []
    for test_year, grp in merged.groupby(merged["target_month"].dt.year):
        if len(grp) < 50:
            continue
        # feature selection for the sensitivity analysis itself: which
        # characteristics correlate with XGBoost's OWN predictions that
        # year (not the real return -- this analysis explains the model,
        # not the market, so using same-period data is not look-ahead: no
        # forecast is being made).
        ic = grp[stock_vars].corrwith(grp["xgb_pred"])
        selected = ic.abs().sort_values(ascending=False).head(N_SELECTED).index.tolist()

        poly = PolynomialFeatures(degree=2, include_bias=False)
        Z = poly.fit_transform(grp[selected].values)
        feature_names = poly.get_feature_names_out(selected)

        model = Ridge(alpha=10.0)
        model.fit(Z, grp["xgb_pred"].values)
        pred = model.predict(Z)
        ss_res = float(np.sum((grp["xgb_pred"].values - pred) ** 2))
        ss_tot = float(np.sum((grp["xgb_pred"].values - grp["xgb_pred"].values.mean()) ** 2))
        surrogate_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        coefs = pd.Series(model.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
        for term, coef in coefs.head(8).items():
            term_rows.append({"test_year": int(test_year), "term": term, "coefficient": float(coef)})

        summary_rows.append({
            "test_year": int(test_year), "n_rows": int(len(grp)),
            "surrogate_r2_explaining_xgb_predictions": surrogate_r2,
            "selected_characteristics": selected,
        })
        print(f"  [sensitivity {test_year}] surrogate R^2 vs XGBoost predictions: {surrogate_r2:.4f}  "
              f"(top term: {coefs.index[0]} = {coefs.iloc[0]:+.5f})")

    return pd.DataFrame(summary_rows), pd.DataFrame(term_rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors")

    print("Cross-sectional preprocessing...")
    df = cross_sectional_rank_transform(df, stock_vars)

    print(f"\nWalk-forward expanding-window degree-2 polynomial Ridge (top {N_SELECTED} chars "
          f"reselected per fold from TRAINING data only, alpha tuned on validation)...")
    preds, fold_params, top_terms_df = run_walk_forward(df, stock_vars)
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / "oos_predictions_poly.csv", index=False)
    top_terms_df.to_csv(OUTPUT / "poly_top_terms.csv", index=False)

    r2 = oos_r2(preds[TARGET_COL].values, preds["pred"].values)
    print(f"\nOOS R^2 (full 2021-01 to 2026-08 sample): {r2 * 100:.4f}%")

    all_results = {
        "oos_r2": float(r2),
        "n_poly_terms": fold_params[0]["n_poly_terms"] if fold_params else None,
        "oos_prediction_rows": int(len(preds)), "fold_hyperparameters": fold_params,
    }

    print("\nBuilding portfolio: Beta-neutral...")
    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / "portfolio_holdings_beta_neutral_poly.csv", index=False)

    print("Computing performance...")
    bn_results = compute_performance(bn_stats, tag="beta_neutral_poly")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    all_results["beta_neutral"] = bn_results

    print("\n=== Summary: Beta-neutral (Polynomial, standalone model) ===")
    for k, v in bn_results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    print("\nPost-hoc polynomial-feature-sensitivity analysis of xgb.py's predictions "
          "(the technique NBER WP 35195 actually describes)...")
    sensitivity_summary, sensitivity_terms = run_sensitivity_analysis(df, stock_vars)
    if not sensitivity_summary.empty:
        sensitivity_summary.to_csv(OUTPUT / "poly_xgb_sensitivity_summary.csv", index=False)
        sensitivity_terms.to_csv(OUTPUT / "poly_xgb_sensitivity_analysis.csv", index=False)
        all_results["xgb_sensitivity_analysis"] = {
            "avg_surrogate_r2": float(sensitivity_summary["surrogate_r2_explaining_xgb_predictions"].mean()),
            "by_year": sensitivity_summary.to_dict(orient="records"),
        }

    with open(OUTPUT / "poly_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to experiments/poly/output/poly_results.json")
