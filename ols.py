"""
FIAM 2026 - Baseline OLS strategy.

Pipeline:
  1. Load the monthly characteristics panel and the 147 predictor columns.
  2. Cross-sectionally winsorize/rank-transform each predictor to [-1, 1] within
     each characteristic month (median-fill missing values first).
  3. Walk forward: expanding training window, 2-year rolling validation window
     (unused for plain OLS, which has no hyperparameters -- kept for schedule
     parity with the penalized-linear models), one calendar year of test.
     Refit annually, predict monthly. Splits are assigned by TARGET month.
  4. Score every stock each out-of-sample month, form a dollar-neutral
     top-100 / bottom-100 equal-weight portfolio, and evaluate it against the
     T-bill + 4% benchmark and the S&P 500.

Run: .venv/bin/python ols.py
cache/ holds only downloaded external data (TB3MS.csv, SP500.csv).
Outputs (all in output/): oos_predictions.csv,
    portfolio_holdings_{unfiltered,investable}.csv,
    portfolio_returns_{unfiltered,investable}.csv, ols_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"  # downloaded external data only (FRED series)
OUTPUT = BASE / "output"  # everything this script produces
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"
N_LEGS = 100  # top/bottom N stocks per leg -> 2*N_LEGS positions

# Out-of-sample evaluation window (competition spec)
OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")

# ---------------------------------------------------------------------------
# 1. Load data and build the model table
# ---------------------------------------------------------------------------


def load_model_table() -> pd.DataFrame:
    stock_vars = pd.read_csv(FACTOR_LIST)["variable"].tolist()
    keep_cols = stock_vars + [
        "permno",
        "eom",
        "ticker",
        "company_name",
        "me",  # raw market equity ($M), auxiliary column, used only for the
               # investability screen below -- distinct from the predictor
               # `market_equity` and never fed to the model.
        TARGET_COL,
    ]
    df = pd.read_parquet(CHARS_FILE, columns=keep_cols)
    df["eom"] = pd.to_datetime(df["eom"])

    # `prc` and `dolvol_126d` are themselves predictor columns and get
    # overwritten by the rank transform below -- snapshot the raw values now
    # so the portfolio-construction stage can screen for investability.
    df["raw_prc"] = df["prc"]
    df["raw_dolvol_126d"] = df["dolvol_126d"]
    df["raw_me"] = df["me"]

    # Target-month key: characteristics at eom (month t) predict returns
    # realized in month t+1. Assign every row to that t+1 month for splitting.
    df["target_month"] = (df["eom"] + pd.offsets.MonthBegin(1)).values.astype(
        "datetime64[M]"
    )
    df["target_month"] = pd.to_datetime(df["target_month"]) + pd.offsets.MonthEnd(0)

    # Drop rows with no realized next-month return (terminal obs / gaps).
    df = df[df[TARGET_COL].notna()].copy()
    return df, stock_vars


def cross_sectional_rank_transform(df: pd.DataFrame, stock_vars: list[str]) -> pd.DataFrame:
    """Median-fill then rank-transform each predictor to [-1, 1], per
    characteristic month (eom), matching the template's convention."""
    out = df.copy()
    g = out.groupby("eom")
    for var in stock_vars:
        med = g[var].transform("median")
        filled = out[var].fillna(med)
        # a handful of months can be all-NaN for a rarely populated factor;
        # median-of-median-fill leaves those as NaN, fill with 0 (neutral rank)
        filled = filled.fillna(0.0)
        out[var] = filled

    g = out.groupby("eom")
    for var in stock_vars:
        r = g[var].rank(method="dense")
        rmax = r.groupby(out["eom"]).transform("max")
        out[var] = np.where(rmax > 0, (r / rmax) * 2 - 1, 0.0)
    return out


# ---------------------------------------------------------------------------
# 2. Expanding-window walk-forward OLS
# ---------------------------------------------------------------------------


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str]) -> pd.DataFrame:
    test_years = range(2021, 2027)  # 2026 is partial -> filtered to OOS_END below
    preds = []

    for test_year in test_years:
        test_start, test_end = year_bounds(test_year)
        test_end = min(test_end, OOS_END)

        val_start, _ = year_bounds(test_year - 2)
        _, val_end = year_bounds(test_year - 1)

        train_mask = df["target_month"] < val_start
        val_mask = (df["target_month"] >= val_start) & (df["target_month"] <= val_end)
        test_mask = (df["target_month"] >= test_start) & (df["target_month"] <= test_end)

        train = df[train_mask]
        test = df[test_mask]
        if train.empty or test.empty:
            continue

        X_train = train[stock_vars].values
        y_train = train[TARGET_COL].values
        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = test[stock_vars].values
        y_pred = model.predict(X_test)

        fold = test[
            [
                "permno",
                "target_month",
                "ticker",
                "company_name",
                TARGET_COL,
                "raw_prc",
                "raw_dolvol_126d",
                "raw_me",
            ]
        ].copy()
        fold["pred"] = y_pred
        fold["test_year"] = test_year
        fold["n_train"] = len(train)
        fold["n_val"] = int(val_mask.sum())
        preds.append(fold)

        print(
            f"[fold {test_year}] train={train['target_month'].min().date()}"
            f"->{train['target_month'].max().date()} ({len(train):,} rows), "
            f"val={val_start.date()}->{val_end.date()} ({int(val_mask.sum()):,} rows), "
            f"test={test_start.date()}->{test_end.date()} ({len(test):,} rows)"
        )

    return pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------


def oos_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum(actual**2)


# Investability screen: an unscreened top/bottom-N portfolio ends up shorting
# median-$44M, median-$2.47 stocks (see docs/OLS.md) -- diagnosed directly
# from this run's own holdings. Applied only at portfolio-formation time
# (using values as of the characteristic month, i.e. known before the trade),
# never during model training. Single threshold: 6-month average daily
# dollar volume, the most direct measure of whether a position is tradable.
MIN_DOLLAR_VOLUME = 10_000_000.0  # $10M average daily dollar volume


def build_portfolio(preds: pd.DataFrame, screen: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-N / bottom-N equal-weight, dollar-neutral, monthly rebalance."""
    holdings = []
    monthly_stats = []

    for month, grp in preds.groupby("target_month"):
        if screen:
            grp = grp[grp["raw_dolvol_126d"] >= MIN_DOLLAR_VOLUME]
        grp = grp.sort_values("pred", ascending=False)
        n = len(grp)
        n_leg = min(N_LEGS, n // 2)
        if n_leg < 1:
            continue
        longs = grp.head(n_leg).copy()
        shorts = grp.tail(n_leg).copy()
        longs["weight"] = 1.0 / n_leg
        shorts["weight"] = -1.0 / n_leg

        both = pd.concat([longs, shorts])
        both["target_month"] = month
        holdings.append(
            both[["target_month", "permno", "ticker", "company_name", "weight", TARGET_COL, "pred"]]
        )

        port_ret = (both["weight"] * both[TARGET_COL]).sum()
        gross = both["weight"].abs().sum()
        net = both["weight"].sum()
        monthly_stats.append(
            {
                "target_month": month,
                "port_excess_ret": port_ret,
                "gross_exposure": gross,
                "net_exposure": net,
                "n_long": n_leg,
                "n_short": n_leg,
                "n_positions": 2 * n_leg,
            }
        )

    holdings_df = pd.concat(holdings, ignore_index=True)
    stats_df = pd.DataFrame(monthly_stats).sort_values("target_month").reset_index(drop=True)
    return holdings_df, stats_df


def load_fred_series() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def compute_performance(stats_df: pd.DataFrame, tag: str) -> dict:
    tb3ms, sp500 = load_fred_series()
    perf = stats_df.merge(tb3ms, left_on="target_month", right_on="month", how="left")
    perf = perf.merge(sp500, left_on="target_month", right_on="month", how="left", suffixes=("", "_sp"))

    # Benchmark hurdle (a time series, not a constant): TB3MS/100/12 + 4%/12
    perf["benchmark_monthly"] = perf["rf_monthly"] + 0.04 / 12.0

    # Convention: the dollar-neutral spread return is already earned in excess
    # of a risk-free rate (the pipeline's RF embedded in ret_exc_lead1m, which
    # this analysis assumes is close enough to TB3MS to treat as the same
    # cash rate -- see docs/OLS.md for the caveat). Active return over the
    # competition benchmark is therefore the spread return minus the +4%/yr
    # hurdle only.
    perf["active_ret"] = perf["port_excess_ret"] - 0.04 / 12.0

    ir = np.sqrt(12) * perf["active_ret"].mean() / perf["active_ret"].std()
    sharpe = np.sqrt(12) * perf["port_excess_ret"].mean() / perf["port_excess_ret"].std()

    # Alpha/beta regression: R_p - r_f = alpha + beta*(R_sp500 - r_f) + eps
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

    calendar_year = (
        perf.assign(year=perf["target_month"].dt.year)
        .groupby("year")
        .apply(lambda g: (1 + g["port_excess_ret"]).prod() - 1, include_groups=False)
        .to_dict()
    )

    results = {
        "n_months": int(n_months),
        "avg_monthly_return": float(perf["port_excess_ret"].mean()),
        "annualized_return_arith": float(ann_ret_arith),
        "annualized_return_geo_cagr": float(ann_ret_geo),
        "cumulative_return": float(cum_ret),
        "hit_rate_active": float(hit_rate),
        "information_ratio": float(ir),
        "sharpe_ratio": float(sharpe),
        "alpha_monthly": float(alpha_monthly),
        "alpha_annualized": float(alpha_monthly * 12),
        "alpha_tstat": float(alpha_t),
        "alpha_se_monthly": float(alpha_se),
        "beta": float(beta),
        "beta_se": float(beta_se),
        "beta_tstat": float(beta_t),
        "corr_with_sp500": float(perf["port_excess_ret"].corr(perf["sp500_ret"])),
        "avg_gross_exposure": float(stats_df["gross_exposure"].mean()),
        "max_gross_exposure": float(stats_df["gross_exposure"].max()),
        "avg_net_exposure": float(stats_df["net_exposure"].mean()),
        "min_net_exposure": float(stats_df["net_exposure"].min()),
        "max_net_exposure": float(stats_df["net_exposure"].max()),
        "avg_n_positions": float(stats_df["n_positions"].mean()),
        "best_month": {"date": str(best_month["target_month"].date()), "ret": float(best_month["port_excess_ret"])},
        "worst_month": {"date": str(worst_month["target_month"].date()), "ret": float(worst_month["port_excess_ret"])},
        "calendar_year_returns": {int(k): float(v) for k, v in calendar_year.items()},
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

    print("Cross-sectional preprocessing (median fill + rank to [-1,1] per month)...")
    df = cross_sectional_rank_transform(df, stock_vars)

    print("Walk-forward expanding-window OLS...")
    preds = run_walk_forward(df, stock_vars)
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / "oos_predictions.csv", index=False)

    r2 = oos_r2(preds[TARGET_COL].values, preds["pred"].values)
    print(f"\nOOS R^2 (full 2021-01 to 2026-08 sample): {r2 * 100:.4f}%")

    all_results = {
        "oos_r2": float(r2),
        "n_predictor_columns": len(stock_vars),
        "oos_prediction_rows": int(len(preds)),
    }

    for screen, tag, label in [
        (False, "unfiltered", "Unfiltered (raw top/bottom-100)"),
        (True, "investable", "Investability-screened (dolvol_126d>=$10M/day)"),
    ]:
        print(f"\nBuilding portfolio: {label}...")
        holdings, stats = build_portfolio(preds, screen=screen)
        holdings.to_csv(OUTPUT / f"portfolio_holdings_{tag}.csv", index=False)

        print("Computing performance vs T-bill+4% and S&P 500...")
        perf_results = compute_performance(stats, tag=tag)
        all_results[tag] = perf_results

        print(f"\n=== Summary: {label} ===")
        for k, v in perf_results.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")

    with open(OUTPUT / "ols_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nFull results written to output/ols_results.json")
