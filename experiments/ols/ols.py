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
  4. Score every stock each out-of-sample month, screen for investability,
     form a dollar- and beta-neutral portfolio via a per-month LP, and
     evaluate it against the T-bill + 4% benchmark and the S&P 500 (including
     max drawdown, recovery, and Calmar ratio; monthly `drawdown`/
     `sp500_drawdown` series are written to portfolio_returns_beta_neutral.csv
     for the underwater chart).

Run: .venv/bin/python experiments/ols/ols.py
cache/ holds only downloaded external data (TB3MS.csv, SP500.csv).
Outputs (all in output/): oos_predictions.csv,
    portfolio_holdings_beta_neutral.csv, portfolio_returns_beta_neutral.csv,
    ols_results.json
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
CACHE = BASE / "cache"  # downloaded external data only (FRED series)
OUTPUT = HERE / "output"  # everything this script produces
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"

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

    # `prc`, `dolvol_126d` and `beta_60m` are themselves predictor columns and
    # get overwritten by the rank transform below -- snapshot the raw values
    # now so the portfolio-construction stage can screen for investability
    # and constrain for beta neutrality.
    df["raw_prc"] = df["prc"]
    df["raw_dolvol_126d"] = df["dolvol_126d"]
    df["raw_me"] = df["me"]
    df["raw_beta_60m"] = df["beta_60m"]

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
                "raw_beta_60m",
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
# median-$44M, median-$2.47 stocks (see experiments/ols/README.md) -- diagnosed directly
# from this run's own holdings. Applied only at portfolio-formation time
# (using values as of the characteristic month, i.e. known before the trade),
# never during model training. Single threshold: 6-month average daily
# dollar volume, the most direct measure of whether a position is tradable.
MIN_DOLLAR_VOLUME = 10_000_000.0  # $10M average daily dollar volume

MAX_WEIGHT = 0.01  # per-name cap: forces diversification across >=200 names


def build_beta_neutral_portfolio(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Investability-screened, dollar-neutral, and beta-neutral by explicit
    construction: each month, solve a small linear program that maximizes
    expected (predicted) return subject to
        sum(w) = 0                      (dollar/net neutral)
        sum(w_i * beta_60m_i) = 0       (beta neutral, per docs/FIAM.md's
                                          "beta-weighted long exposure =
                                          beta-weighted short exposure")
        sum(|w_i|) = 2.0                (gross = 200%)
        |w_i| <= MAX_WEIGHT             (per-name cap, forces >=200 names)
    Variables are split w_i = w_i^+ - w_i^- (both >=0) so every constraint is
    linear; scipy.optimize.linprog (HiGHS) solves it directly. Stocks with a
    missing beta_60m cannot enter the beta constraint and are excluded from
    the eligible universe for this construction (see experiments/ols/README.md).
    """
    holdings = []
    monthly_stats = []

    for month, grp in preds.groupby("target_month"):
        grp = grp[grp["raw_dolvol_126d"] >= MIN_DOLLAR_VOLUME]
        grp = grp.dropna(subset=["raw_beta_60m"]).reset_index(drop=True)
        n = len(grp)
        if n < 2 * int(1.0 / MAX_WEIGHT):
            continue  # not enough names to reach 200% gross under the cap

        pred = grp["pred"].values
        beta = grp["raw_beta_60m"].values

        # x = [w+_1..w+_n, w-_1..w-_n], all >= 0
        c = np.concatenate([-pred, pred])  # minimize -sum(pred*(w+ - w-))
        A_eq = np.array(
            [
                np.concatenate([np.ones(n), -np.ones(n)]),  # net = 0
                np.concatenate([beta, -beta]),  # beta-weighted exposure = 0
                np.concatenate([np.ones(n), np.ones(n)]),  # gross = 2.0
            ]
        )
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
        holdings.append(
            both[["target_month", "permno", "ticker", "company_name", "weight", TARGET_COL, "pred", "raw_beta_60m"]]
        )

        port_ret = (both["weight"] * both[TARGET_COL]).sum()
        long_mask = both["weight"] > 0
        long_ret = (both.loc[long_mask, "weight"] * both.loc[long_mask, TARGET_COL]).sum()
        short_ret = (both.loc[~long_mask, "weight"] * both.loc[~long_mask, TARGET_COL]).sum()
        gross = both["weight"].abs().sum()
        net = both["weight"].sum()
        beta_exposure = (both["weight"] * both["raw_beta_60m"]).sum()
        monthly_stats.append(
            {
                "target_month": month,
                "port_excess_ret": port_ret,
                "long_leg_ret": long_ret,
                "short_leg_ret": short_ret,
                "gross_exposure": gross,
                "net_exposure": net,
                "beta_exposure": beta_exposure,
                "n_long": int((both["weight"] > 0).sum()),
                "n_short": int((both["weight"] < 0).sum()),
                "n_positions": int(keep.sum()),
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


def compute_turnover_and_concentration(holdings_df: pd.DataFrame) -> dict:
    """Average monthly one-way turnover (share of gross book replaced) and
    position-concentration stats, both required by docs/FIAM.md's reporting
    list but independent of the benchmark/FRED join above."""
    piv = holdings_df.pivot_table(
        index="target_month", columns="permno", values="weight", fill_value=0.0
    ).sort_index()
    # one-way turnover = half the L1 change in weights, as a share of gross (2.0)
    # min_count=1 avoids pandas silently treating an all-NaN first-diff row as 0
    diffs = piv.diff().abs().sum(axis=1, min_count=1) / 2.0
    turnover = (diffs.iloc[1:] / 2.0).rename("turnover")  # first month has no prior book

    abs_w = holdings_df["weight"].abs()
    top10_share_by_month = holdings_df.groupby("target_month")["weight"].apply(
        lambda w: w.abs().nlargest(10).sum() / 2.0  # share of 200% gross book
    )

    return {
        "avg_monthly_turnover": float(turnover.mean()),
        "min_monthly_turnover": float(turnover.min()),
        "max_monthly_turnover": float(turnover.max()),
        "avg_position_weight_abs": float(abs_w.mean()),
        "max_position_weight_abs": float(abs_w.max()),
        "avg_top10_share_of_gross": float(top10_share_by_month.mean()),
    }


def compute_short_book_characteristics(holdings_df: pd.DataFrame) -> dict:
    """Average/median market cap, average dollar volume, and small-cap share
    of the short leg, as required by docs/FIAM.md's reporting list. No direct
    borrow-cost/hard-to-borrow data is provided in this dataset, so small
    market cap is used as the stated proxy for "plausibly hard to borrow"."""
    chars = pd.read_parquet(CHARS_FILE, columns=["permno", "eom", "me", "dolvol_126d"])
    chars["eom"] = pd.to_datetime(chars["eom"])

    h = holdings_df.copy()
    h["target_month"] = pd.to_datetime(h["target_month"])
    h["char_month"] = (h["target_month"] - pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    h["char_month"] = pd.to_datetime(h["char_month"]) + pd.offsets.MonthEnd(0)
    m = h.merge(chars, left_on=["permno", "char_month"], right_on=["permno", "eom"], how="left")
    short = m[m["weight"] < 0]

    return {
        "short_book_avg_market_cap_musd": float(short["me"].mean()),
        "short_book_median_market_cap_musd": float(short["me"].median()),
        "short_book_avg_dollar_volume_usd": float(short["dolvol_126d"].mean()),
        "short_book_share_below_2000m_cap": float((short["me"] < 2000).mean()),
        "short_book_share_below_1000m_cap": float((short["me"] < 1000).mean()),
    }


def drawdown_series(rets: pd.Series) -> pd.Series:
    """Drawdown from running peak of compounded wealth, W_t / max(W_0..W_t) - 1.
    Wealth starts at W_0 = 1 before the first month, so a loss in month one
    counts as a drawdown -- unlike fiam/portfolio_analysis_hackathon.py, whose
    cummax over cumulative log returns omits the starting capital (and reports
    log rather than simple drawdown)."""
    wealth = (1 + rets).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def compute_drawdown_stats(rets: pd.Series, dates: pd.Series, cagr: float) -> dict:
    """Max drawdown with its peak/trough/recovery dates and durations (in
    months), longest underwater spell, current drawdown, and Calmar ratio.
    `rets` and `dates` must be aligned and sorted by date."""
    rets = rets.reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    dd = drawdown_series(rets)

    trough_i = int(dd.idxmin())
    max_dd = float(dd.iloc[trough_i])

    if max_dd < 0:
        # peak = last month at or before the trough where wealth set a new high;
        # -1 means the peak is the starting capital, before the first month
        at_high = dd.iloc[: trough_i + 1] >= 0
        peak_i = int(at_high[at_high].index.max()) if at_high.any() else -1
        recovered = dd.iloc[trough_i + 1 :] >= 0
        recovery_i = int(recovered[recovered].index.min()) if recovered.any() else None
    else:
        peak_i, recovery_i = trough_i, trough_i

    def date_at(i):
        if i is None:
            return None
        if i < 0:  # starting capital, dated one month before the first return
            return str((dates.iloc[0] - pd.offsets.MonthEnd(1)).date())
        return str(dates.iloc[i].date())

    # longest run of consecutive months below the prior peak
    underwater = (dd < 0).astype(int)
    runs = underwater.groupby((underwater == 0).cumsum()).sum()
    longest_underwater = int(runs.max()) if len(runs) else 0

    return {
        "max_drawdown": max_dd,
        "max_drawdown_peak_date": date_at(peak_i),
        "max_drawdown_trough_date": date_at(trough_i),
        "max_drawdown_recovery_date": date_at(recovery_i),
        "max_drawdown_peak_to_trough_months": int(trough_i - peak_i),
        "max_drawdown_recovery_months": None if recovery_i is None else int(recovery_i - trough_i),
        "longest_underwater_months": longest_underwater,
        "current_drawdown": float(dd.iloc[-1]),
        "avg_drawdown_when_underwater": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else None,
    }


def compute_performance(stats_df: pd.DataFrame, tag: str) -> dict:
    tb3ms, sp500 = load_fred_series()
    perf = stats_df.merge(tb3ms, left_on="target_month", right_on="month", how="left")
    perf = perf.merge(sp500, left_on="target_month", right_on="month", how="left", suffixes=("", "_sp"))

    # Benchmark hurdle (a time series, not a constant): TB3MS/100/12 + 4%/12
    perf["benchmark_monthly"] = perf["rf_monthly"] + 0.04 / 12.0

    # Convention: the dollar-neutral spread return is already earned in excess
    # of a risk-free rate (the pipeline's RF embedded in ret_exc_lead1m, which
    # this analysis assumes is close enough to TB3MS to treat as the same
    # cash rate -- see experiments/ols/README.md for the caveat). Active return over the
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

    def calendar_year_of(col: str) -> dict:
        d = perf.dropna(subset=[col]).assign(year=lambda x: x["target_month"].dt.year)
        return (
            d.groupby("year")
            .apply(lambda g: (1 + g[col]).prod() - 1, include_groups=False)
            .to_dict()
        )

    calendar_year = calendar_year_of("port_excess_ret")
    calendar_year_benchmark = calendar_year_of("benchmark_monthly")
    calendar_year_sp500 = calendar_year_of("sp500_ret")

    long_leg_cagr = (1 + perf["long_leg_ret"]).prod() ** (12 / n_months) - 1
    short_leg_cagr = (1 + perf["short_leg_ret"]).prod() ** (12 / n_months) - 1

    # Drawdown series for the underwater chart (strategy with S&P 500 overlaid)
    perf["drawdown"] = drawdown_series(perf["port_excess_ret"])
    perf["sp500_drawdown"] = drawdown_series(perf["sp500_ret"].fillna(0.0))
    sp500_cagr = (1 + perf["sp500_ret"].fillna(0.0)).prod() ** (12 / n_months) - 1
    dd_stats = compute_drawdown_stats(perf["port_excess_ret"], perf["target_month"], ann_ret_geo)
    sp500_dd_stats = compute_drawdown_stats(
        perf["sp500_ret"].fillna(0.0), perf["target_month"], sp500_cagr
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
        "avg_n_long": float(stats_df["n_long"].mean()),
        "avg_n_short": float(stats_df["n_short"].mean()),
        "best_month": {"date": str(best_month["target_month"].date()), "ret": float(best_month["port_excess_ret"])},
        "worst_month": {"date": str(worst_month["target_month"].date()), "ret": float(worst_month["port_excess_ret"])},
        "calendar_year_returns": {int(k): float(v) for k, v in calendar_year.items()},
        "calendar_year_returns_benchmark": {int(k): float(v) for k, v in calendar_year_benchmark.items()},
        "calendar_year_returns_sp500": {int(k): float(v) for k, v in calendar_year_sp500.items()},
        "long_leg_avg_monthly_ret": float(perf["long_leg_ret"].mean()),
        "long_leg_cagr": float(long_leg_cagr),
        "short_leg_avg_monthly_ret": float(perf["short_leg_ret"].mean()),
        "short_leg_cagr": float(short_leg_cagr),
        **dd_stats,
        "sp500_drawdown": sp500_dd_stats,
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

    print("\nBuilding portfolio: Beta-neutral (investability-screened + LP beta constraint)...")
    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / "portfolio_holdings_beta_neutral.csv", index=False)

    print("Computing performance vs T-bill+4% and S&P 500...")
    bn_results = compute_performance(bn_stats, tag="beta_neutral")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    all_results["beta_neutral"] = bn_results

    print("\n=== Summary: Beta-neutral ===")
    for k, v in bn_results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    with open(OUTPUT / "ols_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nFull results written to experiments/ols/output/ols_results.json")
