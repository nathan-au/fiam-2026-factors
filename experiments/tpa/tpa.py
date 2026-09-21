"""
FIAM 2026 - Total Portfolio Approach (TPA) adaptation, factors only (tpa.py).

TPA manages ONE total portfolio against one objective and gives each opportunity (sleeve) capital according to its
marginal contribution to the total's risk and return, instead of fixed silos or equal buckets. We hold a single
long/short equity book, so the sleeves here are the SEVEN pre-registered factor groups of largecap.py (value,
profitability, investment/issuance/accruals, quality, surprise, volatility/beta, liquidity). Question: does sizing
the sleeves by their contribution to total risk / risk-adjusted return beat largecap.py's equal-group composite?
(No 8-K sleeve yet: that plugs in later as an eighth sleeve.)

PRE-REGISTERED DESIGN (fixed before any result was seen)
  Universe / factors / portfolio LP   exactly largecap.py's (floor $2B default, --floor 1000 sensitivity; 18 factors;
                                      variants lc_free, lc_t20, lc_t10 [HEADLINE], lc_t10_w05).
  Sleeve signal    group score = mean of the group's signed within-universe monthly ranks (as in largecap.py),
                   centred within month and scaled to mean |score| = 1, so a unit of sleeve capital is a unit of
                   gross exposure whatever the group's size.
  Sleeve return    r[g,t] = sum_i z[g,i] * ret_exc_lead1m[i] / sum_i |z[g,i]| over the universe in target month t:
                   the return of a unit-gross, score-weighted long/short sleeve (not beta-neutral; used only for
                   allocation).
  Estimation       Each test year Y uses ONLY sleeve returns with target month < Y-01-01 (expanding window: 47 months
                   for 2021 ... 107 for 2026); sleeve weights are fixed within the year. No test-period return
                   influences any weight.
  Arms             tpa_eq   equal capital across sleeves (1/7)                              (CONTROL, ~ largecap comp)
                   tpa_erc  equal RISK contribution: sleeve weights with equal marginal-risk-weighted shares of total
                            sleeve-portfolio variance (Ledoit-Wolf covariance); uses no return forecasts
                   tpa_ms   marginal-Sharpe allocation: long-only max-Sharpe on sleeve returns, means shrunk 50%
                            toward the cross-sleeve mean, Ledoit-Wolf covariance, each sleeve <= 35%
  Composite        score_i = sum_g w_g * z[g,i]; then the same LP variants as largecap.py.
  Verdict          A TPA arm is called useful only if it beats `tpa_eq` on net IR in BOTH lc_t10 and lc_free AND the
                   paired monthly universe-IC difference has t >= 2. Between "beats on IR" and t < 2 it is reported
                   as suggestive only. Expectation before running: with 47-107 monthly observations for 7 sleeves,
                   estimation error is large and 1/N is hard to beat (DeMiguel et al.), and experiments/factor_filter/README.md sec 2
                   found factor IC does not persist in the investable universe. Not tuned; not re-run.

Uses the frozen harness from et.py and the pre-registered definitions from largecap.py (both copied in below, unchanged, so this file is standalone).

Run:  .venv/bin/python experiments/tpa/tpa.py [--floor 2000] [--smoke]
Outputs (output/): oos_predictions_<tag>_tpa_<arm>.csv, portfolio_{holdings,returns}_<variant>_<tag>_tpa_<arm>.csv,
    tpa_results_<tag>.json, tpa_summary_<tag>.csv, tpa_weights_<tag>.csv, tpa_monthly_ic_<tag>.csv
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

HERE = Path(__file__).resolve().parent  # experiments/tpa/
OUTPUT = HERE / "output"


# ---------------------------------------------------------------------------
# Inlined from et.py (copied verbatim so this file runs on its own; only the parts used here)
# ---------------------------------------------------------------------------


import statsmodels.api as sm
from scipy import sparse
from scipy.optimize import linprog
BASE = HERE.parents[1]  # project root: fiam/ data and cache/ are shared
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"  # downloaded external data only (FRED series)
OUT = OUTPUT  # rebound by --smoke so plumbing tests never touch real outputs
CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"
TARGET_COL = "ret_exc_lead1m"
OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")
TEST_YEARS = range(2021, 2027)


def load_model_table() -> tuple[pd.DataFrame, list[str]]:
    stock_vars = pd.read_csv(FACTOR_LIST)["variable"].tolist()
    keep_cols = stock_vars + ["permno", "eom", "ticker", "company_name", "me", "gics", TARGET_COL]
    df = pd.read_parquet(CHARS_FILE, columns=keep_cols)
    df["eom"] = pd.to_datetime(df["eom"])

    # Raw snapshots before the rank transform overwrites the predictor columns.
    df["raw_prc"] = df["prc"]
    df["raw_dolvol_126d"] = df["dolvol_126d"]
    df["raw_me"] = df["me"]
    df["raw_beta_60m"] = df["beta_60m"]
    df["raw_betabab_1260d"] = df["betabab_1260d"]
    # 2-digit GICS sector (11 sectors); unclassified rows share one "NA" bucket.
    df["sector"] = df["gics"].astype("string").str[:2].fillna("NA").astype(str)

    df["target_month"] = (df["eom"] + pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    df["target_month"] = pd.to_datetime(df["target_month"]) + pd.offsets.MonthEnd(0)
    df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
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


class Context:
    """Design matrix built once; folds are index arrays; `meta` holds the test rows
    (identical for every experiment -- only the prediction vector changes)."""

    META_COLS = ["permno", "target_month", "ticker", "company_name", TARGET_COL, "sector",
                 "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m", "raw_betabab_1260d"]

    def __init__(self, df: pd.DataFrame, stock_vars: list[str], max_folds=None):
        self.stock_vars = stock_vars
        self.col_idx = {c: i for i, c in enumerate(stock_vars)}
        self.Xall = df[stock_vars].to_numpy(dtype=np.float32)
        self.y = df[TARGET_COL].to_numpy(dtype=np.float64)
        self.tm = df["target_month"].to_numpy()
        tm = df["target_month"]
        self.folds, te_all, start = [], [], 0
        for year in TEST_YEARS:
            test_start, test_end = pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), OOS_END)
            val_start, val_end = pd.Timestamp(f"{year - 2}-01-01"), pd.Timestamp(f"{year - 1}-12-31")
            tr = np.flatnonzero((tm < val_start).to_numpy())
            va = np.flatnonzero(((tm >= val_start) & (tm <= val_end)).to_numpy())
            te = np.flatnonzero(((tm >= test_start) & (tm <= test_end)).to_numpy())
            if len(tr) == 0 or len(va) == 0 or len(te) == 0:
                continue
            self.folds.append({"year": year, "tr": tr, "va": va, "te": te, "sl": slice(start, start + len(te))})
            start += len(te)
            te_all.append(te)
            if max_folds and len(self.folds) >= max_folds:
                break
        self.meta = df.iloc[np.concatenate(te_all)][self.META_COLS].reset_index(drop=True)
        # Rows that pass the PM portfolio's tradeability screens (price >= $5, market cap >= $500M, $10M dollar
        # volume), as of the characteristic month. Used only by the `_trad` arms to restrict TRAINING/VALIDATION rows.
        self.tradeable = ((df["raw_prc"].abs() >= PM_CFG["min_price"]) & (df["raw_me"] >= PM_CFG["min_mcap"])
                          & (df["raw_dolvol_126d"] >= PM_CFG["min_dolvol"])).to_numpy()

    def cols(self, features):
        return [self.col_idx[f] for f in features]


def pred_rank_autocorr(preds, min_dolvol=10_000_000.0) -> float:
    """Mean month-over-month Spearman correlation of the prediction ranking on the
    liquid universe -- a stability measure (low = the ranking reshuffles = high turnover)."""
    p = preds[preds["raw_dolvol_126d"] >= min_dolvol]
    R = p.pivot(index="target_month", columns="permno", values="pred").sort_index().rank(axis=1)
    vals = [R.iloc[i].corr(R.iloc[i - 1]) for i in range(1, len(R))]
    return float(np.nanmean(vals))


def preds_frame(ctx: Context, pred_vec) -> pd.DataFrame:
    out = ctx.meta.copy()
    out["pred"] = pred_vec
    return out[(out["target_month"] >= OOS_START) & (out["target_month"] <= OOS_END)].reset_index(drop=True)


PM_CFG = dict(min_dolvol=10_000_000.0, min_price=5.0, min_mcap=500.0, beta_cols=("raw_beta_60m", "raw_betabab_1260d"),
              max_weight=0.01, sector_net=0.05, sector_gross=0.70, turnover=0.10)

GROSS = 2.0
RELAX_LADDER = (1.0, 1.5, 2.0, 3.0, 5.0, None)  # if a turnover cap is infeasible, loosen it stepwise (logged)

# 3b. Cost assumptions (NOT measured -- the panel has no borrow or spread-by-name data).
COST_TIERS = [  # (min market cap $M, one-way trading cost bp, annual borrow bp on short notional)
    (10_000.0, 5.0, 30.0),
    (2_000.0, 10.0, 75.0),
    (0.0, 20.0, 200.0),
]


def cost_bps(me):
    me = np.asarray(me, dtype=float)
    trade = np.full(me.shape, COST_TIERS[-1][1])
    borrow = np.full(me.shape, COST_TIERS[-1][2])
    for lo, t, b in reversed(COST_TIERS[:-1]):
        m = me >= lo
        trade[m], borrow[m] = t, b
    return trade, borrow


def _solve_lp(g, cfg, prev, mult):
    n = len(g)
    pred = g["pred"].to_numpy()
    turnover = cfg.get("turnover")
    use_turn = turnover is not None and mult is not None and len(prev) > 0
    a_prev, budget = None, None
    if use_turn:
        a_prev = prev.reindex(g["permno"]).fillna(0.0).to_numpy()
        dropped = float(prev[~prev.index.isin(g["permno"])].abs().sum())
        budget = mult * 2.0 * GROSS * turnover - dropped  # sum|dw| = 2 * gross * one-way turnover share
        if budget < 0:
            return None

    def pad(M):
        M = sparse.csr_matrix(M)
        return sparse.hstack([M, sparse.csr_matrix((M.shape[0], n))], format="csr") if use_turn else M

    def row(a, b):
        return pad(sparse.csr_matrix(np.concatenate([a, b]).reshape(1, -1)))

    ones, zeros = np.ones(n), np.zeros(n)
    eq_rows = [row(ones, -ones)]
    b_eq = [0.0]
    for bc in cfg["beta_cols"]:
        beta = g[bc].to_numpy()
        eq_rows.append(row(beta, -beta))
        b_eq.append(0.0)
    eq_rows.append(row(ones, ones))
    b_eq.append(GROSS)
    A_eq = sparse.vstack(eq_rows, format="csr")

    ub_rows, b_ub = [], []
    if cfg.get("sector_net") is not None or cfg.get("sector_gross") is not None:
        for s in g["sector"].unique():
            a = (g["sector"].to_numpy() == s).astype(float)
            if cfg.get("sector_net") is not None:
                ub_rows += [row(a, -a), row(-a, a)]
                b_ub += [cfg["sector_net"]] * 2
            if cfg.get("sector_gross") is not None:
                ub_rows.append(row(a, a))
                b_ub.append(cfg["sector_gross"])
    if use_turn:
        I = sparse.identity(n, format="csr")
        ub_rows.append(sparse.hstack([I, -I, -I], format="csr"))
        b_ub += list(a_prev)
        ub_rows.append(sparse.hstack([-I, I, -I], format="csr"))
        b_ub += list(-a_prev)
        ub_rows.append(sparse.csr_matrix(np.concatenate([zeros, zeros, ones]).reshape(1, -1)))
        b_ub.append(budget)
    A_ub = sparse.vstack(ub_rows, format="csr") if ub_rows else None

    c = np.concatenate([-pred, pred] + ([zeros] if use_turn else []))
    bounds = [(0.0, cfg["max_weight"])] * (2 * n) + ([(0.0, None)] * n if use_turn else [])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub if ub_rows else None, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[:n] - res.x[n:2 * n]


def build_portfolio(preds, cfg):
    """One LP per month, solved sequentially (the turnover constraint depends on last month's book,
    drifted by that month's realized returns -- information available at the rebalance date)."""
    holdings, monthly = [], []
    prev = pd.Series(dtype=float)
    for month, grp in preds.groupby("target_month"):
        g = grp[(grp["raw_dolvol_126d"] >= cfg["min_dolvol"]) & (grp["raw_prc"].abs() >= cfg["min_price"])
                & (grp["raw_me"] >= cfg["min_mcap"])]
        g = g.dropna(subset=list(cfg["beta_cols"])).reset_index(drop=True)
        n = len(g)
        if n < 2 * int(1.0 / cfg["max_weight"]):
            continue
        w, used = None, None
        for mult in RELAX_LADDER:
            w = _solve_lp(g, cfg, prev, mult)
            if w is not None:
                used = mult
                break
        if w is None:
            print(f"  [warn] LP infeasible for {month.date()}, skipping month")
            continue
        relaxed = (cfg.get("turnover") is not None) and (used != 1.0) and len(prev) > 0

        keep = np.abs(w) > 1e-8
        both = g.loc[keep].copy()
        both["weight"] = w[keep]
        both["target_month"] = month
        holdings.append(both[["target_month", "permno", "ticker", "company_name", "sector", "weight", TARGET_COL, "pred",
                              "raw_me", "raw_beta_60m", "raw_betabab_1260d"]])
        prev = pd.Series(both["weight"].to_numpy() * (1.0 + both[TARGET_COL].to_numpy()), index=both["permno"].to_numpy())

        long_mask = both["weight"] > 0
        sec_net = both.groupby("sector")["weight"].sum().abs()
        monthly.append({
            "target_month": month,
            "port_excess_ret": (both["weight"] * both[TARGET_COL]).sum(),
            "long_leg_ret": (both.loc[long_mask, "weight"] * both.loc[long_mask, TARGET_COL]).sum(),
            "short_leg_ret": (both.loc[~long_mask, "weight"] * both.loc[~long_mask, TARGET_COL]).sum(),
            "gross_exposure": both["weight"].abs().sum(), "net_exposure": both["weight"].sum(),
            "beta_exposure": (both["weight"] * both["raw_beta_60m"]).sum(),
            "betabab_exposure": (both["weight"] * both["raw_betabab_1260d"]).sum(),
            "max_abs_sector_net": float(sec_net.max()),
            "max_sector_gross_share": float(both.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).max() / GROSS),
            "turnover_cap_relaxed": bool(relaxed), "turnover_relax_mult": (np.nan if used is None else float(used)),
            "n_long": int(long_mask.sum()), "n_short": int((~long_mask).sum()), "n_positions": int(keep.sum()),
        })
    return pd.concat(holdings, ignore_index=True), pd.DataFrame(monthly).sort_values("target_month").reset_index(drop=True)


def trade_frame(holdings_df):
    """Per-month traded notional sum|w_new - w_drifted| (multiple of capital; gross book = 2.0), plus the
    tiered transaction cost of it. Prior weights are drifted by realized returns before rebalancing; the
    first month is a full build; names that leave the book are charged at their last-held tier."""
    W = holdings_df.pivot_table(index="target_month", columns="permno", values="weight", fill_value=0.0).sort_index()
    R = holdings_df.pivot_table(index="target_month", columns="permno", values=TARGET_COL, fill_value=0.0).sort_index()
    ME = holdings_df.pivot_table(index="target_month", columns="permno", values="raw_me").sort_index()
    tier_trade = pd.DataFrame(cost_bps(ME.to_numpy())[0], index=ME.index, columns=ME.columns).where(ME.notna())
    tier_trade = tier_trade.fillna(tier_trade.shift(1))
    drifted = (W * (1.0 + R)).shift(1).fillna(0.0)
    trade = (W - drifted).abs()
    return trade.sum(axis=1), (trade * tier_trade.fillna(20.0) / 1e4).sum(axis=1)


def borrow_series(holdings_df):
    s = holdings_df[holdings_df["weight"] < 0]
    _, b = cost_bps(s["raw_me"].to_numpy())
    return (s["weight"].abs() * b / 1e4 / 12.0).groupby(s["target_month"]).sum()


def load_fred_series():
    tb3ms = pd.read_csv(CACHE / "TB3MS.csv", parse_dates=["observation_date"])
    tb3ms = tb3ms.rename(columns={"observation_date": "month", "TB3MS": "tb3ms_annual_pct"})
    tb3ms["month"] = tb3ms["month"].values.astype("datetime64[M]")
    tb3ms["month"] = pd.to_datetime(tb3ms["month"]) + pd.offsets.MonthEnd(0)
    tb3ms["rf_monthly"] = tb3ms["tb3ms_annual_pct"] / 100.0 / 12.0

    sp = pd.read_csv(CACHE / "SP500.csv", parse_dates=["observation_date"])
    sp = sp.rename(columns={"observation_date": "date", "SP500": "close"}).dropna(subset=["close"])
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
    top10 = holdings_df.groupby("target_month")["weight"].apply(lambda w: w.abs().nlargest(10).sum() / 2.0)
    return {
        "avg_monthly_turnover": float(turnover.mean()), "min_monthly_turnover": float(turnover.min()),
        "max_monthly_turnover": float(turnover.max()), "avg_position_weight_abs": float(abs_w.mean()),
        "max_position_weight_abs": float(abs_w.max()), "avg_top10_share_of_gross": float(top10.mean()),
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
        "short_book_avg_market_cap_musd": float(short["me"].mean()),
        "short_book_median_market_cap_musd": float(short["me"].median()),
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
    return {
        "max_drawdown": max_dd, "max_drawdown_peak_date": date_at(peak_i),
        "max_drawdown_trough_date": date_at(trough_i), "max_drawdown_recovery_date": date_at(recovery_i),
        "max_drawdown_peak_to_trough_months": int(trough_i - peak_i),
        "max_drawdown_recovery_months": None if recovery_i is None else int(recovery_i - trough_i),
        "longest_underwater_months": int(runs.max()) if len(runs) else 0, "current_drawdown": float(dd.iloc[-1]),
        "avg_drawdown_when_underwater": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else None,
    }


def compute_performance(stats_df):
    """Returns (results dict, per-month frame). Same metrics as every prior script (experiments/ols/README.md sec 2.6),
    plus rolling-12-month beta to the S&P 500 (docs/FIAM.md sec 12 chart; the financial engineer's
    'no peaks over 1 in the middle' check)."""
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
    ols_res = sm.OLS(y, sm.add_constant(x)).fit()
    alpha_monthly, beta = ols_res.params
    alpha_se, beta_se = ols_res.bse
    alpha_t, beta_t = ols_res.tvalues

    # rolling 12m beta of excess return on S&P excess return
    xs = perf["sp500_ret"] - perf["rf_monthly"]
    perf["rolling_beta_12m"] = perf["port_excess_ret"].rolling(12).cov(xs) / xs.rolling(12).var()
    rb = perf["rolling_beta_12m"].dropna()

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
        "rolling_beta_12m_min": float(rb.min()), "rolling_beta_12m_max": float(rb.max()),
        "rolling_beta_12m_months_above_1": int((rb > 1.0).sum()), "rolling_beta_12m_months_below_minus1": int((rb < -1.0).sum()),
        "rolling_beta_12m_share_abs_above_0.5": float((rb.abs() > 0.5).mean()), "rolling_beta_12m_n": int(len(rb)),
        "corr_with_sp500": float(perf["port_excess_ret"].corr(perf["sp500_ret"])),
        "avg_gross_exposure": float(perf["gross_exposure"].mean()), "max_gross_exposure": float(perf["gross_exposure"].max()),
        "avg_net_exposure": float(perf["net_exposure"].mean()), "min_net_exposure": float(perf["net_exposure"].min()),
        "max_net_exposure": float(perf["net_exposure"].max()),
        "avg_n_positions": float(perf["n_positions"].mean()), "min_n_positions": int(perf["n_positions"].min()),
        "max_n_positions": int(perf["n_positions"].max()),
        "avg_n_long": float(perf["n_long"].mean()), "avg_n_short": float(perf["n_short"].mean()),
        "best_month": {"date": str(best_month["target_month"].date()), "ret": float(best_month["port_excess_ret"])},
        "worst_month": {"date": str(worst_month["target_month"].date()), "ret": float(worst_month["port_excess_ret"])},
        "calendar_year_returns": {int(k): float(v) for k, v in calendar_year_of("port_excess_ret").items()},
        "calendar_year_returns_benchmark": {int(k): float(v) for k, v in calendar_year_of("benchmark_monthly").items()},
        "calendar_year_returns_sp500": {int(k): float(v) for k, v in calendar_year_of("sp500_ret").items()},
        "long_leg_avg_monthly_ret": float(perf["long_leg_ret"].mean()), "long_leg_cagr": float(long_leg_cagr),
        "short_leg_avg_monthly_ret": float(perf["short_leg_ret"].mean()), "short_leg_cagr": float(short_leg_cagr),
        **dd_stats, "sp500_drawdown": sp500_dd_stats,
    }
    return results, perf


def evaluate_variant(preds, name, cfg, tag, arm):
    """Build one portfolio variant, report it gross and net of the tiered costs, write its files."""
    holdings, stats = build_portfolio(preds, cfg)
    gross_perf, gross_frame = compute_performance(stats)

    trade, tcost = trade_frame(holdings)
    borrow = borrow_series(holdings)
    net_stats = stats.copy()
    net_stats["trade_cost"] = tcost.reindex(net_stats["target_month"]).to_numpy()
    net_stats["borrow_cost"] = borrow.reindex(net_stats["target_month"]).fillna(0.0).to_numpy()
    net_stats["traded_notional"] = trade.reindex(net_stats["target_month"]).to_numpy()
    net_stats["port_excess_ret"] = stats["port_excess_ret"].to_numpy() - net_stats["trade_cost"] - net_stats["borrow_cost"]
    net_perf, net_frame = compute_performance(net_stats)

    frame = gross_frame.copy()
    frame["trade_cost"] = net_stats["trade_cost"].to_numpy()
    frame["borrow_cost"] = net_stats["borrow_cost"].to_numpy()
    frame["traded_notional"] = net_stats["traded_notional"].to_numpy()
    frame["net_port_excess_ret"] = net_stats["port_excess_ret"].to_numpy()
    frame["net_rolling_beta_12m"] = net_frame["rolling_beta_12m"].to_numpy()
    frame.to_csv(OUT / f"portfolio_returns_{name}_{tag}_{arm}.csv", index=False)
    holdings.to_csv(OUT / f"portfolio_holdings_{name}_{tag}_{arm}.csv", index=False)

    tn = net_stats["traded_notional"].to_numpy()
    res = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "gross": gross_perf, "net": net_perf,
        "avg_beta_exposure_at_formation": float(stats["beta_exposure"].mean()),
        "max_abs_beta_exposure_at_formation": float(stats["beta_exposure"].abs().max()),
        "max_abs_betabab_exposure_at_formation": float(stats["betabab_exposure"].abs().max()),
        "avg_abs_betabab_exposure": float(stats["betabab_exposure"].abs().mean()),
        "max_abs_sector_net": float(stats["max_abs_sector_net"].max()),
        "max_sector_gross_share": float(stats["max_sector_gross_share"].max()),
        "months_turnover_cap_relaxed": int(stats["turnover_cap_relaxed"].sum()),
        "traded_notional_avg_x_capital_ex_initial": float(np.mean(tn[1:])),
        "traded_notional_min_x_capital_ex_initial": float(np.min(tn[1:])),
        "traded_notional_max_x_capital_ex_initial": float(np.max(tn[1:])),
        "drift_adjusted_one_way_turnover_pct_of_gross": float(np.mean(tn[1:]) / (2.0 * GROSS)),
        "avg_trade_cost_bp_of_nav_per_month": float(1e4 * net_stats["trade_cost"].mean()),
        "avg_borrow_cost_bp_of_nav_per_month": float(1e4 * net_stats["borrow_cost"].mean()),
        **compute_turnover_and_concentration(holdings), **compute_short_book_characteristics(holdings),
    }
    return res


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def summary_row(tag, arm, name, res, model_stats):
    g, n = res["gross"], res["net"]
    return {
        "model": tag, "arm": arm, "portfolio": name, **model_stats,
        "ir_gross": g["information_ratio"], "ir_net": n["information_ratio"],
        "sharpe_gross": g["sharpe_ratio"], "sharpe_net": n["sharpe_ratio"],
        "cagr_gross_pct": 100 * g["annualized_return_geo_cagr"], "cagr_net_pct": 100 * n["annualized_return_geo_cagr"],
        "alpha_t_gross": g["alpha_tstat"], "alpha_t_net": n["alpha_tstat"],
        "beta": g["beta"], "beta_t": g["beta_tstat"],
        "roll_beta_min": g["rolling_beta_12m_min"], "roll_beta_max": g["rolling_beta_12m_max"],
        "roll_beta_months_gt1": g["rolling_beta_12m_months_above_1"],
        "max_dd_gross_pct": 100 * g["max_drawdown"], "max_dd_net_pct": 100 * n["max_drawdown"],
        "hit_rate_net": n["hit_rate_active"],
        "turnover_one_way_pct_gross": 100 * res["drift_adjusted_one_way_turnover_pct_of_gross"],
        "turnover_project_convention_pct": 100 * res["avg_monthly_turnover"],
        "n_pos_min": g["min_n_positions"], "n_pos_max": g["max_n_positions"],
        "trade_cost_bp_nav_month": res["avg_trade_cost_bp_of_nav_per_month"],
        "borrow_cost_bp_nav_month": res["avg_borrow_cost_bp_of_nav_per_month"],
        "short_median_mcap_musd": res["short_book_median_market_cap_musd"],
        "short_share_lt_2b": res["short_book_share_below_2000m_cap"],
        "max_abs_sector_net": res["max_abs_sector_net"], "max_sector_gross_share": res["max_sector_gross_share"],
        "months_cap_relaxed": res["months_turnover_cap_relaxed"],
    }


# ---------------------------------------------------------------------------
# Inlined from largecap.py (copied verbatim so this file runs on its own; only the parts used here)
# ---------------------------------------------------------------------------


def monthly_rank_ic(months, pred, y) -> pd.Series:
    d = pd.DataFrame({"m": months, "p": pred, "y": y})
    g = d.groupby("m")
    d["p"] = g["p"].rank()
    d["y"] = g["y"].rank()
    g = d.groupby("m")
    d["p"] = d["p"] - g["p"].transform("mean")
    d["y"] = d["y"] - g["y"].transform("mean")
    num = (d["p"] * d["y"]).groupby(d["m"]).sum()
    den = np.sqrt((d["p"] ** 2).groupby(d["m"]).sum() * (d["y"] ** 2).groupby(d["m"]).sum())
    return num / den


# ---------------------------------------------------------------------------
# Pre-registered factor set: (group -> factors with the sign a long-short composite should use).
# Signs come from the direction of each characteristic's documented effect in docs/FACTORS.md sec 2-16 / the
# cited literature, fixed before any return was looked at. + = higher is better (long), - = higher is worse.
# ---------------------------------------------------------------------------
FACTOR_GROUPS = {
    "value": {"be_me": +1, "ni_me": +1, "fcf_me": +1},
    "profitability": {"gp_at": +1, "ni_be": +1, "ebit_sale": +1},
    "investment_issuance_accruals": {"at_gr1": -1, "chcsho_12m": -1, "oaccruals_at": -1},
    "quality": {"qmj": +1, "f_score": +1},
    "surprise": {"niq_su": +1, "saleq_su": +1},
    "volatility_beta": {"ivol_capm_21d": -1, "rmax5_21d": -1, "betabab_1260d": -1},
    "liquidity": {"ami_126d": +1, "turnover_126d": -1},
}

SIGNS = {f: s for g in FACTOR_GROUPS.values() for f, s in g.items()}
FEATURES = list(SIGNS)
MIN_PRICE, MIN_DOLVOL = 5.0, 10_000_000.0


def make_variants(floor):
    base = dict(min_dolvol=MIN_DOLVOL, min_price=MIN_PRICE, min_mcap=float(floor),
                beta_cols=("raw_beta_60m", "raw_betabab_1260d"), max_weight=0.01,
                sector_net=0.05, sector_gross=0.70, turnover=0.10)
    return {
        "lc_free": {**base, "turnover": None},
        "lc_t20": {**base, "turnover": 0.20},
        "lc_t10": base,
        "lc_t10_w05": {**base, "max_weight": 0.005},
    }


def universe_mask(f: pd.DataFrame, floor) -> pd.Series:
    """Rows a large-cap LP can hold, from characteristic-month raw values (known at the rebalance date)."""
    return ((f["raw_prc"].abs() >= MIN_PRICE) & (f["raw_me"] >= floor) & (f["raw_dolvol_126d"] >= MIN_DOLVOL)
            & f["raw_beta_60m"].notna() & f["raw_betabab_1260d"].notna())


def universe_ic(preds, floor) -> pd.Series:
    p = preds[universe_mask(preds, floor)]
    return monthly_rank_ic(p["target_month"].to_numpy(), p["pred"].to_numpy(), p[TARGET].to_numpy())


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def decile_table(preds, floor) -> pd.Series:
    """Mean next-month excess return (%) by within-month prediction decile, universe rows only, months pooled
    with equal month weight."""
    p = preds[universe_mask(preds, floor)].copy()
    p["dec"] = p.groupby("target_month")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    return 100 * p.groupby(["target_month", "dec"])[TARGET].mean().groupby("dec").mean()


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    return {"mean_diff": float(d.mean()), "t": tstat(d), "share_a_better": float((d > 0).mean())}


OUTPUT.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")
TARGET = TARGET_COL
GROUPS = list(FACTOR_GROUPS)
MEAN_SHRINK, SLEEVE_CAP = 0.5, 0.35
ARMS = ("tpa_eq", "tpa_erc", "tpa_ms")


# ---------------------------------------------------------------------------
# Sleeve signals and returns (universe rows only)
# ---------------------------------------------------------------------------


def sleeve_frames(ctx, mask_all: np.ndarray):
    """Returns (Z_full [N_all x 7, zeros outside the universe], sleeve_returns [target month x 7])."""
    idx = np.flatnonzero(mask_all)
    D = pd.DataFrame(ctx.Xall[idx][:, ctx.cols(FEATURES)], columns=FEATURES)
    month = pd.Series(ctx.tm[idx], name="month")
    R = (D.groupby(month.to_numpy())[FEATURES].rank(pct=True) * 2 - 1) * pd.Series(SIGNS)
    G = pd.DataFrame({g: R[list(fs)].mean(axis=1) for g, fs in FACTOR_GROUPS.items()})
    G = G - G.groupby(month.to_numpy()).transform("mean")
    G = G / G.abs().groupby(month.to_numpy()).transform("mean")
    Z_full = np.zeros((len(ctx.Xall), len(GROUPS)))
    Z_full[idx] = G.to_numpy()
    y = pd.Series(ctx.y[idx])
    num = G.mul(y.to_numpy(), axis=0).groupby(month.to_numpy()).sum()
    den = G.abs().groupby(month.to_numpy()).sum()
    return Z_full, (num / den)


# ---------------------------------------------------------------------------
# Allocation rules
# ---------------------------------------------------------------------------


def lw_cov(S: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(S.to_numpy()).covariance_


def risk_contributions(w, cov):
    mrc = cov @ w
    rc = w * mrc
    return rc / rc.sum()


def alloc_eq(S):
    return np.full(S.shape[1], 1.0 / S.shape[1])


def alloc_erc(S):
    cov = lw_cov(S)
    n = len(cov)
    # standard ERC objective: min 0.5 w'Cw - (1/n) sum log w  -> equal risk contributions once normalised
    res = minimize(lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n, np.full(n, 1.0 / n),
                   jac=lambda w: cov @ w - 1.0 / (n * w), bounds=[(1e-8, None)] * n, method="L-BFGS-B")
    w = res.x / res.x.sum()
    return w


def alloc_ms(S):
    cov = lw_cov(S)
    mu = S.mean().to_numpy()
    mu = (1 - MEAN_SHRINK) * mu + MEAN_SHRINK * mu.mean()
    n = len(mu)
    if not (mu > 0).any():
        return alloc_eq(S)
    res = minimize(lambda w: -(w @ mu) / np.sqrt(w @ cov @ w), np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, SLEEVE_CAP)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.clip(res.x, 0.0, None)
    return w / w.sum() if res.success and w.sum() > 0 else alloc_eq(S)


ALLOC = {"tpa_eq": alloc_eq, "tpa_erc": alloc_erc, "tpa_ms": alloc_ms}


def run_arms(ctx, Z_full, sleeves, floor, tag, args, ic_store):
    variants = make_variants(floor)
    weights_rows, results, rows = [], {}, []
    for arm in ARMS:
        print(f"\n{'=' * 78}\nTPA | ARM {arm} | floor ${floor:,.0f}M | 7 factor-group sleeves\n{'=' * 78}", flush=True)
        t0 = time.time()
        pred = np.zeros(len(ctx.meta))
        for fold in ctx.folds:
            Y = fold["year"]
            S = sleeves[sleeves.index < pd.Timestamp(f"{Y}-01-01")]
            w = ALLOC[arm](S)
            rc = risk_contributions(w, lw_cov(S))
            weights_rows.append({"arm": arm, "test_year": Y, "n_sleeve_months": len(S),
                                 **{f"w_{g}": float(x) for g, x in zip(GROUPS, w)},
                                 **{f"rc_{g}": float(x) for g, x in zip(GROUPS, rc)}})
            pred[fold["sl"]] = Z_full[fold["te"]] @ w
        wdf = pd.DataFrame([r for r in weights_rows if r["arm"] == arm])
        print("  sleeve weights by test year:\n" + wdf[["test_year"] + [f"w_{g}" for g in GROUPS]].round(3).to_string(index=False), flush=True)
        preds = preds_frame(ctx, pred)
        preds.to_csv(OUT / f"oos_predictions_{tag}_{arm}.csv", index=False)
        ic = universe_ic(preds, floor)
        ic_store[arm] = ic
        dec = decile_table(preds, floor)
        u = preds[universe_mask(preds, floor)]
        model_stats = {
            "universe_rank_ic": float(ic.mean()), "universe_rank_ic_t": tstat(ic),
            "universe_rank_ic_share_positive": float((ic > 0).mean()),
            "universe_ic_by_year": {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()},
            "decile_top_minus_bottom_pct_per_month": float(dec.loc[10] - dec.loc[1]),
            "pred_rank_autocorr": pred_rank_autocorr(u),
        }
        print(f"  universe: {json.dumps(_clean(model_stats))}", flush=True)
        out = {"model_stats": model_stats, "variants": {}}
        for name, cfg in variants.items():
            res = evaluate_variant(preds, name, cfg, tag, arm)
            out["variants"][name] = res
            rows.append(summary_row(tag, arm, name, res, {k: v for k, v in model_stats.items() if not isinstance(v, dict)}))
            g, n = res["gross"], res["net"]
            print(f"  [{name:10s}] gross IR {g['information_ratio']:.2f} / net IR {n['information_ratio']:.2f} | "
                  f"beta {g['beta']:+.2f} (t {g['beta_tstat']:+.2f}) roll12 [{g['rolling_beta_12m_min']:+.2f},"
                  f"{g['rolling_beta_12m_max']:+.2f}] | turnover {100 * res['drift_adjusted_one_way_turnover_pct_of_gross']:.0f}% | "
                  f"maxDD {100 * g['max_drawdown']:.0f}% | short med mcap ${res['short_book_median_market_cap_musd']:,.0f}M", flush=True)
        results[arm] = out
        print(f"  arm done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return results, rows, pd.DataFrame(weights_rows)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=2000.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    floor = args.floor
    tag = "lc" if floor == 2000.0 else f"lc{int(floor)}"
    if args.smoke:
        OUT = OUTPUT / "_smoke_tpa"
        OUT.mkdir(exist_ok=True)

    print("Loading model table...", flush=True)
    df, stock_vars = load_model_table()
    if args.smoke:
        df = df[df["permno"] % 3 == 0].reset_index(drop=True)
    df = cross_sectional_rank_transform(df, stock_vars)
    ctx = Context(df, stock_vars, max_folds=2 if args.smoke else None)
    mask_all = universe_mask(df, floor).to_numpy()
    del df
    Z_full, sleeves = sleeve_frames(ctx, mask_all)
    print(f"  sleeve return months: {len(sleeves)} ({sleeves.index.min().date()} .. {sleeves.index.max().date()})", flush=True)
    pre = sleeves[sleeves.index < pd.Timestamp("2021-01-01")]
    print("  pre-2021 sleeve Sharpe (ann.): " + ", ".join(f"{g[:8]} {np.sqrt(12) * pre[g].mean() / pre[g].std():+.2f}" for g in GROUPS), flush=True)

    ic_store = {}
    results, rows, wdf = run_arms(ctx, Z_full, sleeves, floor, tag, args, ic_store)
    wdf.to_csv(OUT / f"tpa_weights_{tag}.csv", index=False)
    pd.DataFrame(ic_store).rename_axis("target_month").to_csv(OUT / f"tpa_monthly_ic_{tag}.csv")
    paired_ic = {a: paired(ic_store[a], ic_store["tpa_eq"]) for a in ("tpa_erc", "tpa_ms")}
    print("\nPaired monthly universe-IC differences vs tpa_eq:", json.dumps(paired_ic, indent=1), flush=True)
    (OUT / f"tpa_results_{tag}.json").write_text(json.dumps(_clean({"results": results, "paired_ic_vs_eq": paired_ic}), indent=2))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / f"tpa_summary_{tag}.csv", index=False)
    cols = ["arm", "portfolio", "ir_gross", "ir_net", "sharpe_net", "cagr_net_pct", "beta", "roll_beta_min", "roll_beta_max",
            "turnover_one_way_pct_gross", "max_dd_net_pct", "universe_rank_ic", "universe_rank_ic_t"]
    print(f"\n{'=' * 78}\nSUMMARY TPA (floor ${floor:,.0f}M)\n{'=' * 78}")
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
