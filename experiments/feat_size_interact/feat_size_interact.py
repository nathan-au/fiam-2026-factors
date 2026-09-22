"""
FIAM 2026 - Size-conditional composites: within-size-tercile ranking and tercile-specific group weights (feat_size_interact.py).

Research origin: docs/RESEARCH.md Part I sec 3.3 item 6 (characteristic x size interactions: trees can find them, a linear composite cannot - put size-tercile-conditional
composites into the composite arm) and sec 3.2 (Howard: size buckets matter).

HYPOTHESIS. The composite's value/investment tilt works differently across size terciles of the >= $2B universe. (a) Ranking every factor within size terciles (a size-neutral
composite) or (b) weighting the seven factor groups differently by tercile, with weights fitted only on earlier months (expanding window, weight = max(group IC, 0)), raises the universe IC.
Both are REPLACEMENT composites compared directly with the frozen composite (paired monthly IC, LP `lc_t10`). Variant (b) is the only fitted quantity; it uses months before the test year only.
PRE-REGISTERED KILL RULE: paired IC vs the frozen composite has t < 1 (or IR gross not above the composite's at equal turnover) -> kill.

DESIGN. Frozen experiments/largecap harness (copied in below); universe = price >= $5, mcap >= $2B, $10M dollar volume, both betas observed. Size tercile = tercile of market cap within the universe month.

Run:  .venv/bin/python experiments/feat_size_interact/feat_size_interact.py [--smoke]
Outputs (output/): results.json, summary.csv, monthly_ic.csv, run.log, portfolio_{holdings,returns}_lc_t10_lc_<arm>.csv
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # experiments/feat_size_interact/
OUTPUT = HERE / "output"


# ---------------------------------------------------------------------------
# Inlined from et.py (copied verbatim so this file runs on its own; only the parts used here)
# ---------------------------------------------------------------------------


import statsmodels.api as sm
from scipy import sparse
from scipy.optimize import linprog
from sklearn.ensemble import ExtraTreesRegressor
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
SEED = 42


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


def pred_rank_autocorr(preds, min_dolvol=10_000_000.0) -> float:
    """Mean month-over-month Spearman correlation of the prediction ranking on the
    liquid universe -- a stability measure (low = the ranking reshuffles = high turnover)."""
    p = preds[preds["raw_dolvol_126d"] >= min_dolvol]
    R = p.pivot(index="target_month", columns="permno", values="pred").sort_index().rank(axis=1)
    vals = [R.iloc[i].corr(R.iloc[i - 1]) for i in range(1, len(R))]
    return float(np.nanmean(vals))


def run_walk_forward(ctx: Context, features: list[str], label: str, winsor=True, train_universe="all"):
    """fit_fold(Xtr, ytr_fit, Xva) is supplied by each model script and returns a list of candidate
    dicts {params, val_pred, predict(X)->pred, importance}. The candidate with the highest
    VALIDATION mean rank IC is used on the test year. Test data never influences selection."""
    X = ctx.Xall[:, ctx.cols(features)]
    pred = np.zeros(len(ctx.meta))
    fold_info, imps = [], []
    for fold in ctx.folds:
        t0 = time.time()
        tr, va = fold["tr"], fold["va"]
        if train_universe == "tradeable":
            tr, va = tr[ctx.tradeable[tr]], va[ctx.tradeable[va]]
        Xtr, ytr = X[tr], ctx.y[tr]
        Xva, yva = X[va], ctx.y[va]
        ytr_fit = np.clip(ytr, *np.quantile(ytr, [0.01, 0.99])) if winsor else ytr
        months_va = ctx.tm[va]
        cands = fit_fold(Xtr, ytr_fit, Xva)
        for c in cands:
            c["val_ic"] = float(monthly_rank_ic(months_va, c["val_pred"], yva).mean())
            c["val_mse"] = float(np.mean((yva - c["val_pred"]) ** 2))
        best = max(cands, key=lambda c: c["val_ic"])
        pred[fold["sl"]] = best["predict"](X[fold["te"]])
        imps.append(best["importance"])
        fold_info.append({
            "test_year": fold["year"], "chosen": best["params"], "val_ic": best["val_ic"], "val_mse": best["val_mse"],
            "n_train": len(tr), "n_val": len(va),
            "candidates": [{**c["params"], "val_ic": c["val_ic"], "val_mse": c["val_mse"]} for c in cands],
        })
        print(f"  [{label} fold {fold['year']}] chosen={best['params']} val_ic={best['val_ic']:.4f} "
              f"(candidates: {len(cands)}, val_ic range {min(c['val_ic'] for c in cands):.4f}..{max(c['val_ic'] for c in cands):.4f}) "
              f"{time.time() - t0:.0f}s", flush=True)
    imp = pd.Series(np.mean(imps, axis=0), index=features).sort_values(ascending=False)
    return pred, imp, fold_info


def preds_frame(ctx: Context, pred_vec) -> pd.DataFrame:
    out = ctx.meta.copy()
    out["pred"] = pred_vec
    return out[(out["target_month"] >= OOS_START) & (out["target_month"] <= OOS_END)].reset_index(drop=True)


def oos_r2(actual, predicted):
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum(actual**2)


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


# Split thresholds are drawn at random (not optimized) and trees use every training row (no bootstrap), so
# individual trees fit less noise than a RF's: the regularization is in the randomness. Depth is capped only
# through min_samples_leaf; max_features trades tree diversity against per-tree strength.
GRID = [{"max_features": mf, "min_samples_leaf": leaf} for mf in ("sqrt", 0.33) for leaf in (100, 300, 1000)]

N_ESTIMATORS = 300


def fit_fold(Xtr, ytr, Xva):
    cands = []
    for p in GRID:
        m = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
            bootstrap=False, n_jobs=-1, random_state=SEED,
        ).fit(Xtr, ytr)
        cands.append({"params": dict(p), "val_pred": m.predict(Xva), "predict": m.predict,
                      "importance": m.feature_importances_})
    return cands


OUTPUT.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")
TARGET = TARGET_COL

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
HEADLINE_VARIANT = "lc_t10"
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


# ---------------------------------------------------------------------------
# Composite (no fitting): re-rank each factor WITHIN the universe each month, flip by sign, average within group,
# then average the groups equally.
# ---------------------------------------------------------------------------


def composite_scores(ctx, meta, floor) -> np.ndarray:
    Xt = ctx.Xall[np.concatenate([f["te"] for f in ctx.folds])]  # rows are in meta order
    D = pd.DataFrame(Xt[:, ctx.cols(FEATURES)], columns=FEATURES)
    D["month"] = meta["target_month"].to_numpy()
    m = universe_mask(meta, floor).to_numpy()
    sub = D[m]
    R = sub.groupby("month")[FEATURES].rank(pct=True) * 2 - 1
    R = R * pd.Series(SIGNS)
    grp = pd.DataFrame({g: R[list(fs)].mean(axis=1) for g, fs in FACTOR_GROUPS.items()})
    score = grp.mean(axis=1)
    out = np.zeros(len(meta))
    out[score.index.to_numpy()] = score.to_numpy()
    return out


# ---------------------------------------------------------------------------
# Universe-only diagnostics
# ---------------------------------------------------------------------------


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




# ---------------------------------------------------------------------------
# Experiment code: candidate signal blocks tested ADDITIVELY on top of the frozen composite
# ---------------------------------------------------------------------------
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
FLOOR = 2000.0
TAG = "lc"
HEADLINE_VARIANT = "lc_t10"
N_PERM = 200
PUBLISHED = {"comp_ic": 0.0366, "comp_ir_gross": 0.61, "comp_ir_net": 0.54}  # experiments/largecap/README.md, $2B floor
MONTH_INDEX = pd.date_range("2015-01-31", "2026-08-31", freq="ME")


def load_context(smoke=False):
    df, stock_vars = load_model_table()
    if smoke:
        df = df[df["permno"] % 3 == 0].reset_index(drop=True)
    df = cross_sectional_rank_transform(df, stock_vars)
    ctx = Context(df, stock_vars, max_folds=1 if smoke else None)
    ctx.u = universe_mask(df, FLOOR).to_numpy()
    ctx.raw_me = df["raw_me"].to_numpy()
    ctx.sector = df["sector"].to_numpy()
    ctx.te_idx = np.concatenate([f["te"] for f in ctx.folds])
    ctx.all_permno, ctx.all_eom = df["permno"].to_numpy(), df["eom"].to_numpy()  # identity of every context row
    m = ctx.meta
    m["eom_char"] = pd.to_datetime(m["target_month"]) - pd.offsets.MonthEnd(1)  # characteristic month of the row
    ctx.um = universe_mask(m, FLOOR).to_numpy()  # universe mask on the meta (test) rows
    ctx.permnos = set(df["permno"].unique())
    print(f"  {len(df):,} rows; universe rows {int(ctx.u.sum()):,}; test rows {len(m):,} (universe {int(ctx.um.sum()):,})", flush=True)
    return ctx


def load_panel(cols, ctx=None):
    """Long panel of raw columns from the characteristics file (sorted by permno, eom), same stocks as the context."""
    p = pd.read_parquet(CHARS_FILE, columns=["permno", "eom"] + cols)
    p["eom"] = pd.to_datetime(p["eom"])
    if ctx is not None:
        p = p[p["permno"].isin(ctx.permnos)]
    return p.sort_values(["permno", "eom"]).reset_index(drop=True)


def wide(p, col):
    """(month x permno) matrix on the full monthly calendar; gaps are NaN."""
    return p.pivot(index="eom", columns="permno", values=col).reindex(MONTH_INDEX)


def wide_to_meta(W, meta):
    """Value of a (month x permno) matrix at each meta row's (characteristic month, permno). NaN if absent."""
    r = W.index.get_indexer(meta["eom_char"])
    c = W.columns.get_indexer(meta["permno"])
    out = np.full(len(meta), np.nan)
    ok = (r >= 0) & (c >= 0)
    out[ok] = W.to_numpy()[r[ok], c[ok]]
    return out


def urank(v, ctx, key=None, min_group=5):
    """Within-month (and optional key) percentile rank in (-1, 1] over UNIVERSE rows of the meta frame; 0 for rows outside the
    universe, NaN values (median fill, as everywhere in this project) and groups smaller than min_group."""
    d = pd.DataFrame({"m": ctx.meta["target_month"].to_numpy(), "v": np.asarray(v, dtype=float)})
    grp = ["m"]
    if key is not None:
        d["k"] = np.asarray(key)
        grp.append("k")
    d = d[ctx.um]
    r = d.groupby(grp)["v"].rank(pct=True) * 2 - 1
    if key is not None:
        r = r.where(d.groupby(grp)["v"].transform("count") >= min_group)
    out = np.zeros(len(ctx.meta))
    out[r.index.to_numpy()] = r.fillna(0.0).to_numpy()
    return out


def grp_scores(ctx, rows, key=None):
    """Seven factor-group scores (rows x 7) of the frozen composite for arbitrary row indices of the context, universe rows only
    (0 elsewhere): re-rank each factor within the month (and optional key) over universe rows, flip by sign, average within group."""
    rows = np.asarray(rows)
    Xs = pd.DataFrame(ctx.Xall[rows][:, ctx.cols(FEATURES)], columns=FEATURES)
    keep = ctx.u[rows]
    Xs["m"] = ctx.tm[rows]
    grp = ["m"]
    if key is not None:
        Xs["k"] = np.asarray(key)
        grp.append("k")
    sub = Xs[keep]
    R = sub.groupby(grp)[FEATURES].rank(pct=True) * 2 - 1
    R = R * pd.Series(SIGNS)
    G = pd.DataFrame({g: R[list(fs)].mean(axis=1) for g, fs in FACTOR_GROUPS.items()})
    if key is not None:
        G = G.where(sub.groupby(grp)[FEATURES[0]].transform("count") >= 5)
    out = np.zeros((len(rows), len(FACTOR_GROUPS)))
    out[np.flatnonzero(keep)] = G.fillna(0.0).to_numpy()
    return out


class Base:
    """Frozen composite on the test rows + everything needed to score add-ons against it."""

    def __init__(self, ctx):
        self.ctx = ctx
        m = ctx.meta
        self.months = m["target_month"].to_numpy()
        self.y = m[TARGET_COL].to_numpy()
        self.G7 = grp_scores(ctx, ctx.te_idx)
        self.comp = self.G7.mean(axis=1)
        chk = composite_scores(ctx, m, FLOOR)  # the harness's own composite: must agree with the re-implementation
        assert np.allclose(chk[ctx.um], self.comp[ctx.um], atol=1e-9), "composite re-implementation disagrees with harness"
        me = pd.Series(ctx.meta["raw_me"].to_numpy())[ctx.um]
        self.terc = np.full(len(m), -1)
        t = me.groupby(self.months[ctx.um]).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
        self.terc[t.index.to_numpy()] = t.to_numpy()

    def ic(self, score, mask=None):
        mk = self.ctx.um if mask is None else (self.ctx.um & mask)
        return monthly_rank_ic(self.months[mk], np.asarray(score)[mk], self.y[mk])

    def preds(self, score):
        return preds_frame(self.ctx, np.asarray(score, dtype=float))


def resid_ic(base, b):
    """Rank IC (raw next-month return) of the part of block b that is orthogonal to the composite, month by month
    (OLS of b on comp within the universe; the shared-core diagnostic of docs/RESEARCH.md Part II sec 2.5)."""
    um = base.ctx.um
    d = pd.DataFrame({"m": base.months[um], "b": np.asarray(b)[um], "c": base.comp[um], "y": base.y[um]})
    out, corr = {}, {}
    for m, g in d.groupby("m"):
        c = g["c"].to_numpy() - g["c"].mean()
        bb = g["b"].to_numpy() - g["b"].mean()
        den = float(c @ c)
        r = bb - (float(c @ bb) / den) * c if den > 0 else bb
        if np.std(r) < 1e-12:
            continue
        out[m] = np.corrcoef(rankdata(r), rankdata(g["y"].to_numpy()))[0, 1]
        corr[m] = np.corrcoef(rankdata(g["b"].to_numpy()), rankdata(g["c"].to_numpy()))[0, 1]
    return pd.Series(out), pd.Series(corr)


def perm_null(base, b, n_perm=N_PERM, seed=0):
    """Within-(month, sector) shuffled null for the additive gain in mean IC when block b (already signed, in [-1,1]) is added to the
    composite as an 8th equal-weight group. Shuffling keeps each block's cross-sectional distribution and industry composition but
    breaks its link to the stock (the knockoff-style null of docs/RESEARCH.md Part II H4)."""
    ctx = base.ctx
    U = np.flatnonzero(ctx.um)
    base7 = base.G7.sum(axis=1)[U]
    bU = np.asarray(b)[U]
    mU, sU = base.months[U], ctx.meta["sector"].to_numpy()[U]
    yU = base.y[U]
    mgroups = [np.flatnonzero(mU == m) for m in np.unique(mU)]
    sgroups = [np.flatnonzero((mU == m) & (sU == s)) for m in np.unique(mU) for s in np.unique(sU[mU == m])]
    yr = [rankdata(yU[g]) - rankdata(yU[g]).mean() for g in mgroups]

    def mean_ic(score):
        v = []
        for g, y_ in zip(mgroups, yr):
            r = rankdata(score[g])
            r = r - r.mean()
            v.append((r @ y_) / np.sqrt((r @ r) * (y_ @ y_)))
        return float(np.mean(v))

    obs = mean_ic((base7 + bU) / 8) - mean_ic(base7 / 7)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        bp = bU.copy()
        for g in sgroups:
            bp[g] = bU[rng.permutation(g)]
        null[k] = mean_ic((base7 + bp) / 8) - mean_ic(base7 / 7)
    return {"observed_gain": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.quantile(null, 0.95)), "null_max": float(null.max()),
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "n_perm": n_perm}


def lp_stats(base, score, name, variants=("lc_t10",)):
    preds = base.preds(score)
    cfgs = make_variants(FLOOR)
    out = {}
    for v in variants:
        res = evaluate_variant(preds, v, cfgs[v], TAG, name)
        g, n = res["gross"], res["net"]
        out[v] = {"ir_gross": g["information_ratio"], "ir_net": n["information_ratio"], "sharpe_net": n["sharpe_ratio"],
                  "cagr_net_pct": 100 * n["annualized_return_geo_cagr"], "beta": g["beta"], "beta_t": g["beta_tstat"],
                  "roll_beta_min": g["rolling_beta_12m_min"], "roll_beta_max": g["rolling_beta_12m_max"],
                  "max_dd_net_pct": 100 * n["max_drawdown"], "calendar_year_returns_net": n["calendar_year_returns"],
                  "turnover_one_way_pct_gross": 100 * res["drift_adjusted_one_way_turnover_pct_of_gross"],
                  "short_median_mcap_musd": res["short_book_median_market_cap_musd"]}
    return out


def eval_block(base, name, raw, sign, ic_store, do_lp=True, n_perm=N_PERM):
    """Additive test: composite + block(sign*rank(raw)) as an 8th equal-weight group, vs the frozen 7-group composite."""
    b = sign * urank(raw, base.ctx)
    comb = (base.G7.sum(axis=1) + b) / 8
    ic_b, ic_c, ic_k = base.ic(b), base.ic(base.comp), base.ic(comb)
    ic_store[f"{name}__block_alone"], ic_store[f"{name}__comp_plus_block"] = ic_b, ic_k
    ri, cor = resid_ic(base, b)
    res = {"sign": sign, "n_universe_rows_nonnull": int(np.isfinite(np.asarray(raw, dtype=float))[base.ctx.um].sum()),
           "share_universe_nonnull": float(np.isfinite(np.asarray(raw, dtype=float))[base.ctx.um].mean()),
           "block_ic": float(ic_b.mean()), "block_ic_t": tstat(ic_b),
           "block_ic_by_year": {int(y): float(v) for y, v in ic_b.groupby(ic_b.index.year).mean().items()},
           "comp_ic": float(ic_c.mean()), "comp_plus_block_ic": float(ic_k.mean()), "comp_plus_block_ic_t": tstat(ic_k),
           "paired_gain_vs_comp": paired(ic_k, ic_c),
           "residual_ic_after_orthogonalising_to_comp": float(ri.mean()), "residual_ic_t": tstat(ri),
           "mean_rank_corr_block_vs_comp": float(cor.mean()), "n_perm": n_perm}
    for t, lab in enumerate(("small", "mid", "large")):
        mk = base.terc == t
        bt, ct, kt = base.ic(b, mk), base.ic(base.comp, mk), base.ic(comb, mk)
        res[f"tercile_{lab}"] = {"block_ic": float(bt.mean()), "comp_ic": float(ct.mean()), "comp_plus_block_ic": float(kt.mean()),
                                 "paired_gain_mean": float((kt - ct).mean()), "paired_gain_t": tstat(kt - ct)}
    res["perm_null"] = perm_null(base, b, n_perm) if n_perm else None
    if do_lp:
        res["lp"] = lp_stats(base, comb, f"comp_plus_{name}")
    return res


def eval_alt_composite(base, name, score, ic_store, do_lp=True):
    """A REPLACEMENT composite (not an add-on): compare its IC / portfolio with the frozen composite."""
    ic_a, ic_c = base.ic(score), base.ic(base.comp)
    ic_store[f"{name}"] = ic_a
    res = {"ic": float(ic_a.mean()), "ic_t": tstat(ic_a), "comp_ic": float(ic_c.mean()), "paired_vs_comp": paired(ic_a, ic_c),
           "ic_by_year": {int(y): float(v) for y, v in ic_a.groupby(ic_a.index.year).mean().items()},
           "rank_corr_with_comp": float(pd.Series(rank_corr_by_month(base, score, base.comp)).mean())}
    for t, lab in enumerate(("small", "mid", "large")):
        mk = base.terc == t
        res[f"tercile_{lab}"] = {"ic": float(base.ic(score, mk).mean()), "comp_ic": float(base.ic(base.comp, mk).mean())}
    if do_lp:
        res["lp"] = lp_stats(base, score, name)
    return res


def rank_corr_by_month(base, a, b):
    um = base.ctx.um
    d = pd.DataFrame({"m": base.months[um], "a": np.asarray(a)[um], "b": np.asarray(b)[um]})
    return {m: np.corrcoef(rankdata(g["a"]), rankdata(g["b"]))[0, 1] for m, g in d.groupby("m") if g["a"].std() > 0}


def truncation_test(builder, panel, n_dates=3, seed=0, tol=1e-9):
    """Truncation-invariance (docs/RESEARCH.md Part II sec 2.9): rebuild every feature from ONLY rows with eom <= t and require the
    value at t to equal the value built from the full panel, for every stock, at randomly drawn test-period dates t."""
    full = builder(panel)
    rng = np.random.default_rng(seed)
    cand = [d for d in MONTH_INDEX if d >= pd.Timestamp("2021-01-31") and d <= pd.Timestamp("2026-07-31")]
    dates = [cand[i] for i in rng.choice(len(cand), n_dates, replace=False)]
    worst, n_cmp = {}, 0
    for t in dates:
        trunc = builder(panel[panel["eom"] <= t])
        for k, W in full.items():
            a = W.loc[t].reindex(trunc[k].columns) if t in W.index else None
            bvals = trunc[k].loc[t] if t in trunc[k].index else None
            if a is None or bvals is None:
                continue
            both = a.notna() | bvals.notna()
            diff = np.abs(a[both].to_numpy(dtype=float) - bvals[both].to_numpy(dtype=float))
            diff = np.where(np.isnan(diff), np.inf, diff)  # NaN in exactly one of the two builds counts as a violation
            worst[k] = max(worst.get(k, 0.0), float(np.max(diff)) if len(diff) else 0.0)
            n_cmp += int(both.sum())
    ok = all(v <= tol for v in worst.values())
    return {"passed": bool(ok), "dates": [str(d.date()) for d in dates], "n_values_compared": n_cmp, "max_abs_diff_by_feature": worst}


def ind_mean(ctx, vals, key, min_group=3):
    """Mean of vals within (month, key) over UNIVERSE rows of the meta frame; NaN outside the universe / small groups."""
    d = pd.DataFrame({"m": ctx.meta["target_month"].to_numpy(), "k": np.asarray(key), "v": np.asarray(vals, dtype=float)})[ctx.um]
    g = d.groupby(["m", "k"])["v"]
    mean = g.transform("mean").where(g.transform("count") >= min_group)
    out = np.full(len(ctx.meta), np.nan)
    out[mean.index.to_numpy()] = mean.to_numpy()
    return out


def xall_te(ctx, name):
    """Rank-transformed panel characteristic (all stocks, within month) for the test rows."""
    return ctx.Xall[ctx.te_idx, ctx.col_idx[name]].astype(float)


# ---------------------------------------------------------------------------
# Size-conditional composites. (a) rank every factor within size terciles; (b) tercile-specific group weights fitted ONLY on
# earlier months (expanding window; weight = max(mean monthly group IC, 0), normalised) -- the one place this experiment fits anything.
# ---------------------------------------------------------------------------


def size_tercile_rows(ctx, rows):
    """Within-month size tercile (0 small .. 2 large) of universe rows among `rows` (-1 outside the universe)."""
    d = pd.DataFrame({"m": ctx.tm[rows], "me": ctx.raw_me[rows]})[ctx.u[rows]]
    t = d.groupby("m")["me"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    out = np.full(len(rows), -1)
    out[np.flatnonzero(ctx.u[rows])] = t.to_numpy()
    return out


def make_blocks(ctx):
    te = ctx.te_idx
    terc_te = size_tercile_rows(ctx, te)
    G_te = grp_scores(ctx, te)
    base_c = G_te.mean(axis=1)
    within = grp_scores(ctx, te, key=terc_te).mean(axis=1)  # (a) size-neutral composite
    # (b) tercile-specific weights, refit each test year on all universe months before that year
    fitted = np.zeros(len(te))
    W_used = {}
    for fold in ctx.folds:
        rows = np.flatnonzero(ctx.u & (ctx.tm < np.datetime64(f"{fold['year']}-01-01")))
        rows = rows[np.isfinite(ctx.y[rows])]
        Gtr = grp_scores(ctx, rows)
        ttr = size_tercile_rows(ctx, rows)
        W = np.zeros((3, Gtr.shape[1]))
        for t in range(3):
            for j in range(Gtr.shape[1]):
                k = ttr == t
                W[t, j] = max(float(monthly_rank_ic(ctx.tm[rows][k], Gtr[k, j], ctx.y[rows][k]).mean()), 0.0)
            W[t] = W[t] / W[t].sum() if W[t].sum() > 0 else np.full(Gtr.shape[1], 1.0 / Gtr.shape[1])
        W_used[int(fold["year"])] = W.round(3).tolist()
        sl = np.isin(te, fold["te"])
        for t in range(3):
            k = sl & (terc_te == t)
            fitted[k] = G_te[k] @ W[t]
    print("  fitted tercile weights by test year (rows small/mid/large; cols = groups):", json.dumps(W_used), flush=True)
    ctx.extra_out = {"tercile_ic_weights_by_test_year": W_used, "groups": list(FACTOR_GROUPS)}
    return {}, {"comp_within_size_tercile": within, "comp_tercile_ic_weighted": fitted}, {"passed": None, "note": "no time-series features built"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="plumbing run: 1/3 of stocks, 1 fold, scratch output dir")
    ap.add_argument("--nperm", type=int, default=N_PERM)
    ap.add_argument("--no-lp", action="store_true")
    args = ap.parse_args()
    OUT = OUTPUT / "_smoke" if args.smoke else OUTPUT
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading...", flush=True)
    ctx = load_context(args.smoke)
    base = Base(ctx)
    ic_store = {"comp": base.ic(base.comp)}
    comp_lp = lp_stats(base, base.comp, "comp", variants=("lc_t10", "lc_free")) if not args.no_lp else {}
    print(f"BASELINE composite: IC {ic_store['comp'].mean():.4f} (t {tstat(ic_store['comp']):.2f}); "
          f"lc_t10 IR gross {comp_lp.get('lc_t10', {}).get('ir_gross', float('nan')):.3f} / net "
          f"{comp_lp.get('lc_t10', {}).get('ir_net', float('nan')):.3f}   [published: {PUBLISHED}]", flush=True)

    blocks, alts, trunc = make_blocks(ctx)
    print("Truncation-invariance test:", json.dumps(_clean(trunc)), flush=True)
    results = {"baseline": {"comp_ic": float(ic_store["comp"].mean()), "comp_ic_t": tstat(ic_store["comp"]), "lp": comp_lp,
                            "published_reference": PUBLISHED},
               "truncation_test": trunc, "blocks": {}, "alt_composites": {}, "extra": getattr(ctx, "extra_out", None)}
    rows = []
    for name, spec in blocks.items():
        r = eval_block(base, name, spec["raw"], spec["sign"], ic_store, do_lp=not args.no_lp, n_perm=args.nperm)
        r["note"] = spec.get("note", "")
        results["blocks"][name] = r
        pg = r["paired_gain_vs_comp"]
        lp = r.get("lp", {}).get("lc_t10", {})
        rows.append({"name": name, "kind": "block", "sign": spec["sign"], "coverage": r["share_universe_nonnull"],
                     "block_ic": r["block_ic"], "block_ic_t": r["block_ic_t"], "comp_ic": r["comp_ic"],
                     "comp_plus_block_ic": r["comp_plus_block_ic"], "gain": pg["mean_diff"], "gain_t": pg["t"],
                     "resid_ic": r["residual_ic_after_orthogonalising_to_comp"], "resid_ic_t": r["residual_ic_t"],
                     "corr_with_comp": r["mean_rank_corr_block_vs_comp"],
                     "perm_p": (r["perm_null"] or {}).get("p_value"),
                     "gain_small": r["tercile_small"]["paired_gain_mean"], "gain_mid": r["tercile_mid"]["paired_gain_mean"],
                     "gain_large": r["tercile_large"]["paired_gain_mean"],
                     "ir_gross": lp.get("ir_gross"), "ir_net": lp.get("ir_net"),
                     "d_ir_gross": (lp.get("ir_gross", np.nan) - comp_lp.get("lc_t10", {}).get("ir_gross", np.nan)) if lp else None})
        print(f"  {name:28s} block IC {r['block_ic']:+.4f} (t {r['block_ic_t']:+.2f}) | comp {r['comp_ic']:.4f} -> "
              f"{r['comp_plus_block_ic']:.4f} gain {pg['mean_diff']:+.4f} (paired t {pg['t']:+.2f}) | resid IC "
              f"{r['residual_ic_after_orthogonalising_to_comp']:+.4f} (t {r['residual_ic_t']:+.2f}) | perm p "
              f"{(r['perm_null'] or {}).get('p_value', float('nan')):.3f} | IR gross "
              f"{lp.get('ir_gross', float('nan')):.2f} net {lp.get('ir_net', float('nan')):.2f}", flush=True)
        (OUT / "results.json").write_text(json.dumps(_clean(results), indent=2))
    for name, score in alts.items():
        r = eval_alt_composite(base, name, score, ic_store, do_lp=not args.no_lp)
        results["alt_composites"][name] = r
        lp = r.get("lp", {}).get("lc_t10", {})
        rows.append({"name": name, "kind": "replacement_composite", "block_ic": r["ic"], "block_ic_t": r["ic_t"],
                     "comp_ic": r["comp_ic"], "comp_plus_block_ic": r["ic"], "gain": r["paired_vs_comp"]["mean_diff"],
                     "gain_t": r["paired_vs_comp"]["t"], "corr_with_comp": r["rank_corr_with_comp"],
                     "gain_small": r["tercile_small"]["ic"] - r["tercile_small"]["comp_ic"],
                     "gain_mid": r["tercile_mid"]["ic"] - r["tercile_mid"]["comp_ic"],
                     "gain_large": r["tercile_large"]["ic"] - r["tercile_large"]["comp_ic"],
                     "ir_gross": lp.get("ir_gross"), "ir_net": lp.get("ir_net"),
                     "d_ir_gross": (lp.get("ir_gross", np.nan) - comp_lp.get("lc_t10", {}).get("ir_gross", np.nan)) if lp else None})
        print(f"  {name:28s} IC {r['ic']:.4f} (t {r['ic_t']:.2f}) vs comp {r['comp_ic']:.4f}: paired {r['paired_vs_comp']['mean_diff']:+.4f} "
              f"(t {r['paired_vs_comp']['t']:+.2f}) | IR gross {lp.get('ir_gross', float('nan')):.2f} net {lp.get('ir_net', float('nan')):.2f}", flush=True)
        (OUT / "results.json").write_text(json.dumps(_clean(results), indent=2))

    summ = pd.DataFrame(rows)
    k = max(1, len(blocks))
    if len(summ) and (summ["kind"] == "block").any():
        blk = summ["kind"] == "block"
        summ.loc[blk, "kill_gain_t_lt_1"] = summ.loc[blk, "gain_t"] < 1
        summ.loc[blk, "kill_only_small_tercile"] = (summ.loc[blk, "gain_mid"] <= 0) & (summ.loc[blk, "gain_large"] <= 0)
        summ.loc[blk, "perm_p_bonferroni"] = (summ.loc[blk, "perm_p"] * k).clip(upper=1.0)
        summ.loc[blk, "passes"] = (summ.loc[blk, "gain_t"] >= 2) & (summ.loc[blk, "perm_p_bonferroni"] <= 0.05) & ~summ.loc[blk, "kill_only_small_tercile"].astype(bool)
    summ.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(ic_store).rename_axis("target_month").to_csv(OUT / "monthly_ic.csv")
    (OUT / "results.json").write_text(json.dumps(_clean(results), indent=2))
    print(f"\n{'=' * 78}\nSUMMARY ({k} blocks tested; Bonferroni factor {k})\n{'=' * 78}")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
