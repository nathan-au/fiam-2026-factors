"""
FIAM 2026 - Rank / Gaussian-rank training targets on the large-cap harness (target_rank.py).

Research origin: docs/NEW.md sec 3.1a (Cakici & Zaremba, "Getting the Target Right in Return Prediction": rank targets
roughly doubled return/Sharpe in LARGE-cap universes; magnitude-preserving targets are dominated by tails).

HYPOTHESIS. Training on a within-month rank (uniform in [-1,1]) or Gaussianised rank of next-month excess return, instead of the
harness's 1/99-winsorised raw return, raises the universe rank IC of the same model on the same 18 factors, because the raw
target's variance is dominated by a few extreme names whose behaviour does not generalise (docs/NEGATIVE_RESULT.md).
PRE-REGISTERED KILL RULE (docs/NEW.md sec 4): adopt only if paired monthly universe-IC t >= 2 vs the control on TWO model
families, or IC >= the composite's 0.037 with t >= 2. Otherwise the idea fails.

DESIGN. Universe/factors/folds/model-selection/portfolio LP are the frozen experiments/largecap harness (copied in below).
Control = same model, same 18 factors, 1/99 winsorised raw target (this is largecap.py's `et` arm). Only the fit target changes.
Candidate selection uses validation rank IC against the RAW return for every arm; scoring is raw-return rank IC on
price>=$5, mcap>=$2B, dollar volume>=$10M rows. Families: Extra-Trees (largecap config), forest-style LightGBM.

Run:  .venv/bin/python experiments/target_rank/target_rank.py [--models et,lgbm] [--seed 42] [--smoke]
Outputs (output/): results.json, summary.csv, paired_vs_control.csv, monthly_ic.csv, run.log,
    oos_predictions_universe_<model>__<arm>.csv, portfolio_{holdings,returns}_<variant>_lc_<model>__<arm>.csv
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # experiments/target_rank/
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
# Experiment code: target engineering on the frozen large-cap harness
# ---------------------------------------------------------------------------
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import norm

warnings.filterwarnings("ignore")
HEADLINE_VARIANT = "lc_t10"
FLOOR = 2000.0
TAG = "lc"
# LightGBM configured like a forest (same idea as experiments/lgbm/README.md, smaller min_child_samples because the
# large-cap training set is ~50-150k rows): tiny learning rate, shallow trees, randomized thresholds, strong L2.
LGB_GRID = [{"num_leaves": 7, "min_child_samples": mcs} for mcs in (500, 2000)]
LGB_ROUNDS, LGB_CHECKPOINTS = 300, (100, 200, 300)


def load_context(smoke=False):
    """Frozen-harness data path (rank-transformed 147 characteristics) plus the raw columns the experiments need."""
    df, stock_vars = load_model_table()
    if smoke:
        df = df[df["permno"] % 3 == 0].reset_index(drop=True)
    n0 = len(df)
    df = cross_sectional_rank_transform(df, stock_vars)
    ctx = Context(df, stock_vars, max_folds=1 if smoke else None)
    assert len(df) == n0
    ctx.u = universe_mask(df, FLOOR).to_numpy()  # rows the large-cap LP can hold (train/validation universe)
    ctx.tradeable = ctx.u
    ctx.raw_me = df["raw_me"].to_numpy()
    ctx.sector = df["sector"].to_numpy()
    print(f"  {len(df):,} rows; universe rows {int(ctx.u.sum()):,}; folds {[f['year'] for f in ctx.folds]}", flush=True)
    return ctx


# ---------------------------------------------------------------------------
# Targets. All are computed WITHIN month on universe rows only (a month's cross-section of realised next-month
# returns; a training row's target is realised before its fold's validation window starts, as in the frozen harness).
# ---------------------------------------------------------------------------


def _size_bucket(d):
    return d.groupby("m")["me"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))


def build_targets(ctx, names):
    """Return {name: full-length float array, NaN outside the universe}."""
    u = ctx.u
    d = pd.DataFrame({"m": ctx.tm, "y": ctx.y, "me": ctx.raw_me, "sec": ctx.sector})[u]
    out = {}

    def put(name, s):
        a = np.full(len(ctx.y), np.nan)
        a[s.index.to_numpy()] = s.to_numpy()
        out[name] = a

    g = d.groupby("m")["y"]
    if {"rank_uniform", "rank_gauss", "decile"} & set(names):
        r, n = g.rank(method="average"), g.transform("count")
        put("rank_uniform", (r - 0.5) / n * 2 - 1)
        put("rank_gauss", pd.Series(norm.ppf(((r - 0.5) / n).to_numpy()), index=d.index))
        put("decile", np.floor(g.rank(method="first") / (n + 1e-9) * 10).clip(0, 9))
    if {"demean_size", "demean_size_sector"} & set(names):
        d["sz"] = _size_bucket(d)
        put("demean_size", d["y"] - d.groupby(["m", "sz"])["y"].transform("median"))
        cell = d.groupby(["m", "sz", "sec"])["y"]
        med = np.where(cell.transform("count") >= 5, cell.transform("median"), d.groupby(["m", "sec"])["y"].transform("median"))
        put("demean_size_sector", d["y"] - med)
    if "demean_sector" in names:
        put("demean_sector", d["y"] - d.groupby(["m", "sec"])["y"].transform("median"))
    if "resid_factor" in names:
        # Numerai-style: residual of next-month return on log size, betabab, ivol, 12-1 momentum, sector dummies,
        # fitted cross-sectionally each month on the exposures measured at the characteristic month.
        cols = [ctx.col_idx[c] for c in ("betabab_1260d", "ivol_capm_21d", "ret_12_1")]
        Xe = ctx.Xall[:, cols].astype(np.float64)
        res = pd.Series(np.nan, index=d.index)
        for m, idx in d.groupby("m").groups.items():
            idx = np.asarray(idx)
            secd = pd.get_dummies(d.loc[idx, "sec"], drop_first=True).to_numpy(dtype=float)
            A = np.column_stack([np.ones(len(idx)), np.log(d.loc[idx, "me"].to_numpy()), Xe[idx], secd])
            beta = np.linalg.lstsq(A, d.loc[idx, "y"].to_numpy(), rcond=None)[0]
            res.loc[idx] = d.loc[idx, "y"].to_numpy() - A @ beta
        put("resid_factor", res)
    return out


# ---------------------------------------------------------------------------
# Model families. Each takes (Xtr, ytr, Xva, aux) and returns candidate dicts like the harness's fit_fold.
# aux = {"m_tr": target-month array of the train rows, "w_tr": sample weights or None}
# ---------------------------------------------------------------------------


def fit_et(Xtr, ytr, Xva, aux):
    cands = []
    for p in GRID:
        m = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, max_features=p["max_features"],
                                min_samples_leaf=p["min_samples_leaf"], bootstrap=False, n_jobs=-1,
                                random_state=SEED).fit(Xtr, ytr, sample_weight=aux.get("w_tr"))
        cands.append({"params": dict(p), "val_pred": m.predict(Xva), "predict": m.predict,
                      "importance": m.feature_importances_})
    return cands


def _lgb_gain(m, nf):
    g = m.booster_.feature_importance(importance_type="gain").astype(float)
    return g / g.sum() if g.sum() > 0 else np.full(nf, 1.0 / nf)


def fit_lgbm(Xtr, ytr, Xva, aux, init=None):
    cands, nf = [], Xtr.shape[1]
    for p in LGB_GRID:
        m = lgb.LGBMRegressor(n_estimators=LGB_ROUNDS, learning_rate=0.02, num_leaves=p["num_leaves"],
                              min_child_samples=p["min_child_samples"], colsample_bytree=0.5, subsample=0.5,
                              subsample_freq=1, extra_trees=True, reg_lambda=100.0, max_bin=63, n_jobs=-1,
                              random_state=SEED, verbose=-1).fit(Xtr, ytr, sample_weight=aux.get("w_tr"))
        imp = _lgb_gain(m, nf)
        for k in LGB_CHECKPOINTS:
            cands.append({"params": {**p, "n_trees": k}, "val_pred": m.predict(Xva, num_iteration=k),
                          "predict": (lambda X, m=m, k=k: m.predict(X, num_iteration=k)), "importance": imp})
    return cands


FITTERS = {"et": fit_et, "lgbm": fit_lgbm}


def run_wf(ctx, features, y_fit, model, label, winsor=True, fitter=None, weights=None, train_mask=None):
    """Walk-forward exactly like the frozen harness's run_walk_forward, except (a) the fit target y_fit can differ from the
    raw return, and (b) validation rank IC (candidate selection) is ALWAYS computed against the raw next-month return."""
    fitter = fitter or FITTERS[model]
    X = ctx.Xall[:, ctx.cols(features)]
    pred = np.zeros(len(ctx.meta))
    fold_info, imps = [], []
    mask = ctx.u if train_mask is None else train_mask
    for fold in ctx.folds:
        t0 = time.time()
        tr, va = fold["tr"], fold["va"]
        tr, va = tr[mask[tr]], va[mask[va]]
        tr = tr[np.isfinite(y_fit[tr])]
        ytr = y_fit[tr]
        if winsor:
            ytr = np.clip(ytr, *np.quantile(ytr, [0.01, 0.99]))
        aux = {"m_tr": ctx.tm[tr], "w_tr": None if weights is None else weights[tr]}
        cands = fitter(X[tr], ytr, X[va], aux)
        for c in cands:
            c["val_ic"] = float(monthly_rank_ic(ctx.tm[va], c["val_pred"], ctx.y[va]).mean())
        best = max(cands, key=lambda c: c["val_ic"])
        pred[fold["sl"]] = best["predict"](X[fold["te"]])
        imps.append(best["importance"])
        fold_info.append({"test_year": fold["year"], "chosen": best["params"], "val_ic": best["val_ic"],
                          "n_train": len(tr), "n_val": len(va),
                          "val_ic_range": [min(c["val_ic"] for c in cands), max(c["val_ic"] for c in cands)]})
        print(f"  [{label} fold {fold['year']}] chosen={best['params']} val_ic={best['val_ic']:.4f} "
              f"(range {fold_info[-1]['val_ic_range'][0]:.4f}..{fold_info[-1]['val_ic_range'][1]:.4f}) "
              f"n_tr={len(tr):,} {time.time() - t0:.0f}s", flush=True)
    imp = pd.Series(np.mean(imps, axis=0), index=features).sort_values(ascending=False)
    return pred, imp, fold_info


# ---------------------------------------------------------------------------
# Scoring an arm: universe rank IC (raw return), decile spread, portfolio variants
# ---------------------------------------------------------------------------


def score_arm(ctx, pred_vec, arm_key, variants, out_dir, extra_ic=None):
    preds = preds_frame(ctx, pred_vec)
    keep = preds[universe_mask(preds, FLOOR)][["permno", "target_month", "pred", TARGET_COL]]
    keep.to_csv(out_dir / f"oos_predictions_universe_{arm_key}.csv", index=False)
    ic = universe_ic(preds, FLOOR)
    u = preds[universe_mask(preds, FLOOR)]
    dec = decile_table(preds, FLOOR)
    stats = {"universe_rank_ic": float(ic.mean()), "universe_rank_ic_t": tstat(ic),
             "share_months_ic_positive": float((ic > 0).mean()),
             "ic_by_year": {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()},
             "d10_minus_d1_pct_per_month": float(dec.loc[10] - dec.loc[1]),
             "pred_rank_autocorr": pred_rank_autocorr(u),
             "n_nan_pred": int(np.isnan(pred_vec).sum())}
    rows = []
    for name, cfg in variants.items():
        res = evaluate_variant(preds, name, cfg, TAG, arm_key)
        g, n = res["gross"], res["net"]
        stats[name] = {"ir_gross": g["information_ratio"], "ir_net": n["information_ratio"],
                       "sharpe_net": n["sharpe_ratio"], "cagr_net_pct": 100 * n["annualized_return_geo_cagr"],
                       "beta": g["beta"], "beta_t": g["beta_tstat"], "roll_beta_min": g["rolling_beta_12m_min"],
                       "roll_beta_max": g["rolling_beta_12m_max"], "max_dd_net_pct": 100 * n["max_drawdown"],
                       "turnover_one_way_pct_gross": 100 * res["drift_adjusted_one_way_turnover_pct_of_gross"],
                       "short_median_mcap_musd": res["short_book_median_market_cap_musd"]}
    return stats, ic, preds


# Arms: name -> fit target. `control` is the frozen harness's winsorised raw return.
ARMS = {
    "control": {"target": "raw", "winsor": True},
    "rank_uniform": {"target": "rank_uniform", "winsor": False},
    "rank_gauss": {"target": "rank_gauss", "winsor": False},
}
ALT_SCORES = []  # extra targets to score IC against (none here: only raw return counts)
MODELS_DEFAULT = "et,lgbm"




# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    global OUT, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=MODELS_DEFAULT)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="plumbing run: 1/3 of stocks, 1 fold, scratch output dir")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    SEED = args.seed
    OUT = OUTPUT / "_smoke" if args.smoke else OUTPUT
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading...", flush=True)
    ctx = load_context(args.smoke)
    arms = args.arms.split(",")
    needed = {ARMS[a]["target"] for a in arms} - {"raw"}
    targets = build_targets(ctx, needed | set(ALT_SCORES))
    te_idx = np.concatenate([f["te"] for f in ctx.folds])
    for k in ALT_SCORES:
        ctx.meta[f"alt_{k}"] = targets[k][te_idx]
    variants = {k: v for k, v in make_variants(FLOOR).items() if k in ("lc_t10", "lc_free")}

    results, ics = {}, {}
    for model in args.models.split(","):
        for arm in arms:
            key = f"{model}__{arm}"
            spec = ARMS[arm]
            if "fitters" in spec and model not in spec["fitters"]:
                continue  # this arm is defined only for some model families
            y_fit = ctx.y if spec["target"] == "raw" else targets[spec["target"]]
            t0 = time.time()
            print(f"\n{'=' * 78}\n{key}  (target={spec['target']}, winsor={spec['winsor']}, seed={SEED})\n{'=' * 78}", flush=True)
            pred, imp, fold_info = run_wf(ctx, FEATURES, y_fit, model, key, winsor=spec["winsor"],
                                          fitter=spec.get("fitters", {}).get(model))
            stats, ic, preds = score_arm(ctx, pred, key + args.suffix, variants, OUT)
            u = preds[universe_mask(preds, FLOOR)]
            for k in ALT_SCORES:
                stats[f"ic_vs_{k}"] = float(monthly_rank_ic(u["target_month"].to_numpy(), u["pred"].to_numpy(),
                                                            u[f"alt_{k}"].to_numpy()).mean())
            stats["fold_selection"] = fold_info
            results[key], ics[key] = stats, ic
            g = stats[HEADLINE_VARIANT]
            print(f"  IC {stats['universe_rank_ic']:.4f} (t {stats['universe_rank_ic_t']:.2f}) D10-D1 "
                  f"{stats['d10_minus_d1_pct_per_month']:+.2f}%/m | lc_t10 IR gross {g['ir_gross']:.2f} net {g['ir_net']:.2f} "
                  f"| beta {g['beta']:+.2f} | {time.time() - t0:.0f}s", flush=True)
            (OUT / f"results{args.suffix}.json").write_text(json.dumps(_clean(results), indent=2))
            pd.DataFrame(ics).rename_axis("target_month").to_csv(OUT / f"monthly_ic{args.suffix}.csv")

    # paired tests against the control arm within each model family + the pre-registered adoption rule
    paired_rows, adopt = [], {}
    for model in args.models.split(","):
        ck = f"{model}__control"
        for arm in arms:
            key = f"{model}__{arm}"
            if arm == "control" or ck not in ics or key not in ics:
                continue
            p = paired(ics[key], ics[ck])
            paired_rows.append({"model": model, "arm": arm, "ic": results[key]["universe_rank_ic"],
                                "ic_control": results[ck]["universe_rank_ic"], **p,
                                "ir_gross": results[key][HEADLINE_VARIANT]["ir_gross"],
                                "ir_gross_control": results[ck][HEADLINE_VARIANT]["ir_gross"],
                                "ir_net": results[key][HEADLINE_VARIANT]["ir_net"],
                                "ir_net_control": results[ck][HEADLINE_VARIANT]["ir_net"]})
    pr = pd.DataFrame(paired_rows)
    if len(pr):
        pr.to_csv(OUT / f"paired_vs_control{args.suffix}.csv", index=False)
        for arm in arms:
            if arm == "control":
                continue
            sub = pr[pr["arm"] == arm]
            if not len(sub):
                continue
            adopt[arm] = {"families_with_paired_t_ge_2": int((sub["t"] >= 2).sum()),
                          "adopt_rule_A_two_families_t_ge_2": bool((sub["t"] >= 2).sum() >= 2),
                          "adopt_rule_B_ic_ge_0.037_and_t_ge_2": bool(any(
                              (results[f"{m}__{arm}"]["universe_rank_ic"] >= 0.037) and
                              (results[f"{m}__{arm}"]["universe_rank_ic_t"] >= 2) for m in sub["model"]))}
    (OUT / f"results{args.suffix}.json").write_text(json.dumps(_clean({**results, "_adoption_rule": adopt}), indent=2))

    rows = [{"key": k, "ic": v["universe_rank_ic"], "t": v["universe_rank_ic_t"],
             "d10_d1": v["d10_minus_d1_pct_per_month"], "ir_gross_t10": v[HEADLINE_VARIANT]["ir_gross"],
             "ir_net_t10": v[HEADLINE_VARIANT]["ir_net"], "ir_gross_free": v["lc_free"]["ir_gross"],
             **{k2: v[k2] for k2 in v if k2.startswith("ic_vs_")}} for k, v in results.items()]
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / f"summary{args.suffix}.csv", index=False)
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(pr):
        print("\nPaired monthly IC vs control (same model family):")
        print(pr[["model", "arm", "ic", "ic_control", "mean_diff", "t", "ir_gross", "ir_gross_control"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nAdoption rules:", json.dumps(adopt))


if __name__ == "__main__":
    main()
