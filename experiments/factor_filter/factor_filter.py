"""
FIAM 2026 - Does filtering the 147 factors help? (FACTOR_FILTER)

Question (from the user): instead of feeding all 147 characteristics to a model, would criteria / filters /
preprocessing for including a factor help -- e.g. dropping highly correlated ones? This script answers it with
three pieces, none of which touches an existing file:

  A. REDUNDANCY (unsupervised, no target): correlation structure of the 147 rank-transformed characteristics on the
     pre-2021 (training-era) investable universe -- near-duplicate pairs, average-linkage clusters at |rho| > 0.8,
     and how many principal components explain 80/90/95% of the variance.
  B. PERSISTENCE: does a factor's past information coefficient (IC) predict its future IC? Per-factor mean monthly
     rank IC with next-month return in the LP-investable universe (price >= $5, market cap >= $500M, $10M dollar
     volume, both betas observed) and in all stocks, for target months before 2021 (train era) vs 2021-01..2026-08.
  C. FILTER EXPERIMENT: Extra-Trees (experiments/et/README.md) on feature sets chosen by filter rules, WALK-FORWARD -- every rule
     sees only rows whose target month is before that fold's validation window (`tr`), never validation/test data:
       all139         all characteristics except the 8 momentum ones (experiments/rf_3/README.md)
       decorr         cluster pruning at |rho| > 0.8, keep the member with the largest |train IC| per cluster
       ic_top30       the 30 factors with the largest |mean train IC|
       decorr_ic_top30  cluster pruning, then the 30 largest |train IC| among the representatives
       sign_stable    |IC t-stat| >= 2 over the training window AND the same IC sign in both halves of it
                      (at least 5 kept; if fewer qualify, the 5 largest |t|)
       curated12      experiments/rf_3/README.md's momentum-free Modern factors (reference)
       random30_s0..4 five random 30-factor subsets (the NULL: how good is 'any 30 factors'?)
     Models are trained and validated on the LP-investable rows only and scored on every test stock; the headline
     metric is test rank IC on the LP-investable test universe, with pm_free / pm_t10 portfolios (experiments/rf_3/README.md
     portfolio layer, gross and net of the assumed cost tiers) as the secondary check.

Run:  .venv/bin/python experiments/factor_filter/factor_filter.py [--smoke]
Outputs (output/): factor_redundancy_clusters.csv, factor_redundancy_pairs.csv, factor_ic_persistence.csv,
    factor_filter_selected_features.csv, factor_filter_summary.csv, factor_filter_results.json,
    factor_filter_monthly_ic.csv, factor_filter_paired_tests.csv
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.optimize import linprog
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent  # experiments/<name>/
BASE = HERE.parents[1]  # project root: fiam/ data and cache/ are shared
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"  # downloaded external data only (FRED series)
OUTPUT = HERE / "output"
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)
OUT = OUTPUT  # rebound by --smoke so plumbing tests never touch real outputs

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"
OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")
TEST_YEARS = range(2021, 2027)
SEED = 42

# ---------------------------------------------------------------------------
# Feature arms. MODERN is experiments/rf/README.md's 15-factor arm (includes 3 momentum
# factors). MODERN_NOMOM drops them (a financial-engineer review flagged
# momentum as suspect; experiments/rf_2/README.md's leave-group-out already showed dropping
# it RAISES IR). ALL_NOMOM is all 147 characteristics minus the 8 characteristics
# in docs/FACTORS.md sec 4 ("Momentum"). NOTE: `mispricing_perf` is itself a
# composite that includes a momentum component (docs/FACTORS.md sec 13) and is
# kept in every arm -- it is the most load-bearing single factor in
# experiments/rf_2/README.md's leave-one-out.
# ---------------------------------------------------------------------------
MODERN_FEATURES = [
    "ret_12_1", "ret_6_1", "resff3_12_1", "qmj", "qmj_prof", "qmj_growth",
    "mispricing_mgmt", "mispricing_perf", "niq_su", "saleq_su", "rmax5_21d",
    "ivol_capm_21d", "betabab_1260d", "at_gr1", "gp_at",
]
MOMENTUM_FEATURES = [
    "ret_3_1", "ret_6_1", "ret_9_1", "ret_12_1", "ret_12_7", "resff3_6_1", "resff3_12_1", "prc_highprc_252d",
]
MODERN_NOMOM_FEATURES = [f for f in MODERN_FEATURES if f not in MOMENTUM_FEATURES]


def feature_arms(stock_vars):
    base = {
        "modern": MODERN_FEATURES,
        "modern_nomom": MODERN_NOMOM_FEATURES,
        "all_nomom": [f for f in stock_vars if f not in MOMENTUM_FEATURES],
    }
    # `<arm>_trad`: same features, but the model is TRAINED and VALIDATED only on tradeable-universe rows
    # (the universe the PM portfolio can actually hold); test rows are still scored for every stock.
    return {**base, **{f"{k}_trad": v for k, v in base.items()}}


# ---------------------------------------------------------------------------
# 1. Data (ported from rf_2.py; adds the GICS sector and market-cap tier inputs
#    the portfolio-aware LP needs)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. Selection metrics. experiments/rf_2/README.md found validation MSE identical to four
#    decimals across 15 RF configs (it cannot tell configs apart), and
#    experiments/xgb/README.md's 2025 fold early-stopped after 4 trees on MSE. Everything here
#    selects hyperparameters on VALIDATION mean monthly rank IC instead -- the
#    quantity a long/short book actually monetizes.
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


# ---------------------------------------------------------------------------
# 3. Portfolio construction.
#
# LEGACY: the LP used by every prior script (experiments/ols/README.md sec 2.5): $10M
# dollar-volume screen, dollar-neutral, beta_60m-neutral, gross 200%, 1% cap.
# Kept so results here are directly comparable to experiments/rf/README.md, experiments/rf_2/README.md etc.
#
# PM ("portfolio-manager"): the same LP plus the constraints from Valentino's
# tips (Valentino_FIAM_Tips.pdf) and the financial engineer's notes:
#   * turnover budget as a HARD LP constraint (one-way, share of the 200% gross
#     book, drift-adjusted trades; same convention as this project's
#     `avg_monthly_turnover`). The tips suggest ~10% per month.
#   * sector limits: |net| per GICS sector <= 5% of NAV, gross per sector <=
#     35% of the gross book.
#   * neutral to BOTH beta_60m and betabab_1260d (beta consistency).
#   * tradeability screens: price >= $5, market cap >= $500M (short-book borrow).
#   * transaction and borrow costs, tiered by market cap (assumptions, sec 3b).
# ---------------------------------------------------------------------------

LEGACY_CFG = dict(min_dolvol=10_000_000.0, min_price=0.0, min_mcap=0.0, beta_cols=("raw_beta_60m",),
                  max_weight=0.01, sector_net=None, sector_gross=None, turnover=None)
PM_CFG = dict(min_dolvol=10_000_000.0, min_price=5.0, min_mcap=500.0, beta_cols=("raw_beta_60m", "raw_betabab_1260d"),
              max_weight=0.01, sector_net=0.05, sector_gross=0.70, turnover=0.10)
VARIANTS = {
    "legacy": LEGACY_CFG,
    "pm_free": {**PM_CFG, "turnover": None},
    "pm_t20": {**PM_CFG, "turnover": 0.20},
    "pm_t10": {**PM_CFG, "turnover": 0.10},
}
HEADLINE_VARIANT = "pm_t10"
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


# ---------------------------------------------------------------------------
# 4. Bookkeeping
# ---------------------------------------------------------------------------


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


def run_arm(ctx, features, arm, args):
    print(f"\n{'=' * 78}\nMODEL {MODEL_TAG} | ARM {arm} ({len(features)} characteristics)\n{'=' * 78}", flush=True)
    t0 = time.time()
    pred_vec, imp, fold_info = run_walk_forward(ctx, features, f"{MODEL_TAG}/{arm}", winsor=not args.raw_target,
                                                train_universe="tradeable" if arm.endswith("_trad") else "all")
    preds = preds_frame(ctx, pred_vec)
    preds.to_csv(OUT / f"oos_predictions_{MODEL_TAG}_{arm}.csv", index=False)
    imp.rename("importance").rename_axis("variable").to_csv(OUT / f"{MODEL_TAG}_feature_importance_{arm}.csv")

    ic = monthly_rank_ic(preds["target_month"].to_numpy(), preds["pred"].to_numpy(), preds[TARGET_COL].to_numpy())
    model_stats = {
        "n_features": len(features), "oos_r2_pct": 100 * oos_r2(preds[TARGET_COL].values, preds["pred"].values),
        "test_rank_ic": float(ic.mean()), "test_rank_ic_share_positive": float((ic > 0).mean()),
        "mean_val_rank_ic": float(np.mean([f["val_ic"] for f in fold_info])),
        "pred_rank_autocorr": pred_rank_autocorr(preds),
    }
    print(f"  model-level: {json.dumps(_clean(model_stats))}", flush=True)

    out = {"features": features, "model_stats": model_stats, "fold_selection": fold_info,
           "top10_importance": imp.head(10).to_dict(), "variants": {}}
    rows = []
    for name, cfg in VARIANTS.items():
        res = evaluate_variant(preds, name, cfg, MODEL_TAG, arm)
        out["variants"][name] = res
        rows.append(summary_row(MODEL_TAG, arm, name, res, model_stats))
        g, n = res["gross"], res["net"]
        print(f"  [{name:8s}] gross IR {g['information_ratio']:.2f} / net IR {n['information_ratio']:.2f} | "
              f"beta {g['beta']:+.2f} (t {g['beta_tstat']:+.2f}) roll12 [{g['rolling_beta_12m_min']:+.2f},{g['rolling_beta_12m_max']:+.2f}] | "
              f"turnover {100 * res['drift_adjusted_one_way_turnover_pct_of_gross']:.0f}% | maxDD {100 * g['max_drawdown']:.0f}% | "
              f"pos {g['min_n_positions']}-{g['max_n_positions']} | relaxed {res['months_turnover_cap_relaxed']}", flush=True)
    print(f"  arm done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out, rows


def main(default_arms):
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(default_arms), help="comma list of: modern, modern_nomom, all_nomom, and any of those + _trad (train on tradeable universe only)")
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing run (subsampled stocks, 1 fold), scratch output dir")
    ap.add_argument("--raw-target", action="store_true", help="fit on the raw next-month return (default: training label winsorized 1/99%%)")
    ap.add_argument("--suffix", default="", help="appended to the results/summary filenames")
    args = ap.parse_args()
    if args.smoke:
        OUT = OUTPUT / f"_smoke_{MODEL_TAG}"
        OUT.mkdir(exist_ok=True)

    print("Loading model table...", flush=True)
    df, stock_vars = load_model_table()
    if args.smoke:
        df = df[df["permno"] % 4 == 0].reset_index(drop=True)
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors", flush=True)
    print("Cross-sectional preprocessing...", flush=True)
    df = cross_sectional_rank_transform(df, stock_vars)
    ctx = Context(df, stock_vars, max_folds=1 if args.smoke else None)
    del df
    arms = feature_arms(stock_vars)

    results, all_rows = {}, []
    for arm in args.arms.split(","):
        results[arm], rows = run_arm(ctx, arms[arm], arm, args)
        all_rows += rows
        (OUT / f"{MODEL_TAG}_results{args.suffix}.json").write_text(json.dumps(_clean(results), indent=2))
        pd.DataFrame(all_rows).to_csv(OUT / f"{MODEL_TAG}_summary{args.suffix}.csv", index=False)

    summ = pd.DataFrame(all_rows)
    cols = ["arm", "portfolio", "ir_gross", "ir_net", "sharpe_net", "cagr_net_pct", "beta", "roll_beta_min", "roll_beta_max",
            "turnover_one_way_pct_gross", "max_dd_net_pct", "test_rank_ic", "mean_val_rank_ic", "pred_rank_autocorr"]
    print(f"\n{'=' * 78}\nSUMMARY {MODEL_TAG}\n{'=' * 78}")
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nFull results written to {OUT / (MODEL_TAG + '_results' + args.suffix + '.json')}")


# ---------------------------------------------------------------------------
# 5. Factor-filter study
# ---------------------------------------------------------------------------

MODEL_TAG = "factor_filter"
ET_GRID = [{"max_features": "sqrt", "min_samples_leaf": leaf} for leaf in (300, 1000, 3000)]
N_ET = 200
K_TOP = 30
CLUSTER_RHO = 0.8
N_RANDOM = 5


def investable_mask(df):
    return ((df["raw_prc"].abs() >= PM_CFG["min_price"]) & (df["raw_me"] >= PM_CFG["min_mcap"])
            & (df["raw_dolvol_126d"] >= PM_CFG["min_dolvol"]) & df["raw_beta_60m"].notna()
            & df["raw_betabab_1260d"].notna()).to_numpy()


def ic_matrix(ctx, row_mask, cols):
    """Monthly Spearman IC (months x factors) between each factor and the next-month return, computed within the
    rows in `row_mask` (ranks re-computed inside that universe each month)."""
    rows = np.flatnonzero(row_mask)
    rows = rows[np.argsort(ctx.tm[rows], kind="stable")]
    m = ctx.tm[rows]
    cuts = np.flatnonzero(m[1:] != m[:-1]) + 1
    out, months = [], []
    for g in np.split(rows, cuts):
        if len(g) < 100:
            continue
        rx = rankdata(ctx.Xall[g][:, cols], axis=0)
        ry = rankdata(ctx.y[g])
        rx = rx - rx.mean(axis=0)
        ry = ry - ry.mean()
        den = np.sqrt((rx**2).sum(axis=0) * (ry**2).sum())
        out.append(np.where(den > 0, (rx.T @ ry) / np.where(den > 0, den, 1), np.nan))
        months.append(ctx.tm[g[0]])
    return pd.DataFrame(out, index=pd.DatetimeIndex(months), columns=[ctx.stock_vars[c] for c in cols])


def ic_stats(icm):
    mean, sd, n = icm.mean(), icm.std(), icm.count()
    return mean, mean / sd * np.sqrt(n)


def prune_correlated(corr, score, thresh=CLUSTER_RHO):
    d = 1.0 - np.abs(corr)
    np.fill_diagonal(d, 0.0)
    labels = hierarchy.fcluster(hierarchy.linkage(squareform(d, checks=False), "average"), t=1.0 - thresh, criterion="distance")
    keep = [int(np.flatnonzero(labels == lab)[np.argmax(score[labels == lab])]) for lab in np.unique(labels)]
    return sorted(keep), labels


# ---- A. redundancy ---------------------------------------------------------
def part_a(ctx, inv, ic_inv_full):
    cols = list(range(len(ctx.stock_vars)))
    train_rows = np.flatnonzero(inv & (ctx.tm < np.datetime64("2021-01-01")))
    C = np.corrcoef(ctx.Xall[train_rows].astype(np.float64).T)
    names = np.array(ctx.stock_vars)
    absC = np.abs(C)
    iu = np.triu_indices(len(names), 1)
    pairs = pd.DataFrame({"a": names[iu[0]], "b": names[iu[1]], "rho": C[iu]})
    pairs = pairs[pairs["rho"].abs() >= 0.9].sort_values("rho", key=np.abs, ascending=False)
    pairs.to_csv(OUT / "factor_redundancy_pairs.csv", index=False)
    mean_ic, _ = ic_stats(ic_inv_full[ic_inv_full.index < "2021-01-01"])
    keep, labels = prune_correlated(C, np.abs(mean_ic.to_numpy()))
    cl = pd.DataFrame({"factor": names, "cluster": labels, "mean_train_ic_investable": mean_ic.to_numpy()})
    cl["cluster_size"] = cl.groupby("cluster")["factor"].transform("size")
    cl.sort_values(["cluster_size", "cluster"], ascending=[False, True]).to_csv(OUT / "factor_redundancy_clusters.csv", index=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    res = {
        "n_factors": len(names), "n_pairs_abs_rho_ge_0.9": int((absC[iu] >= 0.9).sum()), "n_pairs_abs_rho_ge_0.8": int((absC[iu] >= 0.8).sum()),
        "n_pairs_abs_rho_ge_0.95": int((absC[iu] >= 0.95).sum()), "n_clusters_at_0.8": int(len(np.unique(labels))),
        "n_clusters_with_2plus": int((cl.groupby("cluster").size() >= 2).sum()),
        "pcs_for_50pct": int(np.searchsorted(cum, 0.5) + 1), "pcs_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
        "pcs_for_90pct": int(np.searchsorted(cum, 0.9) + 1), "pcs_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "median_abs_rho": float(np.median(absC[iu])),
    }
    print("\n=== A. REDUNDANCY (pre-2021, investable universe) ===")
    print(json.dumps(res, indent=2))
    print("Clusters with 2+ members:")
    for lab, g in cl.groupby("cluster"):
        if len(g) >= 2:
            print(f"  [{len(g)}] " + ", ".join(g.sort_values("factor")["factor"]))
    return res


# ---- B. persistence --------------------------------------------------------
def part_b(ctx, ic_inv, ic_all):
    split = pd.Timestamp("2021-01-01")
    rows = []
    for uni, icm in (("investable", ic_inv), ("all_stocks", ic_all)):
        m_tr, t_tr = ic_stats(icm[icm.index < split])
        m_te, t_te = ic_stats(icm[icm.index >= split])
        d = pd.DataFrame({"factor": icm.columns, "universe": uni, "ic_train": m_tr.to_numpy(), "t_train": t_tr.to_numpy(),
                          "ic_test": m_te.to_numpy(), "t_test": t_te.to_numpy()})
        rows.append(d)
    pers = pd.concat(rows, ignore_index=True)
    pers.to_csv(OUT / "factor_ic_persistence.csv", index=False)
    res = {}
    print("\n=== B. IC PERSISTENCE: does the train-era IC of a factor predict its test-era IC? ===")
    for uni in ("investable", "all_stocks"):
        d = pers[(pers["universe"] == uni) & ~pers["factor"].isin(MOMENTUM_FEATURES)]
        sp = float(d["ic_train"].corr(d["ic_test"], method="spearman"))
        top = d.reindex(d["t_train"].abs().sort_values(ascending=False).index).head(20)
        same = float((np.sign(top["ic_train"]) == np.sign(top["ic_test"])).mean())
        r = {
            "n_factors": len(d), "spearman_corr_train_ic_vs_test_ic": sp,
            "n_abs_t_ge_2_train": int((d["t_train"].abs() >= 2).sum()), "n_abs_t_ge_2_test": int((d["t_test"].abs() >= 2).sum()),
            "expected_by_chance_at_5pct": round(0.05 * len(d), 1),
            "top20_by_abs_train_t__share_same_sign_in_test": same,
            "top20_mean_test_ic_signed_by_train_sign": float((np.sign(top["ic_train"]) * top["ic_test"]).mean()),
            "all_mean_abs_ic_train": float(d["ic_train"].abs().mean()), "all_mean_abs_ic_test": float(d["ic_test"].abs().mean()),
            "median_abs_ic_test": float(d["ic_test"].abs().median()),
        }
        res[uni] = r
        print(f"-- {uni}: {json.dumps(r, indent=1)}")
    return res


# ---- C. filter experiment --------------------------------------------------
def _light_eval(preds, cfg):
    holdings, stats = build_portfolio(preds, cfg)
    g, _ = compute_performance(stats)
    trade, tcost = trade_frame(holdings)
    borrow = borrow_series(holdings)
    ns = stats.copy()
    ns["port_excess_ret"] = stats["port_excess_ret"].to_numpy() - tcost.reindex(ns["target_month"]).to_numpy() \
        - borrow.reindex(ns["target_month"]).fillna(0.0).to_numpy()
    n, _ = compute_performance(ns)
    return {"ir_gross": g["information_ratio"], "ir_net": n["information_ratio"], "beta": g["beta"], "beta_t": g["beta_tstat"],
            "short_leg_cagr_pct": 100 * g["short_leg_cagr"], "long_leg_cagr_pct": 100 * g["long_leg_cagr"]}


def make_rules(ctx, feats139):
    cols139 = np.array(ctx.cols(feats139))
    idx_of = {f: i for i, f in enumerate(feats139)}

    def stats_for(ic_inv, tr_months):
        m, t = ic_stats(ic_inv.loc[tr_months, feats139])
        return m.to_numpy(), t.to_numpy()

    def sel_all(ctx_, tr, ic_inv, tr_months):
        return list(range(len(feats139)))

    def _corr(tr):
        return np.corrcoef(ctx.Xall[tr][:, cols139].astype(np.float64).T)

    def sel_decorr(ctx_, tr, ic_inv, tr_months):
        m, _ = stats_for(ic_inv, tr_months)
        return prune_correlated(_corr(tr), np.abs(m))[0]

    def sel_top(ctx_, tr, ic_inv, tr_months):
        m, _ = stats_for(ic_inv, tr_months)
        return sorted(np.argsort(-np.abs(m))[:K_TOP].tolist())

    def sel_decorr_top(ctx_, tr, ic_inv, tr_months):
        m, _ = stats_for(ic_inv, tr_months)
        keep = prune_correlated(_corr(tr), np.abs(m))[0]
        keep = sorted(keep, key=lambda i: -abs(m[i]))[:K_TOP]
        return sorted(keep)

    def sel_stable(ctx_, tr, ic_inv, tr_months):
        icm = ic_inv.loc[tr_months, feats139]
        half = len(icm) // 2
        m, t = ic_stats(icm)
        m1, m2 = icm.iloc[:half].mean(), icm.iloc[half:].mean()
        ok = (t.abs() >= 2) & (np.sign(m1) == np.sign(m2)) & (np.sign(m1) == np.sign(m))
        keep = np.flatnonzero(ok.to_numpy()).tolist()
        if len(keep) < 5:
            keep = np.argsort(-t.abs().to_numpy())[:5].tolist()
        return sorted(keep)

    cur = [idx_of[f] for f in MODERN_NOMOM_FEATURES]

    rules = {"all139": sel_all, "decorr": sel_decorr, "ic_top30": sel_top, "decorr_ic_top30": sel_decorr_top,
             "sign_stable": sel_stable, "curated12": lambda *a: cur}
    for s in range(N_RANDOM):
        rules[f"random30_s{s}"] = (lambda seed: (lambda ctx_, tr, ic_inv, tr_months, fold=None:
                                                 sorted(np.random.default_rng(1000 * seed + len(tr_months)).choice(len(feats139), K_TOP, replace=False).tolist())))(s)
    return rules, cols139


def run_rule(ctx, inv, ic_inv, rule_name, rule_fn, feats139, cols139, winsor=True):
    pred = np.zeros(len(ctx.meta))
    picked = []
    for fold in ctx.folds:
        t0 = time.time()
        tr = fold["tr"][inv[fold["tr"]]]
        va = fold["va"][inv[fold["va"]]]
        tr_months = ic_inv.index[ic_inv.index < np.datetime64(f"{fold['year'] - 2}-01-01")]
        sel = rule_fn(ctx, tr, ic_inv, tr_months)
        cols = cols139[sel]
        picked.append({"rule": rule_name, "test_year": fold["year"], "n_features": len(sel), "features": [feats139[i] for i in sel]})
        ytr = ctx.y[tr]
        ytr_fit = np.clip(ytr, *np.quantile(ytr, [0.01, 0.99])) if winsor else ytr
        Xtr, Xva, Xte = ctx.Xall[tr][:, cols], ctx.Xall[va][:, cols], ctx.Xall[fold["te"]][:, cols]
        best, best_ic = None, -np.inf
        for p in ET_GRID:
            m = ExtraTreesRegressor(n_estimators=N_ET, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                    bootstrap=False, n_jobs=-1, random_state=SEED).fit(Xtr, ytr_fit)
            vic = float(monthly_rank_ic(ctx.tm[va], m.predict(Xva), ctx.y[va]).mean())
            if vic > best_ic:
                best, best_ic = m, vic
        pred[fold["sl"]] = best.predict(Xte)
        print(f"  [{rule_name} fold {fold['year']}] {len(sel)} features, val_ic {best_ic:.4f}, {time.time() - t0:.0f}s", flush=True)
    return pred, picked


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        OUT = OUTPUT / "_smoke_factor_filter"
        OUT.mkdir(exist_ok=True)
        N_RANDOM = 1
    print("Loading model table...", flush=True)
    df, stock_vars = load_model_table()
    if args.smoke:
        df = df[df["permno"] % 4 == 0].reset_index(drop=True)
    df = cross_sectional_rank_transform(df, stock_vars)
    inv = investable_mask(df)
    ctx = Context(df, stock_vars, max_folds=1 if args.smoke else None)
    del df
    print(f"  investable rows: {inv.sum():,} of {len(inv):,}", flush=True)

    all_cols = list(range(len(stock_vars)))
    ic_inv = ic_matrix(ctx, inv, all_cols)
    ic_all = ic_matrix(ctx, np.ones(len(inv), bool), all_cols)
    results = {"A_redundancy": part_a(ctx, inv, ic_inv), "B_persistence": part_b(ctx, ic_inv, ic_all)}

    feats139 = [f for f in stock_vars if f not in MOMENTUM_FEATURES]
    rules, cols139 = make_rules(ctx, feats139)
    rows, selected, ic_series = [], [], {}
    print("\n=== C. FILTER EXPERIMENT (Extra-Trees, trained on LP-investable rows, walk-forward) ===", flush=True)
    for name, fn in rules.items():
        pred_vec, picked = run_rule(ctx, inv, ic_inv, name, fn, feats139, cols139)
        selected += picked
        preds = preds_frame(ctx, pred_vec)
        pinv = investable_mask(preds)
        ic_i = monthly_rank_ic(preds.loc[pinv, "target_month"].to_numpy(), preds.loc[pinv, "pred"].to_numpy(), preds.loc[pinv, TARGET_COL].to_numpy())
        ic_series[name] = ic_i
        ic_a = monthly_rank_ic(preds["target_month"].to_numpy(), preds["pred"].to_numpy(), preds[TARGET_COL].to_numpy())
        row = {"rule": name, "avg_n_features": float(np.mean([p["n_features"] for p in picked])),
               "min_n_features": int(min(p["n_features"] for p in picked)), "max_n_features": int(max(p["n_features"] for p in picked)),
               "test_ic_investable": float(ic_i.mean()), "test_ic_investable_t": float(ic_i.mean() / ic_i.std() * np.sqrt(len(ic_i))),
               "test_ic_all_stocks": float(ic_a.mean()), "oos_r2_pct": 100 * oos_r2(preds[TARGET_COL].values, preds["pred"].values)}
        for pname in ("pm_free", "pm_t10"):
            r = _light_eval(preds, VARIANTS[pname])
            row.update({f"{pname}_{k}": v for k, v in r.items()})
        rows.append(row)
        print(f"  => {name:16s} feats {row['avg_n_features']:5.1f} | investable IC {row['test_ic_investable']:.4f} (t {row['test_ic_investable_t']:.1f}) | all IC {row['test_ic_all_stocks']:.4f} "
              f"| pm_free IR {row['pm_free_ir_gross']:+.2f}/{row['pm_free_ir_net']:+.2f} | pm_t10 IR {row['pm_t10_ir_gross']:+.2f}/{row['pm_t10_ir_net']:+.2f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "factor_filter_summary.csv", index=False)
        pd.DataFrame(selected).to_csv(OUT / "factor_filter_selected_features.csv", index=False)
    results["C_filter_experiment"] = rows
    # Paired tests on the monthly investable-universe IC series (same test months for every arm, so differences are paired).
    S = pd.DataFrame(ic_series)
    S.to_csv(OUT / "factor_filter_monthly_ic.csv")
    rand_cols = [c for c in S.columns if c.startswith("random30")]
    S["random30_mean"] = S[rand_cols].mean(axis=1)
    paired = []
    for name in [c for c in S.columns if not c.startswith("random30_s")]:
        for base in ("all139", "random30_mean"):
            if name == base:
                continue
            d_ = (S[name] - S[base]).dropna()
            paired.append({"arm": name, "baseline": base, "mean_ic_diff": float(d_.mean()), "paired_t": float(d_.mean() / d_.std() * np.sqrt(len(d_))),
                           "share_months_arm_better": float((d_ > 0).mean()), "n_months": int(len(d_))})
    pd.DataFrame(paired).to_csv(OUT / "factor_filter_paired_tests.csv", index=False)
    print("\nPaired tests on monthly investable IC (arm minus baseline):")
    print(pd.DataFrame(paired).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results["C_paired_tests"] = paired
    rnd = pd.DataFrame(rows).set_index("rule").loc[rand_cols]
    results["C_random30_null"] = {"mean_ic_investable": float(rnd["test_ic_investable"].mean()), "sd_ic_investable": float(rnd["test_ic_investable"].std()),
                                  "mean_pm_t10_ir_gross": float(rnd["pm_t10_ir_gross"].mean()), "sd_pm_t10_ir_gross": float(rnd["pm_t10_ir_gross"].std())}
    (OUT / "factor_filter_results.json").write_text(json.dumps(_clean(results), indent=2))
    print("\nWrote", OUT / "factor_filter_summary.csv")
