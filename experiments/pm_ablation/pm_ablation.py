"""
FIAM 2026 - Portfolio-constraint ablation and 5-model ensemble (PM_ABLATION).

The first run of rf_3.py showed the legacy LP (docs/OLS.md sec 2.5) at IR ~1.3 but every portfolio-manager
variant (Valentino's tips + financial-engineer notes: price/market-cap screens, sector limits, dual-beta
neutrality, turnover budget) at IR ~0. This script finds WHICH constraint removes the edge, and how the edge
depends on the market-cap floor, using predictions the model scripts already saved (no refitting):
  rf_3.py, et.py, lgbm.py, cat.py, xgb_2.py  ->  output/oos_predictions_<tag>_modern_nomom.csv
plus an equal-weight ensemble `ens5` (per-month cross-sectional rank of each model's prediction, averaged).

Ablations (all on the `modern_nomom` predictions; each row = legacy LP + ONE change, then combinations):
  legacy | +price>=5 | +mcap>=250/500/1000/2000 | +price&mcap500 | +dual beta | +sector limits | +turnover 10%
  | screens+dualbeta+sector (pm_free) | pm_t10
  | decile_ew_tradeable: docs/FIAM.md sec 6's baseline -- long top decile / short bottom decile, equal weight,
    tradeable universe only, NOT beta-neutral (checks whether the tradeable signal survives simple weighting).
Every row is reported gross and net of the market-cap-tiered costs used in the model scripts (assumptions).
The portfolio code below is ported from rf_3.py (same LP, same cost tiers, same metrics).

Run:  .venv/bin/python pm_ablation.py
Outputs (output/): pm_ablation_summary.csv, pm_ablation_results.json, pm_ablation_ic_by_bucket.csv,
    pm_ablation_sector_exposure_legacy_rf3.csv, pm_ablation_top_shorts_legacy_rf3.csv, pm_ablation_deciles_tradeable.csv, oos_predictions_ens5_modern_nomom.csv
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

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"  # downloaded external data only (FRED series)
OUTPUT = BASE / "output"
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
# Feature arms. MODERN is docs/RF.md's 15-factor arm (includes 3 momentum
# factors). MODERN_NOMOM drops them (a financial-engineer review flagged
# momentum as suspect; docs/RF_2.md's leave-group-out already showed dropping
# it RAISES IR). ALL_NOMOM is all 147 characteristics minus the 8 characteristics
# in docs/FACTORS.md sec 4 ("Momentum"). NOTE: `mispricing_perf` is itself a
# composite that includes a momentum component (docs/FACTORS.md sec 13) and is
# kept in every arm -- it is the most load-bearing single factor in
# docs/RF_2.md's leave-one-out.
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
# 2. Selection metrics. docs/RF_2.md found validation MSE identical to four
#    decimals across 15 RF configs (it cannot tell configs apart), and
#    docs/XGB.md's 2025 fold early-stopped after 4 trees on MSE. Everything here
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
# LEGACY: the LP used by every prior script (docs/OLS.md sec 2.5): $10M
# dollar-volume screen, dollar-neutral, beta_60m-neutral, gross 200%, 1% cap.
# Kept so results here are directly comparable to docs/RF.md, docs/RF_2.md etc.
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
    """Returns (results dict, per-month frame). Same metrics as every prior script (docs/OLS.md sec 2.6),
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
# 5. Ablation driver
# ---------------------------------------------------------------------------

MODEL_TAG = "pm_ablation"
MODEL_TAGS = ["rf3", "et", "lgbm", "cat", "xgb2"]
ARM = "modern_nomom"

ABLATIONS = {
    "legacy": {},
    "+price>=5": {"min_price": 5.0},
    "+mcap>=250M": {"min_mcap": 250.0},
    "+mcap>=500M": {"min_mcap": 500.0},
    "+mcap>=1000M": {"min_mcap": 1000.0},
    "+mcap>=2000M": {"min_mcap": 2000.0},
    "+price5&mcap500": {"min_price": 5.0, "min_mcap": 500.0},
    "+dual_beta": {"beta_cols": ("raw_beta_60m", "raw_betabab_1260d")},
    "+sector_limits": {"sector_net": 0.05, "sector_gross": 0.70},
    "+turnover_10pct": {"turnover": 0.10},
    "pm_free": {k: v for k, v in PM_CFG.items() if k != "turnover"},
    "pm_t10": {},
}


def decile_book(preds, frac=0.10):
    """docs/FIAM.md sec 6 baseline: within the TRADEABLE universe (price >= $5, market cap >= $500M, $10M dollar
    volume), long the top `frac` and short the bottom `frac` by prediction, equal weight, 100% long / 100% short.
    Dollar-neutral by construction; NOT beta-neutral, no sector limits, no turnover control."""
    hold, mon = [], []
    for month, grp in preds.groupby("target_month"):
        g = grp[(grp["raw_prc"].abs() >= 5.0) & (grp["raw_me"] >= 500.0) & (grp["raw_dolvol_126d"] >= 1e7)]
        g = g.dropna(subset=["raw_beta_60m", "raw_betabab_1260d"]).sort_values("pred")
        n = max(int(len(g) * frac), 1)
        both = pd.concat([g.iloc[:n], g.iloc[-n:]]).copy()
        both["weight"] = np.r_[np.full(n, -1.0 / n), np.full(n, 1.0 / n)]
        both["target_month"] = month
        hold.append(both[["target_month", "permno", "ticker", "company_name", "sector", "weight", TARGET_COL, "pred",
                          "raw_me", "raw_beta_60m", "raw_betabab_1260d"]])
        lm = both["weight"] > 0
        sec_net = both.groupby("sector")["weight"].sum().abs()
        mon.append({
            "target_month": month, "port_excess_ret": (both["weight"] * both[TARGET_COL]).sum(),
            "long_leg_ret": (both.loc[lm, "weight"] * both.loc[lm, TARGET_COL]).sum(),
            "short_leg_ret": (both.loc[~lm, "weight"] * both.loc[~lm, TARGET_COL]).sum(),
            "gross_exposure": both["weight"].abs().sum(), "net_exposure": both["weight"].sum(),
            "beta_exposure": (both["weight"] * both["raw_beta_60m"]).sum(),
            "betabab_exposure": (both["weight"] * both["raw_betabab_1260d"]).sum(),
            "max_abs_sector_net": float(sec_net.max()),
            "max_sector_gross_share": float(both.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).max() / GROSS),
            "turnover_cap_relaxed": False, "turnover_relax_mult": np.nan,
            "n_long": int(lm.sum()), "n_short": int((~lm).sum()), "n_positions": len(both),
        })
    return pd.concat(hold, ignore_index=True), pd.DataFrame(mon).sort_values("target_month").reset_index(drop=True)


def light_eval(preds, cfg):
    holdings, stats = decile_book(preds) if cfg == "decile" else build_portfolio(preds, cfg)
    g, _ = compute_performance(stats)
    trade, tcost = trade_frame(holdings)
    borrow = borrow_series(holdings)
    ns = stats.copy()
    ns["port_excess_ret"] = stats["port_excess_ret"].to_numpy() - tcost.reindex(ns["target_month"]).to_numpy() \
        - borrow.reindex(ns["target_month"]).fillna(0.0).to_numpy()
    n, _ = compute_performance(ns)
    tn = trade.reindex(ns["target_month"]).to_numpy()
    sb = compute_short_book_characteristics(holdings)
    return {
        "ir_gross": g["information_ratio"], "ir_net": n["information_ratio"], "cagr_gross_pct": 100 * g["annualized_return_geo_cagr"],
        "cagr_net_pct": 100 * n["annualized_return_geo_cagr"], "alpha_t_net": n["alpha_tstat"], "beta": g["beta"],
        "roll_beta_min": g["rolling_beta_12m_min"], "roll_beta_max": g["rolling_beta_12m_max"],
        "long_leg_cagr_pct": 100 * g["long_leg_cagr"], "short_leg_cagr_pct": 100 * g["short_leg_cagr"],
        "turnover_pct_gross": 100 * float(np.mean(tn[1:]) / (2 * GROSS)), "max_dd_net_pct": 100 * n["max_drawdown"],
        "n_pos_min": g["min_n_positions"], "n_pos_max": g["max_n_positions"],
        "short_median_mcap_musd": sb["short_book_median_market_cap_musd"], "short_share_lt_1b": sb["short_book_share_below_1000m_cap"],
        "months_cap_relaxed": int(stats["turnover_cap_relaxed"].sum()),
    }


def ensemble_preds(frames):
    base = frames[MODEL_TAGS[0]][["permno", "target_month", "ticker", "company_name", TARGET_COL, "sector", "raw_prc",
                                  "raw_dolvol_126d", "raw_me", "raw_beta_60m", "raw_betabab_1260d"]].copy()
    r = []
    for t in MODEL_TAGS:
        f = frames[t]
        assert (f["permno"].to_numpy() == base["permno"].to_numpy()).all() and (f["target_month"].to_numpy() == base["target_month"].to_numpy()).all()
        r.append(f.groupby("target_month")["pred"].rank(pct=True).to_numpy())
    base["pred"] = np.mean(r, axis=0) - 0.5
    return base


if __name__ == "__main__":
    frames = {t: pd.read_csv(OUT / f"oos_predictions_{t}_{ARM}.csv", parse_dates=["target_month"]) for t in MODEL_TAGS}
    frames["ens5"] = ensemble_preds(frames)
    frames["ens5"].to_csv(OUT / f"oos_predictions_ens5_{ARM}.csv", index=False)
    rows, ic_rows = [], []
    BUCKETS = {
        "all_stocks": lambda d: np.ones(len(d), bool),
        "liquid_$10M+": lambda d: (d["raw_dolvol_126d"] >= 1e7).to_numpy(),
        "price<5": lambda d: (d["raw_prc"].abs() < 5).to_numpy(),
        "price>=5": lambda d: (d["raw_prc"].abs() >= 5).to_numpy(),
        "mcap<250M": lambda d: (d["raw_me"] < 250).to_numpy(),
        "mcap250-500M": lambda d: ((d["raw_me"] >= 250) & (d["raw_me"] < 500)).to_numpy(),
        "mcap500M-2B": lambda d: ((d["raw_me"] >= 500) & (d["raw_me"] < 2000)).to_numpy(),
        "mcap>=2B": lambda d: (d["raw_me"] >= 2000).to_numpy(),
        "tradeable_and_betas_observed(LP-investable)": lambda d: ((d["raw_prc"].abs() >= 5) & (d["raw_me"] >= 500) & (d["raw_dolvol_126d"] >= 1e7)
                                                                  & d["raw_beta_60m"].notna() & d["raw_betabab_1260d"].notna()).to_numpy(),
        "tradeable(price>=5&mcap>=500M&$10M)": lambda d: ((d["raw_prc"].abs() >= 5) & (d["raw_me"] >= 500) & (d["raw_dolvol_126d"] >= 1e7)).to_numpy(),
    }
    for tag, preds in frames.items():
        for bname, fn in BUCKETS.items():
            sub = preds[fn(preds)]
            bic = monthly_rank_ic(sub["target_month"].to_numpy(), sub["pred"].to_numpy(), sub[TARGET_COL].to_numpy())
            ic_rows.append({"model": tag, "bucket": bname, "mean_monthly_rank_ic": float(bic.mean()), "share_months_positive": float((bic > 0).mean()),
                            "avg_stocks_per_month": float(sub.groupby("target_month").size().mean())})
        pd.DataFrame(ic_rows).to_csv(OUT / "pm_ablation_ic_by_bucket.csv", index=False)
        ic = monthly_rank_ic(preds["target_month"].to_numpy(), preds["pred"].to_numpy(), preds[TARGET_COL].to_numpy())
        for name, delta in [*ABLATIONS.items(), ("decile_ew_tradeable", "decile")]:
            cfg = "decile" if delta == "decile" else ({**PM_CFG, **delta} if name == "pm_t10" else {**LEGACY_CFG, **delta})
            r = light_eval(preds, cfg)
            rows.append({"model": tag, "portfolio": name, "test_rank_ic": float(ic.mean()), **r})
            print(f"[{tag:5s}] {name:16s} gross IR {r['ir_gross']:5.2f} net IR {r['ir_net']:5.2f} | long {r['long_leg_cagr_pct']:5.1f}% short {r['short_leg_cagr_pct']:5.1f}% "
                  f"| beta {r['beta']:+.2f} roll[{r['roll_beta_min']:+.2f},{r['roll_beta_max']:+.2f}] | turn {r['turnover_pct_gross']:.0f}% | short med cap ${r['short_median_mcap_musd']:.0f}M", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "pm_ablation_summary.csv", index=False)
    res = pd.DataFrame(rows)
    (OUT / "pm_ablation_results.json").write_text(json.dumps(_clean(res.to_dict(orient="records")), indent=2))
    # Mean next-month excess return by within-month prediction decile, tradeable universe only
    # (price >= $5, market cap >= $500M, $10M dollar volume) AND with both betas observed -- the same universe the LP and the
    # decile book can hold. Equal-weighted, all OOS months pooled.
    dec_rows = []
    for tag, preds in frames.items():
        t = preds[BUCKETS["tradeable(price>=5&mcap>=500M&$10M)"](preds)].dropna(subset=["raw_beta_60m", "raw_betabab_1260d"]).copy()
        t["dec"] = t.groupby("target_month")["pred"].rank(pct=True, method="first").mul(10).apply(np.ceil).clip(1, 10).astype(int)
        m = t.groupby(["target_month", "dec"])[TARGET_COL].mean().groupby("dec").mean()
        univ = t.groupby("target_month")[TARGET_COL].mean().mean()
        for d_, v in m.items():
            dec_rows.append({"model": tag, "decile": int(d_), "mean_monthly_excess_ret_pct": 100 * float(v), "universe_mean_pct": 100 * float(univ)})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / "pm_ablation_deciles_tradeable.csv", index=False)
    print("\nMean next-month excess return (%) by prediction decile, tradeable universe (1 = lowest prediction):")
    print(dec.pivot(index="decile", columns="model", values="mean_monthly_excess_ret_pct").round(3).to_string())
    # Sector footprint of the legacy (unconstrained-by-sector) book, from rf_3.py's holdings file.
    hold = pd.read_csv(OUT / f"portfolio_holdings_legacy_rf3_{ARM}.csv", dtype={"sector": str})
    nm = hold["target_month"].nunique()
    sec = hold.groupby("sector")["weight"].agg(avg_net_weight=lambda x: x.sum() / nm, avg_gross_weight=lambda x: x.abs().sum() / nm)
    sec["avg_gross_share_of_book"] = sec["avg_gross_weight"] / GROSS
    sec["max_abs_net_in_any_month"] = hold.groupby(["target_month", "sector"])["weight"].sum().abs().groupby("sector").max()
    sec = sec.sort_values("avg_net_weight")
    sec.to_csv(OUT / "pm_ablation_sector_exposure_legacy_rf3.csv")
    top_short = hold[hold["weight"] < 0].groupby(["ticker", "company_name"])["weight"].sum().sort_values().head(10) / nm
    top_short.rename("avg_weight").to_csv(OUT / "pm_ablation_top_shorts_legacy_rf3.csv")
    print("\nSector exposure of the legacy rf3 book (GICS 2-digit; weights are % of NAV as fractions; 35 = Health Care):")
    print(sec.round(3).to_string())
    print("\nLargest average shorts:"); print(top_short.round(3).to_string())
    print("\nRank IC by size bucket (mean monthly Spearman, 2021-01..2026-08):")
    print(pd.DataFrame(ic_rows).pivot(index="bucket", columns="model", values="mean_monthly_rank_ic").round(3).to_string())
    print("\nWrote", OUT / "pm_ablation_summary.csv")
