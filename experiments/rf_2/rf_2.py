"""
FIAM 2026 - Random Forest, round 2: hardening and stress-testing the MODERN arm.

rf.py (docs/RF.md) ran three feature-set arms (Graham / Modern / Combined) and
found the Modern arm (15 characteristics) was the strongest result in the
project (IR 1.05, Sharpe 1.21, CAGR 30%) while Graham lost money. This script
keeps rf.py untouched and follows up on the "next steps" list for that result:

  Stage  Question it answers
  -----  -------------------------------------------------------------------
  grid   Was the Modern arm's tiny grid (depth {6,10}, leaf 100) hiding
         anything? Widened to depth {4,6,8,10,12} x min_samples_leaf
         {50,100,200}. Also reports EVERY grid point at fixed hyperparameters
         (a robustness surface, not just the validation-picked one), and a
         replication check that the original 2-config grid still gives rf.py's
         Modern numbers. Feature-importance concentration is computed here too.
  loo    Is the win broad-based or driven by 1-2 factors? Leave-one-factor-out
         (15 refits) and leave-one-GROUP-out (momentum / QMJ / mispricing /
         surprise / lottery-vol / other).
  seeds  How much of any difference is just forest seed noise? 4 extra seeds
         plus a 1000-tree forest at the fixed config.
  swaps  Is IR 1.05 fragile to the exact 15-factor choice? Random 2-for-2
         swaps of Modern factors against the other 132 characteristics, and a
         null distribution of 15 RANDOM characteristics (how special is a
         curated 15 vs. a random 15?).
  beta   Why did realized beta get worse than the full-147 RF? Tests the
         betabab_1260d LP constraint (docs/NEXT.md sec 2, never applied
         before), both-beta constraints, and adding beta-family factors back
         into the feature set.
  costs  Turnover and net-of-cost performance (one-way trading costs and
         short borrow cost scenarios, drifted-weight trade accounting).
  longonly  Is the Graham failure a short-leg problem? Top-100 long-only /
         bottom-100 / decile-spread returns for the Graham and Modern signals,
         plus the LP book's long-leg vs short-leg CAGR.
  deck   Deck-required exhibits for the tuned Modern book (docs/FIAM.md sec 12):
         calendar-year table, top-10 long/short holdings, top-10 P&L
         contributors, rolling 12m active return / IR / beta, underwater chart,
         cumulative chart, return histogram.

Everything reuses rf.py's data prep, walk-forward schedule, screen, LP and
evaluation harness (ported, not imported, per project convention), except:
  * the LP can constrain more than one beta column (beta_cols),
  * compute_performance can skip writing its CSV (write=False),
  * feature matrices are built once and sliced by column, so many feature sets
    can be evaluated cheaply.

Run:  .venv/bin/python rf_2.py                       (all stages, ~1.5-2h)
      .venv/bin/python rf_2.py --stages grid,costs   (subset; later stages reuse
                                                      the cached grid)
      .venv/bin/python rf_2.py --smoke               (tiny/fast plumbing test,
                                                      writes to a scratch dir)
Outputs (output/, all prefixed rf2_ or suffixed rf2_modern; rf.py's files are
never overwritten): rf2_results.json plus one CSV per stage, and
output/rf2_deck/*.png|csv for the deck exhibits.
"""

import argparse
import itertools
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"
OUTPUT = BASE / "output"
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)
OUT = OUTPUT  # rebound by --smoke so plumbing tests never touch real outputs

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"

GRAHAM_FEATURES = [
    "be_me", "ni_me", "div12m_me", "debt_me", "at_be", "z_score", "f_score",
    "market_equity", "ni_ar1", "eqpo_me", "cash_at", "age",
]
MODERN_FEATURES = [
    "ret_12_1", "ret_6_1", "resff3_12_1", "qmj", "qmj_prof", "qmj_growth",
    "mispricing_mgmt", "mispricing_perf", "niq_su", "saleq_su", "rmax5_21d",
    "ivol_capm_21d", "betabab_1260d", "at_gr1", "gp_at",
]
MODERN_GROUPS = {
    "momentum": ["ret_12_1", "ret_6_1", "resff3_12_1"],
    "qmj": ["qmj", "qmj_prof", "qmj_growth"],
    "mispricing": ["mispricing_mgmt", "mispricing_perf"],
    "surprise": ["niq_su", "saleq_su"],
    "lottery_vol": ["rmax5_21d", "ivol_capm_21d"],
    "other": ["betabab_1260d", "at_gr1", "gp_at"],
}
assert sorted(sum(MODERN_GROUPS.values(), [])) == sorted(MODERN_FEATURES)
BETA_FAMILY = ["beta_60m", "betadown_252d", "beta_dimson_21d", "corr_1260d"]

OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")
TEST_YEARS = range(2021, 2027)

# Widened grid (was depth {6,10}, leaf 100). Same 200 trees / sqrt features as
# rf.py so the (6,100) and (10,100) points replicate rf.py exactly.
DEPTHS = (4, 6, 8, 10, 12)
LEAVES = (50, 100, 200)
ORIG_CONFIGS = [{"max_depth": 6, "min_samples_leaf": 100}, {"max_depth": 10, "min_samples_leaf": 100}]
N_ESTIMATORS = 200
SEED = 42

MIN_DOLLAR_VOLUME = 10_000_000.0
MAX_WEIGHT = 0.01

# ---------------------------------------------------------------------------
# 1. Data (ported from rf.py; adds raw_betabab_1260d for the LP-constraint test)
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
    df["raw_betabab_1260d"] = df["betabab_1260d"]

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

    META_COLS = ["permno", "target_month", "ticker", "company_name", TARGET_COL,
                 "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m", "raw_betabab_1260d"]

    def __init__(self, df: pd.DataFrame, stock_vars: list[str]):
        self.stock_vars = stock_vars
        self.col_idx = {c: i for i, c in enumerate(stock_vars)}
        self.Xall = df[stock_vars].to_numpy(dtype=np.float32)
        self.y = df[TARGET_COL].to_numpy(dtype=np.float64)
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
        self.meta = df.iloc[np.concatenate(te_all)][self.META_COLS].reset_index(drop=True)

    def cols(self, features):
        return [self.col_idx[f] for f in features]

    def preds_frame(self, pred_vec) -> pd.DataFrame:
        out = self.meta.copy()
        out["pred"] = pred_vec
        return out[(out["target_month"] >= OOS_START) & (out["target_month"] <= OOS_END)]


# ---------------------------------------------------------------------------
# 2. Walk-forward RF fitting (same schedule as rf.py)
# ---------------------------------------------------------------------------


def cfg_key(c):
    return f"d{c['max_depth']}_l{c['min_samples_leaf']}"


def run_grid(ctx: Context, features: list[str], configs: list[dict], n_est=N_ESTIMATORS, seed=SEED, label=""):
    """Fit every config on every fold. Returns {cfg_key: {params, val_mse[F], pred[N], imp[F, k]}}."""
    X = ctx.Xall[:, ctx.cols(features)]
    F, N = len(ctx.folds), len(ctx.meta)
    out = {cfg_key(c): {"params": c, "val_mse": np.zeros(F), "pred": np.zeros(N), "imp": np.zeros((F, len(features)))}
           for c in configs}
    for fi, fold in enumerate(ctx.folds):
        t0 = time.time()
        Xtr, ytr = X[fold["tr"]], ctx.y[fold["tr"]]
        Xva, yva = X[fold["va"]], ctx.y[fold["va"]]
        Xte = X[fold["te"]]
        for c in configs:
            m = RandomForestRegressor(
                n_estimators=n_est, max_depth=c["max_depth"], min_samples_leaf=c["min_samples_leaf"],
                max_features="sqrt", n_jobs=-1, random_state=seed,
            ).fit(Xtr, ytr)
            r = out[cfg_key(c)]
            r["val_mse"][fi] = float(np.mean((yva - m.predict(Xva)) ** 2))
            r["pred"][fold["sl"]] = m.predict(Xte)
            r["imp"][fi] = m.feature_importances_
        print(f"  [{label} fold {fold['year']}] {len(configs)} config(s), {time.time() - t0:.0f}s", flush=True)
    return out


def select_tuned(ctx: Context, grid: dict, keys: list[str]):
    """Per fold, pick the config in `keys` with the lowest VALIDATION mse (no test information used)."""
    F = len(ctx.folds)
    pred = np.zeros(len(ctx.meta))
    chosen, imp = [], []
    for fi, fold in enumerate(ctx.folds):
        best = min(keys, key=lambda k: grid[k]["val_mse"][fi])
        chosen.append({"test_year": fold["year"], **grid[best]["params"], "val_mse": float(grid[best]["val_mse"][fi])})
        pred[fold["sl"]] = grid[best]["pred"][fold["sl"]]
        imp.append(grid[best]["imp"][fi])
    return pred, chosen, np.mean(imp, axis=0)


def fit_fixed(ctx, features, cfg, n_est=N_ESTIMATORS, seed=SEED, label=""):
    return run_grid(ctx, features, [cfg], n_est=n_est, seed=seed, label=label)[cfg_key(cfg)]["pred"]


# ---------------------------------------------------------------------------
# 3. Portfolio construction and evaluation (rf.py's, with two small extensions)
# ---------------------------------------------------------------------------


def oos_r2(actual, predicted):
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum(actual**2)


def build_beta_neutral_portfolio(preds, beta_cols=("raw_beta_60m",)):
    """rf.py's LP; `beta_cols` lets the neutrality constraint use one or several beta columns."""
    beta_cols = list(beta_cols)
    holdings, monthly_stats = [], []
    for month, grp in preds.groupby("target_month"):
        grp = grp[grp["raw_dolvol_126d"] >= MIN_DOLLAR_VOLUME]
        grp = grp.dropna(subset=beta_cols).reset_index(drop=True)
        n = len(grp)
        if n < 2 * int(1.0 / MAX_WEIGHT):
            continue

        pred = grp["pred"].values
        c = np.concatenate([-pred, pred])
        rows = [np.concatenate([np.ones(n), -np.ones(n)])]
        rows += [np.concatenate([grp[b].values, -grp[b].values]) for b in beta_cols]
        rows += [np.concatenate([np.ones(n), np.ones(n)])]
        A_eq = np.array(rows)
        b_eq = [0.0] * (1 + len(beta_cols)) + [2.0]
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, MAX_WEIGHT)] * (2 * n), method="highs")
        if not res.success:
            print(f"  [warn] LP infeasible for {month.date()}, skipping month")
            continue

        w = res.x[:n] - res.x[n:]
        keep = np.abs(w) > 1e-8
        both = grp.loc[keep].copy()
        both["weight"] = w[keep]
        both["target_month"] = month
        holdings.append(both[["target_month", "permno", "ticker", "company_name", "weight", TARGET_COL, "pred",
                              "raw_beta_60m", "raw_betabab_1260d"]])

        long_mask = both["weight"] > 0
        monthly_stats.append({
            "target_month": month,
            "port_excess_ret": (both["weight"] * both[TARGET_COL]).sum(),
            "long_leg_ret": (both.loc[long_mask, "weight"] * both.loc[long_mask, TARGET_COL]).sum(),
            "short_leg_ret": (both.loc[~long_mask, "weight"] * both.loc[~long_mask, TARGET_COL]).sum(),
            "gross_exposure": both["weight"].abs().sum(), "net_exposure": both["weight"].sum(),
            "beta_exposure": (both["weight"] * both["raw_beta_60m"]).sum(),
            "betabab_exposure": (both["weight"] * both["raw_betabab_1260d"]).sum(),
            "n_long": int(long_mask.sum()), "n_short": int((~long_mask).sum()), "n_positions": int(keep.sum()),
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


def compute_performance(stats_df, tag, write=True):
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
        "calendar_year_returns": {int(k): float(v) for k, v in calendar_year_of("port_excess_ret").items()},
        "calendar_year_returns_benchmark": {int(k): float(v) for k, v in calendar_year_of("benchmark_monthly").items()},
        "calendar_year_returns_sp500": {int(k): float(v) for k, v in calendar_year_of("sp500_ret").items()},
        "long_leg_avg_monthly_ret": float(perf["long_leg_ret"].mean()), "long_leg_cagr": float(long_leg_cagr),
        "short_leg_avg_monthly_ret": float(perf["short_leg_ret"].mean()), "short_leg_cagr": float(short_leg_cagr),
        **dd_stats, "sp500_drawdown": sp500_dd_stats,
    }
    if write:
        perf.to_csv(OUT / f"portfolio_returns_{tag}.csv", index=False)
    return results


def monthly_spearman(preds, other):
    ranked = pd.DataFrame({
        "m": preds["target_month"], "a": preds.groupby("target_month")["pred"].rank(),
        "b": preds.groupby("target_month")[other].rank(),
    })
    return float(ranked.groupby("m").apply(lambda g: g["a"].corr(g["b"]), include_groups=False).mean())


def evaluate(preds, beta_cols=("raw_beta_60m",), tag=None, extras=False):
    """One experiment -> one flat row of headline metrics. With extras=True also returns
    (holdings, monthly stats, full performance dict) for the caller to write/inspect."""
    holdings, stats = build_beta_neutral_portfolio(preds, beta_cols)
    perf = compute_performance(stats, tag=tag, write=tag is not None)
    turn = compute_turnover_and_concentration(holdings)
    row = {
        "oos_r2_pct": 100 * oos_r2(preds[TARGET_COL].values, preds["pred"].values),
        "rank_ic": monthly_spearman(preds, TARGET_COL),
        "corr_pred_beta60": monthly_spearman(preds, "raw_beta_60m"),
        "corr_pred_betabab": monthly_spearman(preds, "raw_betabab_1260d"),
        "ir": perf["information_ratio"], "sharpe": perf["sharpe_ratio"],
        "cagr_pct": 100 * perf["annualized_return_geo_cagr"], "alpha_t": perf["alpha_tstat"],
        "beta": perf["beta"], "beta_t": perf["beta_tstat"], "hit_rate": perf["hit_rate_active"],
        "max_dd_pct": 100 * perf["max_drawdown"], "calmar": perf["calmar_ratio"],
        "long_leg_cagr_pct": 100 * perf["long_leg_cagr"], "short_leg_cagr_pct": 100 * perf["short_leg_cagr"],
        "avg_turnover": turn["avg_monthly_turnover"],
        "avg_abs_beta60_exposure": float(stats["beta_exposure"].abs().mean()),
        "avg_abs_betabab_exposure": float(stats["betabab_exposure"].abs().mean()),
        "n_months": perf["n_months"],
    }
    return (row, holdings, stats, perf) if extras else row


# ---------------------------------------------------------------------------
# 4. Result bookkeeping
# ---------------------------------------------------------------------------


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def save_stage(key, obj):
    path = OUT / "rf2_results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = _clean(obj)
    path.write_text(json.dumps(data, indent=2))


def show(df: pd.DataFrame, cols=None, floatfmt="{:.3f}"):
    d = df if cols is None else df[cols]
    print(d.to_string(index=False, float_format=lambda v: floatfmt.format(v)), flush=True)


HEADLINE = ["ir", "sharpe", "cagr_pct", "alpha_t", "beta", "beta_t", "oos_r2_pct", "max_dd_pct", "avg_turnover"]

# ---------------------------------------------------------------------------
# 5. Stages
# ---------------------------------------------------------------------------


def stage_grid(ctx: Context, args):
    """Widened grid for MODERN. Cached because every later stage builds on it."""
    configs = [{"max_depth": d, "min_samples_leaf": l} for d in DEPTHS for l in LEAVES]
    if args.smoke:
        configs = ORIG_CONFIGS
    n_est = 20 if args.smoke else N_ESTIMATORS
    cache_file = CACHE / ("rf2_grid_smoke.pkl" if args.smoke else "rf2_grid.pkl")
    sig = {"features": MODERN_FEATURES, "configs": [cfg_key(c) for c in configs], "n_est": n_est, "seed": SEED}
    if cache_file.exists():
        cached = pickle.loads(cache_file.read_bytes())
        if cached["sig"] == sig:
            print(f"[grid] using cached fits ({cache_file.name})")
            return cached["grid"]
    print(f"[grid] fitting {len(configs)} configs x {len(ctx.folds)} folds, {n_est} trees")
    grid = run_grid(ctx, MODERN_FEATURES, configs, n_est=n_est, label="modern-grid")
    cache_file.write_bytes(pickle.dumps({"sig": sig, "grid": grid}))
    return grid


def fixed_config_from_grid(grid):
    """The single config with the best MEAN validation MSE across folds -- chosen without test data."""
    key = min(grid, key=lambda k: grid[k]["val_mse"].mean())
    return key, grid[key]["params"]


def stage_grid_report(ctx, grid, args):
    rows = []
    for k, g in grid.items():
        row = {"config": k, **g["params"], "mean_val_mse": g["val_mse"].mean()}
        row.update(evaluate(ctx.preds_frame(g["pred"])))
        rows.append(row)
        print(f"  [grid eval] {k}: IR={row['ir']:.3f} Sharpe={row['sharpe']:.3f} beta_t={row['beta_t']:.2f}", flush=True)
    fixed = pd.DataFrame(rows)
    fixed.to_csv(OUT / "rf2_grid_fixed_configs.csv", index=False)

    key_fixed, cfg_fixed = fixed_config_from_grid(grid)

    # Original 2-config grid, tuned -> must reproduce rf.py's Modern numbers.
    orig_keys = [cfg_key(c) for c in ORIG_CONFIGS]
    pred_o, chosen_o, _ = select_tuned(ctx, grid, orig_keys)
    row_o = evaluate(ctx.preds_frame(pred_o))
    # Widened grid, tuned per fold on validation MSE -> the rf_2 Modern book.
    pred_w, chosen_w, imp = select_tuned(ctx, grid, list(grid))
    preds_w = ctx.preds_frame(pred_w)
    row_w, holdings_w, stats_w, perf_w = evaluate(preds_w, tag="beta_neutral_rf2_modern", extras=True)
    preds_w.to_csv(OUT / "oos_predictions_rf2_modern.csv", index=False)
    holdings_w.to_csv(OUT / "portfolio_holdings_beta_neutral_rf2_modern.csv", index=False)
    perf_w.update(compute_turnover_and_concentration(holdings_w))
    perf_w.update(compute_short_book_characteristics(holdings_w))
    pd.Series(imp, index=MODERN_FEATURES, name="importance").rename_axis("variable").sort_values(
        ascending=False).to_csv(OUT / "rf2_feature_importance_modern.csv")

    summary = pd.DataFrame([
        {"run": "rf.py replication (orig 2-config grid, tuned)", **row_o},
        {"run": f"fixed best-mean-val config {key_fixed}", **fixed.set_index("config").loc[key_fixed, list(row_o)].to_dict()},
        {"run": "wide grid (15 configs), tuned per fold", **row_w},
    ])
    summary.to_csv(OUT / "rf2_grid_summary.csv", index=False)

    ref = OUTPUT / "rf_results.json"
    replication = None
    if ref.exists():
        old = json.loads(ref.read_text())["modern"]["beta_neutral"]
        replication = {"rf_py_ir": old["information_ratio"], "rf2_orig_grid_ir": row_o["ir"],
                       "rf_py_beta_t": old["beta_tstat"], "rf2_orig_grid_beta_t": row_o["beta_t"]}
    print("\n=== GRID SUMMARY ===")
    show(summary, ["run"] + HEADLINE)
    print("Replication check vs rf.py:", replication)
    print("Wide-grid per-fold choices:", [(c["test_year"], c["max_depth"], c["min_samples_leaf"]) for c in chosen_w])
    save_stage("grid", {
        "fixed_config": key_fixed, "replication_check": replication,
        "orig_grid_tuned": {**row_o, "folds": chosen_o}, "wide_grid_tuned": {**row_w, "folds": chosen_w},
        "wide_grid_tuned_full": perf_w, "n_configs": len(grid),
        "fixed_config_table": fixed.to_dict(orient="records"),
    })
    return key_fixed, cfg_fixed, row_w, imp, chosen_w


def stage_importance(imp):
    s = pd.Series(imp, index=MODERN_FEATURES).sort_values(ascending=False)
    grp = {g: float(s[f].sum()) for g, f in MODERN_GROUPS.items()}
    hhi = float((s**2).sum() / s.sum() ** 2)
    out = {
        "top1_share": float(s.iloc[0]), "top2_share": float(s.iloc[:2].sum()), "top3_share": float(s.iloc[:3].sum()),
        "top5_share": float(s.iloc[:5].sum()), "hhi": hhi, "effective_n_features": 1 / hhi,
        "equal_weight_hhi_for_reference": 1 / len(s), "by_group": grp, "ranked": s.to_dict(),
    }
    print("\n=== IMPORTANCE CONCENTRATION (impurity, tuned wide-grid models, fold-averaged) ===")
    print(f"  top1={out['top1_share']:.1%} top2={out['top2_share']:.1%} top3={out['top3_share']:.1%} "
          f"effective N={out['effective_n_features']:.1f} of {len(s)}")
    print("  by group:", {k: f"{v:.1%}" for k, v in grp.items()})
    save_stage("importance", out)


def stage_loo(ctx, grid, key_fixed, cfg_fixed, args):
    base = evaluate(ctx.preds_frame(grid[key_fixed]["pred"]))
    rows = [{"experiment": "baseline (all 15)", "dropped": "", **base}]
    drops = [(f, [f]) for f in MODERN_FEATURES] + [(f"GROUP:{g}", fs) for g, fs in MODERN_GROUPS.items()]
    if args.smoke:
        drops = drops[:1] + drops[-1:]
    for name, dropped in drops:
        feats = [f for f in MODERN_FEATURES if f not in dropped]
        pred = fit_fixed(ctx, feats, cfg_fixed, n_est=grid_trees(args), label=f"loo -{name}")
        row = {"experiment": f"drop {name}", "dropped": ",".join(dropped), **evaluate(ctx.preds_frame(pred))}
        row["d_ir"] = row["ir"] - base["ir"]
        rows.append(row)
        print(f"  [loo] drop {name}: IR={row['ir']:.3f} (d={row['d_ir']:+.3f}) beta_t={row['beta_t']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rf2_leave_one_out.csv", index=False)
    print("\n=== LEAVE-ONE-OUT / GROUP-OUT (fixed config", key_fixed, ") ===")
    show(df, ["experiment"] + HEADLINE)
    save_stage("leave_one_out", df.to_dict(orient="records"))
    return base


def grid_trees(args):
    return 20 if args.smoke else N_ESTIMATORS


def stage_seeds(ctx, grid, key_fixed, cfg_fixed, args):
    rows = [{"experiment": f"seed {SEED}, {grid_trees(args)} trees (grid run)",
             **evaluate(ctx.preds_frame(grid[key_fixed]["pred"]))}]
    extra_seeds = [1] if args.smoke else [1, 2, 3, 4]
    for s in extra_seeds:
        pred = fit_fixed(ctx, MODERN_FEATURES, cfg_fixed, n_est=grid_trees(args), seed=s, label=f"seed {s}")
        rows.append({"experiment": f"seed {s}, {grid_trees(args)} trees", **evaluate(ctx.preds_frame(pred))})
    big = 40 if args.smoke else 1000
    pred = fit_fixed(ctx, MODERN_FEATURES, cfg_fixed, n_est=big, seed=SEED, label=f"{big} trees")
    rows.append({"experiment": f"seed {SEED}, {big} trees", **evaluate(ctx.preds_frame(pred))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rf2_seed_sensitivity.csv", index=False)
    seed_rows = df.iloc[: 1 + len(extra_seeds)]
    summ = {"ir_mean": float(seed_rows["ir"].mean()), "ir_std": float(seed_rows["ir"].std()),
            "ir_min": float(seed_rows["ir"].min()), "ir_max": float(seed_rows["ir"].max()),
            "beta_t_min": float(seed_rows["beta_t"].min()), "beta_t_max": float(seed_rows["beta_t"].max())}
    print("\n=== SEED / TREE-COUNT SENSITIVITY ===")
    show(df, ["experiment"] + HEADLINE)
    print("  across seeds:", summ)
    save_stage("seeds", {"summary": summ, "rows": df.to_dict(orient="records")})


def stage_swaps(ctx, grid, key_fixed, cfg_fixed, args):
    rng = np.random.default_rng(2026)
    n_draws = 2 if args.smoke else args.n_draws
    pool = [v for v in ctx.stock_vars if v not in MODERN_FEATURES]
    rows = []
    for i in range(n_draws):
        out_f = list(rng.choice(MODERN_FEATURES, size=2, replace=False))
        in_f = list(rng.choice(pool, size=2, replace=False))
        feats = [f for f in MODERN_FEATURES if f not in out_f] + in_f
        pred = fit_fixed(ctx, feats, cfg_fixed, n_est=grid_trees(args), label=f"swap {i + 1}/{n_draws}")
        r = {"experiment": "swap2", "draw": i, "removed": ",".join(out_f), "added": ",".join(in_f),
             **evaluate(ctx.preds_frame(pred))}
        rows.append(r)
        print(f"  [swap {i + 1}/{n_draws}] -{out_f} +{in_f}: IR={r['ir']:.3f}", flush=True)
    swaps = pd.DataFrame(rows)
    swaps.to_csv(OUT / "rf2_random_swaps.csv", index=False)

    null_rows = []
    for i in range(n_draws):
        feats = list(rng.choice(ctx.stock_vars, size=len(MODERN_FEATURES), replace=False))
        pred = fit_fixed(ctx, feats, cfg_fixed, n_est=grid_trees(args), label=f"null15 {i + 1}/{n_draws}")
        r = {"experiment": "random15", "draw": i, "features": ",".join(feats), **evaluate(ctx.preds_frame(pred))}
        null_rows.append(r)
        print(f"  [random15 {i + 1}/{n_draws}] IR={r['ir']:.3f}", flush=True)
    null = pd.DataFrame(null_rows)
    null.to_csv(OUT / "rf2_random15_null.csv", index=False)

    base_ir = evaluate(ctx.preds_frame(grid[key_fixed]["pred"]))["ir"]
    summ = {
        "modern_ir_fixed_config": base_ir,
        "swap2_ir": {"mean": float(swaps["ir"].mean()), "min": float(swaps["ir"].min()), "max": float(swaps["ir"].max()),
                     "share_below_hurdle_ir_0": float((swaps["ir"] <= 0).mean()), "n": n_draws},
        "random15_ir": {"mean": float(null["ir"].mean()), "std": float(null["ir"].std()), "min": float(null["ir"].min()),
                        "max": float(null["ir"].max()), "share_at_or_above_modern": float((null["ir"] >= base_ir).mean()),
                        "n": n_draws},
    }
    print("\n=== FEATURE-SET SENSITIVITY ===")
    print(json.dumps(_clean(summ), indent=2))
    save_stage("swaps", {"summary": summ, "swap2": swaps.to_dict(orient="records"),
                         "random15": null.to_dict(orient="records")})


def stage_beta(ctx, grid, key_fixed, cfg_fixed, args):
    base_pred = grid[key_fixed]["pred"]
    b60, bab = ("raw_beta_60m",), ("raw_betabab_1260d",)
    both = ("raw_beta_60m", "raw_betabab_1260d")
    variants = [
        ("baseline: features=Modern, LP on beta_60m", MODERN_FEATURES, b60, base_pred),
        ("LP on betabab_1260d", MODERN_FEATURES, bab, base_pred),
        ("LP on beta_60m AND betabab_1260d", MODERN_FEATURES, both, base_pred),
        ("+beta_60m feature, LP on beta_60m", MODERN_FEATURES + ["beta_60m"], b60, None),
        ("+beta family (4) features, LP on beta_60m", MODERN_FEATURES + BETA_FAMILY, b60, None),
        ("+beta family (4) features, LP on both betas", MODERN_FEATURES + BETA_FAMILY, both, "reuse-prev"),
    ]
    rows, prev_pred = [], None
    for name, feats, cols, pred in variants:
        if pred is None:
            pred = fit_fixed(ctx, feats, cfg_fixed, n_est=grid_trees(args), label=name[:30])
        elif isinstance(pred, str):
            pred = prev_pred
        prev_pred = pred
        r = {"experiment": name, "lp_constraint": "+".join(c.replace("raw_", "") for c in cols),
             **evaluate(ctx.preds_frame(pred), beta_cols=cols)}
        rows.append(r)
        print(f"  [beta] {name}: IR={r['ir']:.3f} beta={r['beta']:+.3f} (t={r['beta_t']:+.2f}) "
              f"|b60 exp|={r['avg_abs_beta60_exposure']:.3f} |bab exp|={r['avg_abs_betabab_exposure']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rf2_beta_variants.csv", index=False)
    print("\n=== BETA-CONTROL VARIANTS ===")
    show(df, ["experiment"] + HEADLINE + ["avg_abs_beta60_exposure", "avg_abs_betabab_exposure",
                                          "corr_pred_beta60", "corr_pred_betabab"])
    save_stage("beta", df.to_dict(orient="records"))


def trade_series(holdings_df):
    """Per-month traded notional sum|w_new - w_drifted| (as a multiple of capital; gross book = 2.0).
    Prior weights are drifted by each name's realized return before rebalancing. First month = full build."""
    W = holdings_df.pivot_table(index="target_month", columns="permno", values="weight", fill_value=0.0).sort_index()
    R = holdings_df.pivot_table(index="target_month", columns="permno", values=TARGET_COL, fill_value=0.0).sort_index()
    drifted = (W * (1.0 + R)).shift(1).fillna(0.0)
    return (W - drifted).abs().sum(axis=1)


def stage_costs(holdings, stats):
    trade = trade_series(holdings).reindex(stats["target_month"]).values
    borrow_short = (stats["n_short"] > 0).astype(float).values  # short notional is 1.0 of capital every month
    cost_bps = [0, 5, 10, 20, 30, 50]
    borrow_bps = [0, 50, 200]
    rows = []
    for c, b in itertools.product(cost_bps, borrow_bps):
        net = stats.copy()
        net["port_excess_ret"] = stats["port_excess_ret"].values - trade * c / 1e4 - borrow_short * b / 1e4 / 12
        p = compute_performance(net, tag=None, write=False)
        rows.append({"one_way_cost_bps": c, "borrow_bps_annual": b, "ir": p["information_ratio"],
                     "sharpe": p["sharpe_ratio"], "cagr_pct": 100 * p["annualized_return_geo_cagr"],
                     "alpha_t": p["alpha_tstat"], "beta": p["beta"], "beta_t": p["beta_tstat"],
                     "max_dd_pct": 100 * p["max_drawdown"], "hit_rate": p["hit_rate_active"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rf2_costs.csv", index=False)
    active_mean = (stats["port_excess_ret"] - 0.04 / 12).mean()
    summ = {
        "avg_traded_notional_per_month_x_capital": float(np.mean(trade)),
        "avg_traded_notional_ex_initial_build": float(np.mean(trade[1:])),
        "min_traded_notional_ex_initial_build": float(np.min(trade[1:])),
        "max_traded_notional_ex_initial_build": float(np.max(trade[1:])),
        "annual_turnover_x_capital": float(np.mean(trade[1:]) * 12),
        "breakeven_one_way_cost_bps_for_ir_zero_no_borrow": float(1e4 * active_mean / np.mean(trade)),
        "note": "IR=0 means net excess return equals the 4%/yr hurdle; cost applied to every $ traded incl. initial build",
    }
    print("\n=== TRANSACTION COSTS (tuned wide-grid Modern book) ===")
    print(json.dumps(_clean(summ), indent=2))
    show(df[df["borrow_bps_annual"].isin([0, 50])], ["one_way_cost_bps", "borrow_bps_annual", "ir", "sharpe", "cagr_pct",
                                                      "alpha_t", "beta", "max_dd_pct"])
    save_stage("costs", {"summary": summ, "table": df.to_dict(orient="records")})


def long_only_stats(preds, arm):
    """Top-N / bottom-N equal-weight books inside the same investable screen the LP uses (N = 1/MAX_WEIGHT,
    which is exactly what the LP returns for a long-only, cap-1%, no-beta-constraint book)."""
    n_pick = int(1.0 / MAX_WEIGHT)
    recs = []
    for month, g in preds.groupby("target_month"):
        g = g[(g["raw_dolvol_126d"] >= MIN_DOLLAR_VOLUME)].dropna(subset=["raw_beta_60m"])
        if len(g) < 2 * n_pick:
            continue
        g = g.sort_values("pred", ascending=False)
        dec = max(1, len(g) // 10)
        recs.append({
            "target_month": month, "top": g[TARGET_COL].iloc[:n_pick].mean(), "bottom": g[TARGET_COL].iloc[-n_pick:].mean(),
            "universe": g[TARGET_COL].mean(), "top_decile": g[TARGET_COL].iloc[:dec].mean(),
            "bottom_decile": g[TARGET_COL].iloc[-dec:].mean(), "n_universe": len(g),
        })
    m = pd.DataFrame(recs).sort_values("target_month").reset_index(drop=True)
    tb3ms, sp = load_fred_series()
    m = m.merge(tb3ms, left_on="target_month", right_on="month", how="left").merge(
        sp, left_on="target_month", right_on="month", how="left", suffixes=("", "_sp"))

    def t_of(s):
        return float(s.mean() / s.std() * np.sqrt(len(s)))

    def cagr(s):
        return float((1 + s).prod() ** (12 / len(s)) - 1)

    mkt_x = (m["sp500_ret"] - m["rf_monthly"])
    ok = mkt_x.notna()
    reg = sm.OLS(m.loc[ok, "top"], sm.add_constant(mkt_x[ok])).fit()
    act = m["top"] - m["universe"]
    spread = m["top"] - m["bottom"]
    dspread = m["top_decile"] - m["bottom_decile"]
    return {
        "arm": arm, "n_months": len(m),
        "top100_long_only_cagr_pct": 100 * cagr(m["top"]), "universe_ew_cagr_pct": 100 * cagr(m["universe"]),
        "bottom100_cagr_pct": 100 * cagr(m["bottom"]),
        "top100_minus_universe_ann_pct": 100 * 12 * act.mean(), "top100_minus_universe_t": t_of(act),
        "top100_minus_universe_ir": float(np.sqrt(12) * act.mean() / act.std()),
        "top100_beats_universe_share": float((act > 0).mean()),
        "bottom100_minus_universe_ann_pct": 100 * 12 * (m["bottom"] - m["universe"]).mean(),
        "top_minus_bottom100_ann_pct": 100 * 12 * spread.mean(), "top_minus_bottom100_t": t_of(spread),
        "decile_spread_ann_pct": 100 * 12 * dspread.mean(), "decile_spread_t": t_of(dspread),
        "top100_beta_vs_sp500": float(reg.params.iloc[1]), "top100_beta_t": float(reg.tvalues.iloc[1]),
    }


def stage_longonly(ctx, grid, args):
    orig_keys = [cfg_key(c) for c in ORIG_CONFIGS if cfg_key(c) in grid]
    pred_mod, _, _ = select_tuned(ctx, grid, orig_keys)
    print("[longonly] fitting Graham arm (rf.py's 2-config grid)")
    g_grid = run_grid(ctx, GRAHAM_FEATURES, ORIG_CONFIGS, n_est=grid_trees(args), label="graham")
    pred_gra, _, _ = select_tuned(ctx, g_grid, [cfg_key(c) for c in ORIG_CONFIGS])
    rows = []
    for arm, pred in (("graham", pred_gra), ("modern", pred_mod)):
        preds = ctx.preds_frame(pred)
        lo = long_only_stats(preds, arm)
        book = evaluate(preds)
        lo.update({"lp_book_ir": book["ir"], "lp_book_long_leg_cagr_pct": book["long_leg_cagr_pct"],
                   "lp_book_short_leg_cagr_pct": book["short_leg_cagr_pct"], "lp_book_cagr_pct": book["cagr_pct"],
                   "rank_ic": book["rank_ic"]})
        rows.append(lo)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rf2_long_only.csv", index=False)
    print("\n=== LONG-ONLY / LEG DIAGNOSTIC (rf.py grid, Graham vs Modern) ===")
    print(df.set_index("arm").T.to_string(float_format=lambda v: f"{v:.3f}"))
    save_stage("long_only", df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# 6. Deck exhibits for the tuned Modern book (docs/FIAM.md sec 12)
# ---------------------------------------------------------------------------

INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e6e6e3"
C_STRAT, C_SPX, C_BENCH = "#2a78d6", "#eb6834", "#8a8a86"  # blue / orange / gray


def _save(fig, path, foot):
    import matplotlib.pyplot as plt
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.012, foot, fontsize=7.5, color=INK2, ha="left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10)
    ax.grid(axis="y", color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def stage_deck(holdings, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    d = OUT / "rf2_deck"
    d.mkdir(exist_ok=True)
    perf = pd.read_csv(OUT / "portfolio_returns_beta_neutral_rf2_modern.csv", parse_dates=["target_month"])
    span = f"{perf['target_month'].min():%m/%Y}-{perf['target_month'].max():%m/%Y}"
    foot = f"{span}; gross of transaction costs; monthly rebalance; beta-neutral (beta_60m) book"

    # Convention (stated explicitly): strategy total return = TB3MS cash return + portfolio excess return.
    perf["strategy_total"] = perf["port_excess_ret"] + perf["rf_monthly"]
    perf["year"] = perf["target_month"].dt.year

    def comp(s):
        return (1 + s).prod() - 1

    cal = perf.groupby("year").agg(
        months=("target_month", "count"), strategy_excess=("port_excess_ret", comp), strategy_total=("strategy_total", comp),
        benchmark_tbill_plus_4=("benchmark_monthly", comp), sp500_price=("sp500_ret", comp),
        long_leg=("long_leg_ret", comp), short_leg=("short_leg_ret", comp)).reset_index()
    cal.to_csv(d / "calendar_year_returns.csv", index=False)

    # Top-10 long / short by average weight over the whole OOS window (0 in months not held).
    W = holdings.pivot_table(index="target_month", columns="permno", values="weight", fill_value=0.0)
    avg_w = W.mean().rename("avg_weight")
    months_held = (W != 0).sum().rename("months_held")
    # Label = most frequent "TICKER, Company Name" over the holding's months (panel labels; the deck's
    # naming rule still requires verifying historical labels against an authoritative source).
    holdings = holdings.assign(_lab=holdings["ticker"].fillna("(no ticker)").astype(str) + ", "
                               + holdings["company_name"].fillna("(no name)").astype(str))
    label_map = holdings.groupby("permno")["_lab"].agg(lambda s: s.mode().iloc[0])
    tbl = pd.concat([avg_w, months_held], axis=1).join(label_map.rename("holding"))
    top_long = tbl.sort_values("avg_weight", ascending=False).head(10)
    top_short = tbl.sort_values("avg_weight").head(10)
    pd.concat([top_long.assign(side="long"), top_short.assign(side="short")]).reset_index().to_csv(
        d / "top10_long_short_avg_weight.csv", index=False)

    # P&L contributors: sum over months of weight * realized return.
    holdings = holdings.assign(pnl=holdings["weight"] * holdings[TARGET_COL])
    pnl = holdings.groupby("permno")["pnl"].sum().rename("total_pnl_contribution").to_frame().join(label_map.rename("holding"))
    top_pos, top_neg = pnl.nlargest(10, "total_pnl_contribution"), pnl.nsmallest(10, "total_pnl_contribution")
    pd.concat([top_pos.assign(side="positive"), top_neg.assign(side="negative")]).reset_index().to_csv(
        d / "top10_pnl_contributors.csv", index=False)

    # Rolling 12-month exhibits.
    mkt_x = perf["sp500_ret"] - perf["rf_monthly"]
    roll = pd.DataFrame({"target_month": perf["target_month"]})
    roll["active_12m"] = (1 + perf["active_ret"]).rolling(12).apply(np.prod, raw=True) - 1
    roll["ir_12m"] = np.sqrt(12) * perf["active_ret"].rolling(12).mean() / perf["active_ret"].rolling(12).std()
    cov = perf["port_excess_ret"].rolling(12).cov(mkt_x)
    roll["beta_12m"] = cov / mkt_x.rolling(12).var()
    roll.to_csv(d / "rolling_12m.csv", index=False)

    # ---- charts (blue = strategy, orange = S&P 500, gray = benchmark; single-hue ink for text) ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = perf["target_month"]
    for col, lab, colr in (("strategy_total", "Strategy (T-bill + excess)", C_STRAT),
                           ("benchmark_monthly", "Benchmark (T-bill + 4%)", C_BENCH), ("sp500_ret", "S&P 500 (price)", C_SPX)):
        ax.plot(x, (1 + perf[col].fillna(0)).cumprod(), label=lab, color=colr, linewidth=2)
    _style(ax, "Cumulative growth of $1")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, d / "cumulative_return.png", foot)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(x, perf["drawdown"], 0, color=C_STRAT, alpha=0.35, linewidth=0, label="Strategy")
    ax.plot(x, perf["drawdown"], color=C_STRAT, linewidth=1.5)
    ax.plot(x, perf["sp500_drawdown"], color=C_SPX, linewidth=1.5, label="S&P 500 (price)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _style(ax, "Drawdown from prior peak (underwater)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    _save(fig, d / "underwater.png", foot)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
    axes[0].plot(roll["target_month"], roll["active_12m"], color=C_STRAT, linewidth=2)
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _style(axes[0], "Rolling 12-month active return vs T-bill + 4%")
    axes[1].plot(roll["target_month"], roll["ir_12m"], color=C_STRAT, linewidth=2)
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    _style(axes[1], "Rolling 12-month information ratio")
    _save(fig, d / "rolling_active_ir.png", foot)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(roll["target_month"], roll["beta_12m"], color=C_STRAT, linewidth=2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    _style(ax, "Rolling 12-month beta to S&P 500")
    _save(fig, d / "rolling_beta.png", foot)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(perf["port_excess_ret"], bins=18, color=C_STRAT, alpha=0.85, edgecolor="white")
    ax.axvline(0.04 / 12, color=C_SPX, linewidth=2, label="Monthly hurdle (4%/12 excess of T-bill)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _style(ax, "Distribution of monthly excess returns")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, d / "return_histogram.png", foot)

    contrib = pd.concat([top_pos.iloc[::-1], top_neg.iloc[::-1]])
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.barh(range(len(contrib)), contrib["total_pnl_contribution"],
            color=[C_STRAT if v > 0 else C_SPX for v in contrib["total_pnl_contribution"]])
    ax.set_yticks(range(len(contrib)))
    ax.set_yticklabels(contrib["holding"], fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    _style(ax, "Top-10 positive / negative P&L contributors")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID_C, linewidth=0.8)
    _save(fig, d / "pnl_contributors.png", foot + "; P&L = sum over months of weight x return")

    print(f"\n=== DECK EXHIBITS written to {d} ===")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTop 10 long (avg weight):\n", top_long.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nTop 10 short (avg weight):\n", top_short.to_string(float_format=lambda v: f"{v:.4f}"))
    save_stage("deck", {"calendar_year": cal.to_dict(orient="records"), "dir": str(d),
                        "rolling_beta_min": float(roll["beta_12m"].min()), "rolling_beta_max": float(roll["beta_12m"].max())})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

ALL_STAGES = ["grid", "loo", "seeds", "swaps", "beta", "costs", "longonly", "deck"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--n-draws", type=int, default=15, help="random draws for swaps and the random-15 null")
    ap.add_argument("--smoke", action="store_true", help="tiny fast plumbing test; writes to a scratch dir")
    args = ap.parse_args()
    stages = [s for s in args.stages.split(",") if s]
    unknown = set(stages) - set(ALL_STAGES)
    assert not unknown, f"unknown stages {unknown}"

    if args.smoke:
        OUT = BASE / "cache" / "rf2_smoke_output"
        OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors")
    print("Cross-sectional preprocessing...")
    df = cross_sectional_rank_transform(df, stock_vars)
    ctx = Context(df, stock_vars)
    del df
    print(f"  {len(ctx.folds)} folds, {len(ctx.meta):,} OOS rows")

    # Every stage needs the cached wide grid; the grid stage itself always (re)loads it and reports.
    grid = stage_grid(ctx, args)
    key_fixed, cfg_fixed = fixed_config_from_grid(grid)
    print(f"Fixed config for robustness stages (best mean validation MSE): {key_fixed}")

    tuned_pred, _, imp = select_tuned(ctx, grid, list(grid))
    if "grid" in stages:
        stage_grid_report(ctx, grid, args)
        stage_importance(imp)
    if "loo" in stages:
        stage_loo(ctx, grid, key_fixed, cfg_fixed, args)
    if "seeds" in stages:
        stage_seeds(ctx, grid, key_fixed, cfg_fixed, args)
    if "swaps" in stages:
        stage_swaps(ctx, grid, key_fixed, cfg_fixed, args)
    if "beta" in stages:
        stage_beta(ctx, grid, key_fixed, cfg_fixed, args)
    if {"costs", "deck"} & set(stages):
        _, holdings, stats, _ = evaluate(ctx.preds_frame(tuned_pred), tag="beta_neutral_rf2_modern", extras=True)
        if "costs" in stages:
            stage_costs(holdings, stats)
        if "deck" in stages:
            stage_deck(holdings, args)
    if "longonly" in stages:
        stage_longonly(ctx, grid, args)

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Results: {OUT / 'rf2_results.json'}")
