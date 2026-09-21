"""
FIAM 2026 - Hierarchical Risk Parity (HRP) sizing for the large-cap composite book (hrp.py).

QUESTION. largecap.py's headline book (composite signal, `lc_t10`) is sized by an LP whose linear objective pushes
positions to the per-name cap (roughly equal weight). Does risk-aware sizing -- Lopez de Prado's Hierarchical Risk
Parity -- lower volatility / drawdown without hurting neutrality or return? HRP is a SIZING method, not a signal: it
adds no information, so the expected effect is on risk, not on alpha.

PRE-REGISTERED DESIGN (fixed before any result was seen)
  Selection   The names each month are exactly those held by largecap.py's `lc_t10` book on the `comp` arm (which
              already satisfies universe, sector, beta and turnover rules). Only the SIZES change. Signals, universe,
              floors and the selection LP are not touched. Default floor $2B (--floor 1000 = sensitivity).
  Schemes     ew   equal weight within each leg                       (CONTROL: same names, same repair)
              ivp  inverse-volatility weight within each leg          (CONTROL: risk-aware sizing without clustering)
              hrp  HRP within each leg: 60-month trailing correlation of monthly excess returns (`ret_exc`, months up
                   to the characteristic month, all known at the rebalance date), distance sqrt((1-rho)/2), SINGLE
                   linkage, quasi-diagonalisation by dendrogram leaf order, recursive bisection with inverse-variance
                   cluster variance. No covariance inversion. No shrinkage. Pairwise-complete correlations (>= 48
                   obs), missing -> 0; missing volatility -> leg median. Held names all have >= 5y history (they
                   need beta_60m).
  Neutrality  Each leg's target sizes (sum 1 per leg => gross 200%) are repaired by the smallest L1 change that
              restores exactly the constraints of the selection LP: dollar-neutral, neutral to beta_60m AND
              betabab_1260d, net sector <= 5% NAV, sector gross share <= 35%, each name's side unchanged, per-name
              cap. The selection book's own weights are feasible, so the repair is always feasible.
              Headline per-name cap 2% (twice the selection LP's 1%, so sizes can actually differ); sensitivity 1%.
  Turnover    No turnover constraint in the repair (sizes move with the trailing window); realised turnover and
              tiered costs are reported and net metrics include them.
  Verdict     HRP is called useful only if, versus `ew` at the same cap, net Sharpe is not lower AND net max
              drawdown is shallower AND neutrality is not worse (|beta| t-stat, rolling-12m beta range). `hrp` vs
              `ivp` is reported separately: if `ivp` does as well, clustering adds nothing over simple risk scaling.
              Verdict is not tuned; nothing is re-run after seeing results.

Uses the frozen harness from et.py (copied in below, unchanged, so this file is standalone) and reads largecap.py's saved holdings.

Run:  .venv/bin/python experiments/hrp/hrp.py [--floor 2000] [--caps 0.02,0.01]
Outputs (output/): portfolio_{holdings,returns}_<scheme>_cap<bp>_<tag>_comp.csv, hrp_results.json, hrp_summary.csv
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent  # experiments/hrp/
OUTPUT = HERE / "output"
LC_OUTPUT = HERE.parent / "largecap" / "output"  # selection book and reference results come from largecap.py


# ---------------------------------------------------------------------------
# Inlined from et.py (copied verbatim so this file runs on its own; only the parts used here)
# ---------------------------------------------------------------------------


import statsmodels.api as sm
BASE = HERE.parents[1]  # project root: fiam/ data and cache/ are shared
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"  # downloaded external data only (FRED series)
OUT = OUTPUT  # rebound by --smoke so plumbing tests never touch real outputs
CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
TARGET_COL = "ret_exc_lead1m"
GROSS = 2.0

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


OUTPUT.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")
TARGET = TARGET_COL
WINDOW, MIN_OBS = 60, 48
SECTOR_NET, SECTOR_GROSS = 0.05, 0.70
BETA_COLS = ("raw_beta_60m", "raw_betabab_1260d")
SCHEMES = ("ew", "ivp", "hrp")
HEADLINE = ("hrp", 0.02)


# ---------------------------------------------------------------------------
# HRP (Lopez de Prado 2016)
# ---------------------------------------------------------------------------


def _cluster_var(cov: np.ndarray, idx) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(corr: np.ndarray, vol: np.ndarray) -> np.ndarray:
    n = len(vol)
    if n == 1:
        return np.ones(1)
    cov = corr * np.outer(vol, vol)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w / w.sum()


def leg_targets(scheme: str, window: pd.DataFrame, permnos) -> np.ndarray:
    """Target |weights| (sum 1) for one leg. `window` = months x permnos of trailing monthly excess returns."""
    n = len(permnos)
    if scheme == "ew" or n == 1:
        return np.full(n, 1.0 / n)
    R = window.reindex(columns=permnos)
    vol = R.std().where(R.count() >= MIN_OBS).to_numpy()
    vol = np.where(np.isfinite(vol) & (vol > 1e-4), vol, np.nan)
    vol = np.where(np.isnan(vol), np.nanmedian(vol) if np.isfinite(vol).any() else 0.1, vol)
    if scheme == "ivp":
        w = 1.0 / vol
        return w / w.sum()
    corr = R.corr(min_periods=MIN_OBS).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    return hrp_weights(corr, vol)


# ---------------------------------------------------------------------------
# Neutrality repair: smallest L1 change from the target sizes, all selection-LP constraints kept
# ---------------------------------------------------------------------------


def repair(g: pd.DataFrame, target: np.ndarray, sign: np.ndarray, cap: float):
    n = len(g)
    I = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((n, n))
    # variables: x (n sizes), dp (n), dm (n); x - dp + dm = target
    A_eq = [sparse.hstack([I, -I, I], format="csr")]
    b_eq = [target]
    long_row = np.concatenate([(sign > 0).astype(float), np.zeros(2 * n)])
    short_row = np.concatenate([(sign < 0).astype(float), np.zeros(2 * n)])
    A_eq += [sparse.csr_matrix(long_row.reshape(1, -1)), sparse.csr_matrix(short_row.reshape(1, -1))]
    b_eq += [np.array([1.0]), np.array([1.0])]
    for bc in BETA_COLS:
        row = np.concatenate([sign * g[bc].to_numpy(), np.zeros(2 * n)])
        A_eq.append(sparse.csr_matrix(row.reshape(1, -1)))
        b_eq.append(np.array([0.0]))
    ub_rows, b_ub = [], []
    sec = g["sector"].to_numpy()
    for s in np.unique(sec):
        m = (sec == s).astype(float)
        net = np.concatenate([sign * m, np.zeros(2 * n)])
        gross = np.concatenate([m, np.zeros(2 * n)])
        ub_rows += [net, -net, gross]
        b_ub += [SECTOR_NET, SECTOR_NET, SECTOR_GROSS]
    c = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)])
    res = linprog(c, A_ub=sparse.csr_matrix(np.array(ub_rows)), b_ub=b_ub, A_eq=sparse.vstack(A_eq, format="csr"),
                  b_eq=np.concatenate(b_eq), bounds=[(0.0, cap)] * n + [(0.0, None)] * (2 * n), method="highs")
    return res.x[:n] if res.success else None


# ---------------------------------------------------------------------------
# Portfolio evaluation from a holdings frame (same statistics as evaluate_variant)
# ---------------------------------------------------------------------------


def evaluate_holdings(h: pd.DataFrame, label: str, tag: str):
    rows = []
    for month, both in h.groupby("target_month"):
        lm = both["weight"] > 0
        sec_net = both.groupby("sector")["weight"].sum().abs()
        rows.append({
            "target_month": month, "port_excess_ret": (both["weight"] * both[TARGET]).sum(),
            "long_leg_ret": (both.loc[lm, "weight"] * both.loc[lm, TARGET]).sum(),
            "short_leg_ret": (both.loc[~lm, "weight"] * both.loc[~lm, TARGET]).sum(),
            "gross_exposure": both["weight"].abs().sum(), "net_exposure": both["weight"].sum(),
            "beta_exposure": (both["weight"] * both["raw_beta_60m"]).sum(),
            "betabab_exposure": (both["weight"] * both["raw_betabab_1260d"]).sum(),
            "max_abs_sector_net": float(sec_net.max()),
            "max_sector_gross_share": float(both.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).max() / GROSS),
            "n_long": int(lm.sum()), "n_short": int((~lm).sum()), "n_positions": int(len(both)),
        })
    stats = pd.DataFrame(rows).sort_values("target_month").reset_index(drop=True)
    gross_perf, gross_frame = compute_performance(stats)
    trade, tcost = trade_frame(h)
    borrow = borrow_series(h)
    ns = stats.copy()
    ns["trade_cost"] = tcost.reindex(ns["target_month"]).to_numpy()
    ns["borrow_cost"] = borrow.reindex(ns["target_month"]).fillna(0.0).to_numpy()
    ns["traded_notional"] = trade.reindex(ns["target_month"]).to_numpy()
    ns["port_excess_ret"] = stats["port_excess_ret"].to_numpy() - ns["trade_cost"] - ns["borrow_cost"]
    net_perf, net_frame = compute_performance(ns)
    frame = gross_frame.copy()
    frame["net_port_excess_ret"] = ns["port_excess_ret"].to_numpy()
    frame["trade_cost"], frame["borrow_cost"] = ns["trade_cost"].to_numpy(), ns["borrow_cost"].to_numpy()
    frame["net_rolling_beta_12m"] = net_frame["rolling_beta_12m"].to_numpy()
    frame.to_csv(OUT / f"portfolio_returns_{label}_{tag}_comp.csv", index=False)
    h.to_csv(OUT / f"portfolio_holdings_{label}_{tag}_comp.csv", index=False)
    tn = ns["traded_notional"].to_numpy()
    net_r = ns["port_excess_ret"]
    return {
        "gross": gross_perf, "net": net_perf,
        "net_ann_vol": float(net_r.std() * np.sqrt(12)),
        "max_abs_beta_exposure_at_formation": float(stats["beta_exposure"].abs().max()),
        "max_abs_betabab_exposure_at_formation": float(stats["betabab_exposure"].abs().max()),
        "max_abs_sector_net": float(stats["max_abs_sector_net"].max()),
        "max_sector_gross_share": float(stats["max_sector_gross_share"].max()),
        "drift_adjusted_one_way_turnover_pct_of_gross": float(np.mean(tn[1:]) / (2.0 * GROSS)),
        "avg_trade_cost_bp_of_nav_per_month": float(1e4 * ns["trade_cost"].mean()),
        "avg_borrow_cost_bp_of_nav_per_month": float(1e4 * ns["borrow_cost"].mean()),
        **compute_turnover_and_concentration(h), **compute_short_book_characteristics(h),
    }, frame


def summary_row(label, scheme, cap, res):
    g, n = res["gross"], res["net"]
    return {
        "scheme": scheme, "cap": cap, "label": label, "ir_gross": g["information_ratio"], "ir_net": n["information_ratio"],
        "sharpe_net": n["sharpe_ratio"], "cagr_net_pct": 100 * n["annualized_return_geo_cagr"],
        "ann_vol_net_pct": 100 * res["net_ann_vol"], "max_dd_net_pct": 100 * n["max_drawdown"],
        "alpha_t_net": n["alpha_tstat"], "beta": g["beta"], "beta_t": g["beta_tstat"],
        "roll_beta_min": g["rolling_beta_12m_min"], "roll_beta_max": g["rolling_beta_12m_max"],
        "turnover_one_way_pct_gross": 100 * res["drift_adjusted_one_way_turnover_pct_of_gross"],
        "max_position_weight_pct": 100 * res["max_position_weight_abs"], "top10_share_of_gross": res["avg_top10_share_of_gross"],
        "n_pos_min": g["min_n_positions"], "n_pos_max": g["max_n_positions"],
        "short_median_mcap_musd": res["short_book_median_market_cap_musd"],
        "max_abs_sector_net": res["max_abs_sector_net"], "max_sector_gross_share": res["max_sector_gross_share"],
    }


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a.to_numpy() - b.to_numpy())
    d = d[np.isfinite(d)]
    t = float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else float("nan")
    return {"mean_monthly_diff_net": float(d.mean()), "t": t, "share_a_better": float((d > 0).mean())}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=2000.0)
    ap.add_argument("--caps", default="0.02,0.01", help="per-name caps (headline = first)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "lc" if args.floor == 2000.0 else f"lc{int(args.floor)}"
    caps = [float(c) for c in args.caps.split(",")]
    if args.smoke:
        OUT = OUTPUT / "_smoke_hrp"
        OUT.mkdir(exist_ok=True)

    base = pd.read_csv(LC_OUTPUT / f"portfolio_holdings_lc_t10_{tag}_comp.csv", parse_dates=["target_month"],
                       dtype={"sector": str})
    months = sorted(base["target_month"].unique())
    if args.smoke:
        months = months[:8]
        base = base[base["target_month"].isin(months)]
    print(f"Selection book: {tag} composite lc_t10, {len(months)} months, {base['permno'].nunique()} distinct names", flush=True)

    r = pd.read_parquet(CHARS_FILE, columns=["permno", "eom", "ret_exc"])
    r["eom"] = pd.to_datetime(r["eom"])
    RET = r.pivot_table(index="eom", columns="permno", values="ret_exc").sort_index()
    del r

    out_rows, results, ret_store = [], {}, {}
    t0 = time.time()
    for scheme in SCHEMES:
        for cap in caps:
            label = f"{scheme}_cap{int(round(cap * 1000))}"
            frames, fallbacks = [], 0
            for m in months:
                g = base[base["target_month"] == m].reset_index(drop=True)
                c = (pd.Period(m, "M") - 1).to_timestamp("M")  # characteristic month: last month with known returns
                win = RET.loc[:c].tail(WINDOW)
                sign = np.where(g["weight"] > 0, 1.0, -1.0)
                target = np.zeros(len(g))
                for s in (1.0, -1.0):
                    idx = np.flatnonzero(sign == s)
                    target[idx] = leg_targets(scheme, win, g.loc[idx, "permno"].to_numpy())
                x = repair(g, target, sign, cap)
                if x is None:
                    fallbacks += 1
                    x = g["weight"].abs().to_numpy()
                g["weight"] = sign * x
                frames.append(g[np.abs(g["weight"]) > 1e-8])
            h = pd.concat(frames, ignore_index=True)
            res, frame = evaluate_holdings(h, label, tag)
            res["repair_fallback_months"] = fallbacks
            results[label] = res
            ret_store[label] = frame.set_index("target_month")["net_port_excess_ret"]
            row = summary_row(label, scheme, cap, res)
            out_rows.append(row)
            print(f"  [{label:11s}] gross IR {row['ir_gross']:.2f} / net IR {row['ir_net']:.2f} | net Sharpe {row['sharpe_net']:.2f} "
                  f"vol {row['ann_vol_net_pct']:.1f}% maxDD {row['max_dd_net_pct']:.0f}% | beta {row['beta']:+.2f} (t {row['beta_t']:+.2f}) "
                  f"roll12 [{row['roll_beta_min']:+.2f},{row['roll_beta_max']:+.2f}] | turnover {row['turnover_one_way_pct_gross']:.0f}% | "
                  f"maxW {row['max_position_weight_pct']:.2f}% top10 {row['top10_share_of_gross']:.2f} | fallback {fallbacks} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    # Reference: the selection book itself (largecap.py lc_t10, LP-sized)
    ref = pd.read_csv(LC_OUTPUT / f"portfolio_returns_lc_t10_{tag}_comp.csv", parse_dates=["target_month"]).set_index("target_month")
    ret_store["lp_lc_t10"] = ref["net_port_excess_ret"]
    ref_row = None
    lr = json.load(open(LC_OUTPUT / f"{tag}_results.json"))["comp"]["variants"]["lc_t10"]
    ref_row = {"scheme": "lp(lc_t10)", "cap": 0.01, "ir_gross": lr["gross"]["information_ratio"], "ir_net": lr["net"]["information_ratio"],
               "sharpe_net": lr["net"]["sharpe_ratio"], "cagr_net_pct": 100 * lr["net"]["annualized_return_geo_cagr"],
               "max_dd_net_pct": 100 * lr["net"]["max_drawdown"], "beta": lr["gross"]["beta"], "beta_t": lr["gross"]["beta_tstat"],
               "turnover_one_way_pct_gross": 100 * lr["drift_adjusted_one_way_turnover_pct_of_gross"]}
    out_rows.append(ref_row)

    tests = {}
    for cap in caps:
        k = int(round(cap * 1000))
        tests[f"cap{k}"] = {
            "hrp_minus_ew": paired(ret_store[f"hrp_cap{k}"], ret_store[f"ew_cap{k}"]),
            "hrp_minus_ivp": paired(ret_store[f"hrp_cap{k}"], ret_store[f"ivp_cap{k}"]),
            "ivp_minus_ew": paired(ret_store[f"ivp_cap{k}"], ret_store[f"ew_cap{k}"]),
        }
    print("\nPaired monthly net-return differences:", json.dumps(tests, indent=1), flush=True)

    summ = pd.DataFrame(out_rows)
    summ.to_csv(OUT / f"hrp_summary_{tag}.csv", index=False)
    (OUT / f"hrp_results_{tag}.json").write_text(json.dumps(_clean({"results": results, "paired_tests": tests}), indent=2))
    cols = ["scheme", "cap", "ir_gross", "ir_net", "sharpe_net", "cagr_net_pct", "ann_vol_net_pct", "max_dd_net_pct", "beta", "beta_t",
            "roll_beta_min", "roll_beta_max", "turnover_one_way_pct_gross", "max_position_weight_pct"]
    print(f"\n{'=' * 78}\nSUMMARY HRP sizing on the {tag} composite book\n{'=' * 78}")
    print(summ[[c for c in cols if c in summ]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
