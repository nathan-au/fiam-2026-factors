"""Causal regime / timing tools (experiments/desk_21). Every quantity at formation month-end t uses only target months <= t (realised by t).

theme_returns(ctx, T)        equal-weight top-tercile minus bottom-tercile next-month return of each theme score, universe rows, by target month
momentum_weights(R, ...)     Ehsani-Linnainmaa / Gupta-Kelly time-series factor momentum: w = 1 if trailing-L-month cumulative theme return > 0 else
                             `off` (0 = drop, 0.5 = shrink toward equal weight); before `L` months of history, equal weights
invvol_weights(R, ...)       1 / trailing sd of theme returns (risk balancing), normalised to mean 1
dispersion(ctx)              cross-sectional sd of the month's own return (ret_1_0) over universe rows, by FORMATION eom
"""

import numpy as np
import pandas as pd


def theme_returns(ctx, T: pd.DataFrame) -> pd.DataFrame:
    P = ctx.P
    u = P.u
    out = {}
    m = P.months[u]
    y = P.y[u]
    for th in T.columns:
        s = pd.Series(T[th].to_numpy()[u])
        q = s.groupby(m).rank(pct=True).to_numpy()
        d = pd.DataFrame({"m": m, "y": y, "q": q})
        out[th] = d[d.q > 2 / 3].groupby("m")["y"].mean() - d[d.q <= 1 / 3].groupby("m")["y"].mean()
    R = pd.DataFrame(out)
    R.index = pd.DatetimeIndex(R.index)
    return R.sort_index()


def _formation_index(ctx):
    """target months in the panel, and their formation eom"""
    tm = pd.DatetimeIndex(sorted(set(ctx.P.months)))
    return tm, tm - pd.offsets.MonthEnd(1)


def momentum_weights(ctx, R: pd.DataFrame, L: int = 12, off: float = 0.0) -> pd.DataFrame:
    tm, fe = _formation_index(ctx)
    W = {}
    for t, e in zip(tm, fe):
        h = R[R.index <= e].tail(L)
        if len(h) < L:
            W[t] = pd.Series(1.0, index=R.columns)
        else:
            cum = (1 + h).prod() - 1
            W[t] = pd.Series(np.where(cum > 0, 1.0, off), index=R.columns)
    return pd.DataFrame(W).T


def invvol_weights(ctx, R: pd.DataFrame, L: int = 24, min_obs: int = 12) -> pd.DataFrame:
    tm, fe = _formation_index(ctx)
    W = {}
    for t, e in zip(tm, fe):
        h = R[R.index <= e].tail(L)
        if len(h) < min_obs:
            W[t] = pd.Series(1.0, index=R.columns)
        else:
            v = 1 / h.std()
            W[t] = v / v.mean()
    return pd.DataFrame(W).T


def weighted_composite(ctx, T: pd.DataFrame, W: pd.DataFrame) -> np.ndarray:
    """score_i,t = sum_g W[t, g] T[i, g] / sum_g W[t, g]; rows with all-zero weights get the equal-weight composite."""
    tmonths = pd.DatetimeIndex(ctx.P.months)
    Wr = W.reindex(tmonths)[T.columns].to_numpy()
    num = (T.to_numpy() * Wr).sum(axis=1)
    den = Wr.sum(axis=1)
    eq = T.to_numpy().mean(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), eq)


def dispersion(ctx) -> pd.Series:
    P = ctx.P
    u = P.u
    d = pd.DataFrame({"e": P.df["eom"].to_numpy()[u], "r": P.df["raw_ret_1_0"].to_numpy()[u]})
    return d.groupby("e")["r"].std()


def dispersion_gross_scale(ctx, pct: float = 0.8, low: float = 0.5, min_hist: int = 24) -> pd.Series:
    """Gross multiplier by TARGET month: `low` when the formation month's dispersion exceeds the `pct` quantile of all formation months strictly
    before it (expanding, >= min_hist months), else 1."""
    d = dispersion(ctx).sort_index()
    out = {}
    for i, (e, v) in enumerate(d.items()):
        hist = d.iloc[:i]
        t = e + pd.offsets.MonthEnd(1)
        out[t] = low if (len(hist) >= min_hist and v > hist.quantile(pct)) else 1.0
    return pd.Series(out)


def apply_gross_scale(res: dict, scale: pd.Series, cost_bp: float = 10.0) -> pd.Series:
    """Net monthly excess return of a book whose weights are multiplied by scale_t (gross <= 200% always). Costs: the book's own trade and borrow cost
    scale with it; changing the scale trades |d scale| x gross x cost_bp one way (charged)."""
    fr = res["frame"].set_index("target_month")
    k = scale.reindex(fr.index).fillna(1.0)
    gross_ret = fr["port_excess_ret"] * k
    costs = (fr["trade_cost"] + fr["borrow_cost"]) * k
    extra = k.diff().abs().fillna(0.0) * fr["gross_exposure"] * cost_bp / 1e4
    return gross_ret - costs - extra
