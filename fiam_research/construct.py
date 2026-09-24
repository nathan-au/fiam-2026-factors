"""Portfolio construction alternatives to the frozen LP (experiments/desk_20). Same screens, the same FIAM / LP constraint set (dollar neutral, neutral to
beta_60m and betabab_1260d, sector net <= 5%, sector gross <= 35% of book, gross = 200%, SI cap on shorts, per-name cap, 10% one-way turnover budget with
the same relax ladder), the same drift-then-rebalance accounting and the same cost model. Only the objective changes:

  qp_risk   maximise  alpha'w - lam * (w' X F X' w + sum_i s_i^2 w_i^2)
            alpha_i = IC_PRIOR * s_i * z_i  (Grinold: alpha = IC x residual vol x score; IC_PRIOR fixed a priori at 0.02, not fitted)
            s_i     = monthly idiosyncratic vol = ivol_capm_252d * sqrt(21) (known at formation)
            X       = 7 style exposures (all-stock rank columns of the panel: size, value, momentum, ivol, quality, profitability, investment)
            F       = covariance of monthly Fama-MacBeth returns of X over the trailing 36 target months REALISED BEFORE formation (min 12, else F = 0)
  qp_track  minimise ||w - k z||^2  (score-proportional weights, then projected onto the constraint set) - a 'no optimiser' construction

Everything is causal: F at formation month t uses FM returns of target months <= t (the month-t return is known at t's month-end).
"""

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from fiam_desks import config as C
from fiam_desks import lp as L

STYLE_COLS = ["x_size", "be_me", "x_mom", "x_vol", "qmj", "gp_at", "at_gr1"]
IC_PRIOR = 0.02


def style_fm_returns(ctx) -> pd.DataFrame:
    """Monthly FM factor returns of the 7 style exposures (universe rows), indexed by target month."""
    P = ctx.P
    u = P.u
    X = P.df.loc[u, STYLE_COLS].copy()
    X["x_size"] = X.groupby(P.df.loc[u, "eom"])["x_size"].rank(pct=True) * 2 - 1
    X = X.fillna(0.0)
    y, m = P.y[u], P.months[u]
    out = {}
    for t in np.unique(m):
        i = m == t
        Z = np.column_stack([np.ones(i.sum()), X.to_numpy()[i]])
        out[pd.Timestamp(t)] = np.linalg.lstsq(Z, y[i], rcond=None)[0][1:]
    return pd.DataFrame(out, index=STYLE_COLS).T.sort_index()


def factor_cov(fm: pd.DataFrame, month, window=36, min_obs=12):
    """Covariance of FM returns of target months strictly before `month` (the target month being formed)."""
    h = fm[fm.index < month].tail(window)
    if len(h) < min_obs:
        return np.zeros((fm.shape[1], fm.shape[1]))
    return np.cov(h.to_numpy().T)


def _solve_qp(g, cfg, prev, mult, allow_long, allow_short, wmult, F, mode, lam):
    n = len(g)
    z = g["pred"].to_numpy(dtype=float)
    z = (z - z.mean()) / (z.std() if z.std() > 0 else 1.0)
    s = g["idio_vol_m"].to_numpy(dtype=float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s))
    X = g[["e_" + c for c in STYLE_COLS]].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    wl, ws = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    w = wl - ws
    cons = [cp.sum(wl) - cp.sum(ws) == 0, cp.sum(wl) + cp.sum(ws) == L.GROSS]
    for bc in list(cfg["beta_cols"]) + list(cfg.get("extra_neutral", ())):
        b = g[bc].to_numpy(dtype=float)
        cons.append(b @ w == 0)
    sec = g["sector"].to_numpy()
    for sct in np.unique(sec):
        a = (sec == sct).astype(float)
        cons += [cp.abs(a @ w) <= cfg["sector_net"], a @ (wl + ws) <= cfg["sector_gross"]]
    cap = cfg["max_weight"] * wmult
    cons += [wl <= cap * allow_long, ws <= cap * allow_short]
    if cfg.get("turnover") is not None and mult is not None and len(prev) > 0:
        a_prev = prev.reindex(g["permno"]).fillna(0.0).to_numpy()
        dropped = float(prev[~prev.index.isin(g["permno"])].abs().sum())
        budget = mult * 2.0 * L.GROSS * cfg["turnover"] - dropped
        if budget < 0:
            return None
        cons.append(cp.norm1(w - a_prev) <= budget)
    if mode == "qp_risk":
        alpha = IC_PRIOR * s * z
        fexp = X.T @ w
        Fs = F + 1e-10 * np.eye(F.shape[0])
        risk = cp.quad_form(fexp, cp.psd_wrap(Fs)) + cp.sum_squares(cp.multiply(s, w))
        obj = cp.Maximize(alpha @ w - lam * risk)
    elif mode == "qp_track":
        k = L.GROSS / np.abs(z).sum()
        obj = cp.Minimize(cp.sum_squares(w - k * z))
    else:
        raise ValueError(mode)
    prob = cp.Problem(obj, cons)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or wl.value is None:
        return None
    out = wl.value - ws.value
    out[np.abs(out) < 1e-6] = 0.0
    return out


def build_portfolio_qp(preds, cfg, fm, mode="qp_risk", lam=1.0, verbose=False, tails=0.25):
    """Mirror of fiam_desks.lp.build_portfolio with the QP objective (same screens, vetoes, SI cap, relax ladder, drift accounting, stats)."""
    holdings, monthly = [], []
    prev = pd.Series(dtype=float)
    for month, grp in preds.groupby("target_month"):
        g = grp[(grp["raw_dolvol_126d"] >= cfg["min_dolvol"]) & (grp["raw_prc"].abs() >= cfg["min_price"]) & (grp["raw_me"] >= cfg["min_mcap"])]
        g = g.dropna(subset=list(cfg["beta_cols"])).reset_index(drop=True)
        n = len(g)
        if n < 2 * int(1.0 / cfg["max_weight"]):
            continue
        vl = g["veto_long"].fillna(False).to_numpy(bool) if "veto_long" in g else np.zeros(n, bool)
        vs = g["veto_short"].fillna(False).to_numpy(bool) if "veto_short" in g else np.zeros(n, bool)
        if cfg.get("short_cap_val") is not None and "sir" in g:
            vs = vs | (g["sir"].fillna(0.0).to_numpy() > cfg["short_cap_val"])
        wm = g["w_mult"].fillna(1.0).to_numpy(float) if "w_mult" in g else np.ones(n)
        if tails and "score_q" in g:
            # candidate sets with a hold band (Novy-Marx & Velikov buy/hold spread): new longs only in the top `tails`, existing longs kept while
            # the score stays in the top half; mirror for shorts
            q = g["score_q"].fillna(0.5).to_numpy()
            pw = prev.reindex(g["permno"]).fillna(0.0).to_numpy() if len(prev) else np.zeros(n)
            vl = vl | ~((q >= 1 - tails) | ((pw > 0) & (q >= 0.5)))
            vs = vs | ~((q <= tails) | ((pw < 0) & (q <= 0.5)))
        F = factor_cov(fm, month)
        w, used = None, None
        for mult in C.RELAX_LADDER:
            w = _solve_qp(g, cfg, prev, mult, (~vl).astype(float), (~vs).astype(float), wm, F, mode, lam)
            if w is not None:
                used = mult
                break
        if w is None:
            if verbose:
                print("  [warn] QP infeasible", month)
            continue
        keep = np.abs(w) > 1e-6
        both = g.loc[keep].copy()
        both["weight"] = w[keep]
        both["target_month"] = month
        holdings.append(both)
        prev = pd.Series(both["weight"].to_numpy() * (1.0 + both[C.TARGET_COL].to_numpy()), index=both["permno"].to_numpy())
        lm = both["weight"] > 0
        sec_net = both.groupby("sector")["weight"].sum().abs()
        monthly.append({
            "target_month": month, "port_excess_ret": (both["weight"] * both[C.TARGET_COL]).sum(),
            "long_leg_ret": (both.loc[lm, "weight"] * both.loc[lm, C.TARGET_COL]).sum(), "short_leg_ret": (both.loc[~lm, "weight"] * both.loc[~lm, C.TARGET_COL]).sum(),
            "gross_exposure": both["weight"].abs().sum(), "net_exposure": both["weight"].sum(), "beta_exposure": (both["weight"] * both["raw_beta_60m"]).sum(),
            "betabab_exposure": (both["weight"] * both["raw_betabab_1260d"]).sum(), "max_abs_sector_net": float(sec_net.max()),
            "max_sector_gross_share": float(both.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).max() / L.GROSS),
            "turnover_cap_relaxed": bool(cfg.get("turnover") is not None and used != 1.0 and len(holdings) > 1), "turnover_relax_mult": float(used) if used else np.nan,
            "n_long": int(lm.sum()), "n_short": int((~lm).sum()), "n_positions": int(keep.sum()),
        })
    hold = pd.concat(holdings, ignore_index=True)
    stats = pd.DataFrame(monthly).sort_values("target_month").reset_index(drop=True)
    stats.attrs["veto_dropped_months"] = []
    return hold, stats


def evaluate_with(preds, cfg, builder):
    """fiam_desks.lp.evaluate_book with an injected builder (identical post-processing and cost model)."""
    orig = L.build_portfolio
    L.build_portfolio = lambda p, c, verbose=True: builder(p, c)
    try:
        return L.evaluate_book(preds, cfg, keep_frames=True)
    finally:
        L.build_portfolio = orig


def qp_extras(ctx) -> dict:
    """Per-row columns the QP needs: style exposures (all-stock ranks; size re-ranked to [-1, 1]) and monthly idiosyncratic vol."""
    from . import core
    P = ctx.P
    ex = {}
    for c in STYLE_COLS:
        v = P.df[c]
        if c == "x_size":
            v = v.groupby(P.df["eom"]).rank(pct=True) * 2 - 1
        ex["e_" + c] = v.fillna(0.0).to_numpy()
    iv = core.load_chars(ctx, ["ivol_capm_252d"])["ivol_capm_252d"].to_numpy(dtype=float)
    ex["idio_vol_m"] = iv * np.sqrt(21.0)
    return ex


_FM = None


def tail_vetoes(ctx, score, tails=0.25):
    """Candidate sets: longs only in the top `tails` of the month's universe score distribution, shorts only in the bottom `tails` (keeps the QP
    at <= 2 * tails * N names, i.e. <= 500 for tails=0.25 and N <= 1,000... enforced ex post by the constraint report)."""
    P = ctx.P
    q = pd.Series(np.where(P.u, score, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
    return {"veto_long": ~(q >= 1 - tails), "veto_short": ~(q <= tails)}


def book_qp(ctx, score, period, mode="qp_risk", lam=20.0, fm=None, extras=None, tails=0.25, **kw):
    """Like core.book but with the QP builder; candidate names restricted to the score tails (position-count control)."""
    from . import core
    from fiam_desks import pm as PM
    P = ctx.P
    kw = core.b1_kw(**kw)
    S_ = ctx.S if kw["extended_si"] else ctx.S0
    ex = {"sir": S_["sir"], **qp_extras(ctx), **(extras or {})}
    q = pd.Series(np.where(P.u, score, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
    ex["score_q"] = q
    dec = {"pred": np.asarray(score, float), "veto_long": np.zeros(len(P.df), bool), "veto_short": np.zeros(len(P.df), bool), "w_mult": ex.pop("w_mult", np.ones(len(P.df)))}
    preds = PM.lp_frame(P, dec, period, ex)
    cfg = dict(C.LP_BASE)
    if kw["si_cap"] is not False:
        cfg["short_cap_val"] = C.SI_CAP if kw["si_cap"] is True else float(kw["si_cap"])
    cfg.update(kw.get("cfg") or {})
    fm = style_fm_returns(ctx) if fm is None else fm
    res = evaluate_with(preds, cfg, lambda p, c: build_portfolio_qp(p, c, fm, mode, lam, tails=tails))
    res["cfg"] = cfg
    return res


def transfer_coefficient(ctx, res, score, period) -> float:
    """Mean monthly cross-sectional correlation between held weights (0 for unheld universe names) and the score, over universe rows."""
    P = ctx.P
    m = P.mask(period)
    d = pd.DataFrame({"permno": P.df["permno"].to_numpy()[m], "target_month": P.months[m], "s": np.asarray(score)[m]})
    h = res["holdings"][["permno", "target_month", "weight"]]
    d = d.merge(h, on=["permno", "target_month"], how="left").fillna({"weight": 0.0})
    return float(d.groupby("target_month").apply(lambda g: np.corrcoef(g["s"], g["weight"])[0, 1], include_groups=False).mean())
