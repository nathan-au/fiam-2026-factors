"""Deterministic PM, execution layer: the frozen LP of experiments/largecap (dollar-neutral, neutral to beta_60m and betabab_1260d, net sector
<= 5% NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover budget) with three optional per-row controls read from
the predictions frame, so the PM (fiam_desks/pm.py) can express desk decisions without touching the optimiser:

  veto_long, veto_short   bool columns: the name may not be held on that side
  w_mult                  float column (default 1): multiplies the per-name weight cap of that name (conviction dial)
  sir (+ cfg short_cap_val) shorts only where the short-interest ratio <= short_cap_val (NaN = unknown = allowed, as in feat_si_followup)

With none of these present the LP is the frozen one (checked by experiments/desk_00_infra_audit: composite reproduces lc_t10 IR).
The backtest statistics reuse the frozen functions of experiments/largecap (imported, not copied).
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from . import config as C

sys.path.insert(0, str(C.EXPERIMENTS / "largecap"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import largecap as LC  # noqa: E402  (frozen harness: cost model, performance statistics)

GROSS = C.GROSS


def _solve_lp(g, cfg, prev, mult, allow_long, allow_short, wmult):
    n = len(g)
    pred = g["pred"].to_numpy()
    turnover = cfg.get("turnover")
    use_turn = turnover is not None and mult is not None and len(prev) > 0
    a_prev, budget = None, None
    if use_turn:
        a_prev = prev.reindex(g["permno"]).fillna(0.0).to_numpy()
        dropped = float(prev[~prev.index.isin(g["permno"])].abs().sum())
        budget = mult * 2.0 * GROSS * turnover - dropped
        if budget < 0:
            return None

    def pad(M):
        M = sparse.csr_matrix(M)
        return sparse.hstack([M, sparse.csr_matrix((M.shape[0], n))], format="csr") if use_turn else M

    def row(a, b):
        return pad(sparse.csr_matrix(np.concatenate([a, b]).reshape(1, -1)))

    ones, zeros = np.ones(n), np.zeros(n)
    eq_rows, b_eq = [row(ones, -ones)], [0.0]
    for bc in list(cfg["beta_cols"]) + list(cfg.get("extra_neutral", ())):
        beta = g[bc].to_numpy(dtype=float)
        eq_rows.append(row(beta, -beta))
        b_eq.append(0.0)
    eq_rows.append(row(ones, ones))
    b_eq.append(GROSS)
    A_eq = sparse.vstack(eq_rows, format="csr")

    ub_rows, b_ub = [], []
    sec = g["sector"].to_numpy()
    for s in np.unique(sec):
        a = (sec == s).astype(float)
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
    ubl = cfg["max_weight"] * wmult * allow_long.astype(float)
    ubs = cfg["max_weight"] * wmult * allow_short.astype(float)
    bounds = list(zip(np.zeros(n), ubl)) + list(zip(np.zeros(n), ubs)) + ([(0.0, None)] * n if use_turn else [])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub if ub_rows else None, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[:n] - res.x[n:2 * n]


def build_portfolio(preds, cfg, verbose=True):
    holdings, monthly, notes = [], [], []
    prev = pd.Series(dtype=float)
    for month, grp in preds.groupby("target_month"):
        g = grp[(grp["raw_dolvol_126d"] >= cfg["min_dolvol"]) & (grp["raw_prc"].abs() >= cfg["min_price"]) & (grp["raw_me"] >= cfg["min_mcap"])]
        g = g.dropna(subset=list(cfg["beta_cols"]) + list(cfg.get("extra_neutral", ()))).reset_index(drop=True)
        n = len(g)
        if n < 2 * int(1.0 / cfg["max_weight"]):
            continue
        vl = g["veto_long"].fillna(False).to_numpy(bool) if "veto_long" in g else np.zeros(n, bool)
        vs = g["veto_short"].fillna(False).to_numpy(bool) if "veto_short" in g else np.zeros(n, bool)
        if cfg.get("short_cap_val") is not None and "sir" in g:
            vs = vs | (g["sir"].fillna(0.0).to_numpy() > cfg["short_cap_val"])
        wm = g["w_mult"].fillna(1.0).to_numpy(float) if "w_mult" in g else np.ones(n)
        w, used, vetoes_dropped = None, None, False
        for drop_vetoes in ((False, True) if (vl.any() or vs.any()) else (False,)):
            al, as_ = (np.ones(n, bool), np.ones(n, bool)) if drop_vetoes else (~vl, ~vs)
            for mult in C.RELAX_LADDER:
                w = _solve_lp(g, cfg, prev, mult, al, as_, wm)
                if w is not None:
                    used = mult
                    break
            if w is not None:
                vetoes_dropped = drop_vetoes
                break
        if w is None:
            if verbose:
                print(f"  [warn] LP infeasible for {month.date()}, skipping month")
            continue
        if vetoes_dropped:
            notes.append(str(month.date()))
        relaxed = (cfg.get("turnover") is not None) and (used != 1.0) and len(prev) > 0
        keep = np.abs(w) > 1e-8
        both = g.loc[keep].copy()
        both["weight"] = w[keep]
        both["target_month"] = month
        holdings.append(both)
        prev = pd.Series(both["weight"].to_numpy() * (1.0 + both[C.TARGET_COL].to_numpy()), index=both["permno"].to_numpy())
        long_mask = both["weight"] > 0
        sec_net = both.groupby("sector")["weight"].sum().abs()
        monthly.append({
            "target_month": month, "port_excess_ret": (both["weight"] * both[C.TARGET_COL]).sum(),
            "long_leg_ret": (both.loc[long_mask, "weight"] * both.loc[long_mask, C.TARGET_COL]).sum(),
            "short_leg_ret": (both.loc[~long_mask, "weight"] * both.loc[~long_mask, C.TARGET_COL]).sum(),
            "gross_exposure": both["weight"].abs().sum(), "net_exposure": both["weight"].sum(),
            "beta_exposure": (both["weight"] * both["raw_beta_60m"]).sum(),
            "betabab_exposure": (both["weight"] * both["raw_betabab_1260d"]).sum(),
            "max_abs_sector_net": float(sec_net.max()),
            "max_sector_gross_share": float(both.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).max() / GROSS),
            "turnover_cap_relaxed": bool(relaxed), "turnover_relax_mult": (np.nan if used is None else float(used)),
            "n_long": int(long_mask.sum()), "n_short": int((~long_mask).sum()), "n_positions": int(keep.sum()),
        })
    hold = pd.concat(holdings, ignore_index=True)
    stats = pd.DataFrame(monthly).sort_values("target_month").reset_index(drop=True)
    stats.attrs["veto_dropped_months"] = notes
    return hold, stats


def evaluate_book(preds, cfg, keep_frames=False):
    """One LP book, gross and net of the frozen tiered cost assumptions. Returns a compact dict (+ frames if asked)."""
    holdings, stats = build_portfolio(preds, cfg)
    gross_perf, gross_frame = LC.compute_performance(stats)
    trade, tcost = LC.trade_frame(holdings)
    borrow = LC.borrow_series(holdings)
    ns = stats.copy()
    ns["trade_cost"] = tcost.reindex(ns["target_month"]).to_numpy()
    ns["borrow_cost"] = borrow.reindex(ns["target_month"]).fillna(0.0).to_numpy()
    ns["traded_notional"] = trade.reindex(ns["target_month"]).to_numpy()
    ns["port_excess_ret"] = stats["port_excess_ret"].to_numpy() - ns["trade_cost"] - ns["borrow_cost"]
    net_perf, net_frame = LC.compute_performance(ns)
    tn = ns["traded_notional"].to_numpy()
    out = {
        "ir_gross": gross_perf["information_ratio"], "ir_net": net_perf["information_ratio"],
        "sharpe_net": net_perf["sharpe_ratio"], "cagr_net": net_perf["annualized_return_geo_cagr"],
        "cagr_gross": gross_perf["annualized_return_geo_cagr"], "alpha_t_net": net_perf["alpha_tstat"], "beta": gross_perf["beta"],
        "beta_t": gross_perf["beta_tstat"], "roll_beta_min": gross_perf["rolling_beta_12m_min"], "roll_beta_max": gross_perf["rolling_beta_12m_max"],
        "max_dd_net": net_perf["max_drawdown"], "max_dd_gross": gross_perf["max_drawdown"], "n_months": gross_perf["n_months"],
        "avg_gross": gross_perf["avg_gross_exposure"], "max_gross": gross_perf["max_gross_exposure"], "avg_net_exposure": gross_perf["avg_net_exposure"],
        "min_net_exposure": gross_perf["min_net_exposure"], "max_net_exposure": gross_perf["max_net_exposure"],
        "min_positions": gross_perf["min_n_positions"], "max_positions": gross_perf["max_n_positions"],
        "one_way_turnover": float(np.mean(tn[1:]) / (2.0 * GROSS)), "max_abs_beta_at_formation": float(stats["beta_exposure"].abs().max()),
        "max_abs_betabab_at_formation": float(stats["betabab_exposure"].abs().max()),
        "max_abs_sector_net": float(stats["max_abs_sector_net"].max()), "max_sector_gross_share": float(stats["max_sector_gross_share"].max()),
        "months_turnover_relaxed": int(stats["turnover_cap_relaxed"].sum()), "veto_dropped_months": len(stats.attrs.get("veto_dropped_months", [])),
        "calendar_year_net": net_perf["calendar_year_returns"], "cost_bp_per_month": float(1e4 * (ns["trade_cost"].mean() + ns["borrow_cost"].mean())),
        "long_leg_cagr": gross_perf["long_leg_cagr"], "short_leg_cagr": gross_perf["short_leg_cagr"],
    }
    if keep_frames:
        fr = gross_frame.copy()
        fr["net_port_excess_ret"] = ns["port_excess_ret"].to_numpy()
        fr["trade_cost"], fr["borrow_cost"] = ns["trade_cost"].to_numpy(), ns["borrow_cost"].to_numpy()
        out["frame"], out["holdings"] = fr, holdings
    return out


def constraint_report(res: dict) -> dict:
    """FIAM hard constraints (docs/FIAM.md sec 2) as pass/fail on a finished book."""
    return {
        "positions_100_500": bool(res["min_positions"] >= 100 and res["max_positions"] <= 500),
        "gross_le_200pct": bool(res["max_gross"] <= 2.0 + 1e-6),
        "net_within_pm50pct": bool(max(abs(res["min_net_exposure"]), abs(res["max_net_exposure"])) <= 0.5),
        "dollar_neutral": bool(max(abs(res["min_net_exposure"]), abs(res["max_net_exposure"])) <= 1e-6),
        "beta_neutral_at_formation": bool(res["max_abs_beta_at_formation"] <= 1e-6 and res["max_abs_betabab_at_formation"] <= 1e-6),
        "sector_net_le_5pct": bool(res["max_abs_sector_net"] <= 0.05 + 1e-6),
        "sector_gross_le_35pct": bool(res["max_sector_gross_share"] <= 0.35 + 1e-6),
    }
