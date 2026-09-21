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

Reuses the frozen harness in et.py (imported, not modified) and reads largecap.py's saved holdings.

Run:  .venv/bin/python hrp.py [--floor 2000] [--caps 0.02,0.01]
Outputs (output/): portfolio_{holdings,returns}_<scheme>_cap<bp>_<tag>_comp.csv, hrp_results.json, hrp_summary.csv
"""

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog
from scipy.spatial.distance import squareform

import et

warnings.filterwarnings("ignore")
TARGET = et.TARGET_COL
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
# Portfolio evaluation from a holdings frame (same statistics as et.evaluate_variant)
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
            "max_sector_gross_share": float(both.groupby("sector")["weight"].apply(lambda x: x.abs().sum()).max() / et.GROSS),
            "n_long": int(lm.sum()), "n_short": int((~lm).sum()), "n_positions": int(len(both)),
        })
    stats = pd.DataFrame(rows).sort_values("target_month").reset_index(drop=True)
    gross_perf, gross_frame = et.compute_performance(stats)
    trade, tcost = et.trade_frame(h)
    borrow = et.borrow_series(h)
    ns = stats.copy()
    ns["trade_cost"] = tcost.reindex(ns["target_month"]).to_numpy()
    ns["borrow_cost"] = borrow.reindex(ns["target_month"]).fillna(0.0).to_numpy()
    ns["traded_notional"] = trade.reindex(ns["target_month"]).to_numpy()
    ns["port_excess_ret"] = stats["port_excess_ret"].to_numpy() - ns["trade_cost"] - ns["borrow_cost"]
    net_perf, net_frame = et.compute_performance(ns)
    frame = gross_frame.copy()
    frame["net_port_excess_ret"] = ns["port_excess_ret"].to_numpy()
    frame["trade_cost"], frame["borrow_cost"] = ns["trade_cost"].to_numpy(), ns["borrow_cost"].to_numpy()
    frame["net_rolling_beta_12m"] = net_frame["rolling_beta_12m"].to_numpy()
    frame.to_csv(et.OUT / f"portfolio_returns_{label}_{tag}_comp.csv", index=False)
    h.to_csv(et.OUT / f"portfolio_holdings_{label}_{tag}_comp.csv", index=False)
    tn = ns["traded_notional"].to_numpy()
    net_r = ns["port_excess_ret"]
    return {
        "gross": gross_perf, "net": net_perf,
        "net_ann_vol": float(net_r.std() * np.sqrt(12)),
        "max_abs_beta_exposure_at_formation": float(stats["beta_exposure"].abs().max()),
        "max_abs_betabab_exposure_at_formation": float(stats["betabab_exposure"].abs().max()),
        "max_abs_sector_net": float(stats["max_abs_sector_net"].max()),
        "max_sector_gross_share": float(stats["max_sector_gross_share"].max()),
        "drift_adjusted_one_way_turnover_pct_of_gross": float(np.mean(tn[1:]) / (2.0 * et.GROSS)),
        "avg_trade_cost_bp_of_nav_per_month": float(1e4 * ns["trade_cost"].mean()),
        "avg_borrow_cost_bp_of_nav_per_month": float(1e4 * ns["borrow_cost"].mean()),
        **et.compute_turnover_and_concentration(h), **et.compute_short_book_characteristics(h),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=2000.0)
    ap.add_argument("--caps", default="0.02,0.01", help="per-name caps (headline = first)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "lc" if args.floor == 2000.0 else f"lc{int(args.floor)}"
    caps = [float(c) for c in args.caps.split(",")]
    if args.smoke:
        et.OUT = et.OUTPUT / "_smoke_hrp"
        et.OUT.mkdir(exist_ok=True)

    base = pd.read_csv(et.OUTPUT / f"portfolio_holdings_lc_t10_{tag}_comp.csv", parse_dates=["target_month"],
                       dtype={"sector": str})
    months = sorted(base["target_month"].unique())
    if args.smoke:
        months = months[:8]
        base = base[base["target_month"].isin(months)]
    print(f"Selection book: {tag} composite lc_t10, {len(months)} months, {base['permno'].nunique()} distinct names", flush=True)

    r = pd.read_parquet(et.CHARS_FILE, columns=["permno", "eom", "ret_exc"])
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
    ref = pd.read_csv(et.OUTPUT / f"portfolio_returns_lc_t10_{tag}_comp.csv", parse_dates=["target_month"]).set_index("target_month")
    ret_store["lp_lc_t10"] = ref["net_port_excess_ret"]
    ref_row = None
    lr = json.load(open(et.OUTPUT / f"{tag}_results.json"))["comp"]["variants"]["lc_t10"]
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
    summ.to_csv(et.OUT / f"hrp_summary_{tag}.csv", index=False)
    (et.OUT / f"hrp_results_{tag}.json").write_text(json.dumps(et._clean({"results": results, "paired_tests": tests}), indent=2))
    cols = ["scheme", "cap", "ir_gross", "ir_net", "sharpe_net", "cagr_net_pct", "ann_vol_net_pct", "max_dd_net_pct", "beta", "beta_t",
            "roll_beta_min", "roll_beta_max", "turnover_one_way_pct_gross", "max_position_weight_pct"]
    print(f"\n{'=' * 78}\nSUMMARY HRP sizing on the {tag} composite book\n{'=' * 78}")
    print(summ[[c for c in cols if c in summ]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
