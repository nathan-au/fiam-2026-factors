"""
FIAM 2026 - Total Portfolio Approach (TPA) adaptation, factors only (tpa.py).

TPA manages ONE total portfolio against one objective and gives each opportunity (sleeve) capital according to its
marginal contribution to the total's risk and return, instead of fixed silos or equal buckets. We hold a single
long/short equity book, so the sleeves here are the SEVEN pre-registered factor groups of largecap.py (value,
profitability, investment/issuance/accruals, quality, surprise, volatility/beta, liquidity). Question: does sizing
the sleeves by their contribution to total risk / risk-adjusted return beat largecap.py's equal-group composite?
(No 8-K sleeve yet: that plugs in later as an eighth sleeve.)

PRE-REGISTERED DESIGN (fixed before any result was seen)
  Universe / factors / portfolio LP   exactly largecap.py's (floor $2B default, --floor 1000 sensitivity; 18 factors;
                                      variants lc_free, lc_t20, lc_t10 [HEADLINE], lc_t10_w05).
  Sleeve signal    group score = mean of the group's signed within-universe monthly ranks (as in largecap.py),
                   centred within month and scaled to mean |score| = 1, so a unit of sleeve capital is a unit of
                   gross exposure whatever the group's size.
  Sleeve return    r[g,t] = sum_i z[g,i] * ret_exc_lead1m[i] / sum_i |z[g,i]| over the universe in target month t:
                   the return of a unit-gross, score-weighted long/short sleeve (not beta-neutral; used only for
                   allocation).
  Estimation       Each test year Y uses ONLY sleeve returns with target month < Y-01-01 (expanding window: 47 months
                   for 2021 ... 107 for 2026); sleeve weights are fixed within the year. No test-period return
                   influences any weight.
  Arms             tpa_eq   equal capital across sleeves (1/7)                              (CONTROL, ~ largecap comp)
                   tpa_erc  equal RISK contribution: sleeve weights with equal marginal-risk-weighted shares of total
                            sleeve-portfolio variance (Ledoit-Wolf covariance); uses no return forecasts
                   tpa_ms   marginal-Sharpe allocation: long-only max-Sharpe on sleeve returns, means shrunk 50%
                            toward the cross-sleeve mean, Ledoit-Wolf covariance, each sleeve <= 35%
  Composite        score_i = sum_g w_g * z[g,i]; then the same LP variants as largecap.py.
  Verdict          A TPA arm is called useful only if it beats `tpa_eq` on net IR in BOTH lc_t10 and lc_free AND the
                   paired monthly universe-IC difference has t >= 2. Between "beats on IR" and t < 2 it is reported
                   as suggestive only. Expectation before running: with 47-107 monthly observations for 7 sleeves,
                   estimation error is large and 1/N is hard to beat (DeMiguel et al.), and experiments/factor_filter/README.md sec 2
                   found factor IC does not persist in the investable universe. Not tuned; not re-run.

Reuses the frozen harness in et.py and the pre-registered definitions in largecap.py (imported, not modified).

Run:  .venv/bin/python experiments/tpa/tpa.py [--floor 2000] [--smoke]
Outputs (output/): oos_predictions_<tag>_tpa_<arm>.csv, portfolio_{holdings,returns}_<variant>_<tag>_tpa_<arm>.csv,
    tpa_results_<tag>.json, tpa_summary_<tag>.csv, tpa_weights_<tag>.csv, tpa_monthly_ic_<tag>.csv
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

HERE = Path(__file__).resolve().parent  # experiments/tpa/
OUTPUT = HERE / "output"
sys.path[:0] = [str(HERE.parent / "et"), str(HERE.parent / "largecap")]  # frozen harness + large-cap factor set
import et
import largecap as lc

et.OUT = OUTPUT  # importing largecap pointed et.OUT at its folder; write this experiment's files here instead

warnings.filterwarnings("ignore")
TARGET = et.TARGET_COL
GROUPS = list(lc.FACTOR_GROUPS)
MEAN_SHRINK, SLEEVE_CAP = 0.5, 0.35
ARMS = ("tpa_eq", "tpa_erc", "tpa_ms")


# ---------------------------------------------------------------------------
# Sleeve signals and returns (universe rows only)
# ---------------------------------------------------------------------------


def sleeve_frames(ctx, mask_all: np.ndarray):
    """Returns (Z_full [N_all x 7, zeros outside the universe], sleeve_returns [target month x 7])."""
    idx = np.flatnonzero(mask_all)
    D = pd.DataFrame(ctx.Xall[idx][:, ctx.cols(lc.FEATURES)], columns=lc.FEATURES)
    month = pd.Series(ctx.tm[idx], name="month")
    R = (D.groupby(month.to_numpy())[lc.FEATURES].rank(pct=True) * 2 - 1) * pd.Series(lc.SIGNS)
    G = pd.DataFrame({g: R[list(fs)].mean(axis=1) for g, fs in lc.FACTOR_GROUPS.items()})
    G = G - G.groupby(month.to_numpy()).transform("mean")
    G = G / G.abs().groupby(month.to_numpy()).transform("mean")
    Z_full = np.zeros((len(ctx.Xall), len(GROUPS)))
    Z_full[idx] = G.to_numpy()
    y = pd.Series(ctx.y[idx])
    num = G.mul(y.to_numpy(), axis=0).groupby(month.to_numpy()).sum()
    den = G.abs().groupby(month.to_numpy()).sum()
    return Z_full, (num / den)


# ---------------------------------------------------------------------------
# Allocation rules
# ---------------------------------------------------------------------------


def lw_cov(S: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(S.to_numpy()).covariance_


def risk_contributions(w, cov):
    mrc = cov @ w
    rc = w * mrc
    return rc / rc.sum()


def alloc_eq(S):
    return np.full(S.shape[1], 1.0 / S.shape[1])


def alloc_erc(S):
    cov = lw_cov(S)
    n = len(cov)
    # standard ERC objective: min 0.5 w'Cw - (1/n) sum log w  -> equal risk contributions once normalised
    res = minimize(lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n, np.full(n, 1.0 / n),
                   jac=lambda w: cov @ w - 1.0 / (n * w), bounds=[(1e-8, None)] * n, method="L-BFGS-B")
    w = res.x / res.x.sum()
    return w


def alloc_ms(S):
    cov = lw_cov(S)
    mu = S.mean().to_numpy()
    mu = (1 - MEAN_SHRINK) * mu + MEAN_SHRINK * mu.mean()
    n = len(mu)
    if not (mu > 0).any():
        return alloc_eq(S)
    res = minimize(lambda w: -(w @ mu) / np.sqrt(w @ cov @ w), np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, SLEEVE_CAP)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.clip(res.x, 0.0, None)
    return w / w.sum() if res.success and w.sum() > 0 else alloc_eq(S)


ALLOC = {"tpa_eq": alloc_eq, "tpa_erc": alloc_erc, "tpa_ms": alloc_ms}


def run_arms(ctx, Z_full, sleeves, floor, tag, args, ic_store):
    variants = lc.make_variants(floor)
    weights_rows, results, rows = [], {}, []
    for arm in ARMS:
        print(f"\n{'=' * 78}\nTPA | ARM {arm} | floor ${floor:,.0f}M | 7 factor-group sleeves\n{'=' * 78}", flush=True)
        t0 = time.time()
        pred = np.zeros(len(ctx.meta))
        for fold in ctx.folds:
            Y = fold["year"]
            S = sleeves[sleeves.index < pd.Timestamp(f"{Y}-01-01")]
            w = ALLOC[arm](S)
            rc = risk_contributions(w, lw_cov(S))
            weights_rows.append({"arm": arm, "test_year": Y, "n_sleeve_months": len(S),
                                 **{f"w_{g}": float(x) for g, x in zip(GROUPS, w)},
                                 **{f"rc_{g}": float(x) for g, x in zip(GROUPS, rc)}})
            pred[fold["sl"]] = Z_full[fold["te"]] @ w
        wdf = pd.DataFrame([r for r in weights_rows if r["arm"] == arm])
        print("  sleeve weights by test year:\n" + wdf[["test_year"] + [f"w_{g}" for g in GROUPS]].round(3).to_string(index=False), flush=True)
        preds = et.preds_frame(ctx, pred)
        preds.to_csv(et.OUT / f"oos_predictions_{tag}_{arm}.csv", index=False)
        ic = lc.universe_ic(preds, floor)
        ic_store[arm] = ic
        dec = lc.decile_table(preds, floor)
        u = preds[lc.universe_mask(preds, floor)]
        model_stats = {
            "universe_rank_ic": float(ic.mean()), "universe_rank_ic_t": lc.tstat(ic),
            "universe_rank_ic_share_positive": float((ic > 0).mean()),
            "universe_ic_by_year": {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()},
            "decile_top_minus_bottom_pct_per_month": float(dec.loc[10] - dec.loc[1]),
            "pred_rank_autocorr": et.pred_rank_autocorr(u),
        }
        print(f"  universe: {json.dumps(et._clean(model_stats))}", flush=True)
        out = {"model_stats": model_stats, "variants": {}}
        for name, cfg in variants.items():
            res = et.evaluate_variant(preds, name, cfg, tag, arm)
            out["variants"][name] = res
            rows.append(et.summary_row(tag, arm, name, res, {k: v for k, v in model_stats.items() if not isinstance(v, dict)}))
            g, n = res["gross"], res["net"]
            print(f"  [{name:10s}] gross IR {g['information_ratio']:.2f} / net IR {n['information_ratio']:.2f} | "
                  f"beta {g['beta']:+.2f} (t {g['beta_tstat']:+.2f}) roll12 [{g['rolling_beta_12m_min']:+.2f},"
                  f"{g['rolling_beta_12m_max']:+.2f}] | turnover {100 * res['drift_adjusted_one_way_turnover_pct_of_gross']:.0f}% | "
                  f"maxDD {100 * g['max_drawdown']:.0f}% | short med mcap ${res['short_book_median_market_cap_musd']:,.0f}M", flush=True)
        results[arm] = out
        print(f"  arm done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return results, rows, pd.DataFrame(weights_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=2000.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    floor = args.floor
    tag = "lc" if floor == 2000.0 else f"lc{int(floor)}"
    if args.smoke:
        et.OUT = OUTPUT / "_smoke_tpa"
        et.OUT.mkdir(exist_ok=True)

    print("Loading model table...", flush=True)
    df, stock_vars = et.load_model_table()
    if args.smoke:
        df = df[df["permno"] % 3 == 0].reset_index(drop=True)
    df = et.cross_sectional_rank_transform(df, stock_vars)
    ctx = et.Context(df, stock_vars, max_folds=2 if args.smoke else None)
    mask_all = lc.universe_mask(df, floor).to_numpy()
    del df
    Z_full, sleeves = sleeve_frames(ctx, mask_all)
    print(f"  sleeve return months: {len(sleeves)} ({sleeves.index.min().date()} .. {sleeves.index.max().date()})", flush=True)
    pre = sleeves[sleeves.index < pd.Timestamp("2021-01-01")]
    print("  pre-2021 sleeve Sharpe (ann.): " + ", ".join(f"{g[:8]} {np.sqrt(12) * pre[g].mean() / pre[g].std():+.2f}" for g in GROUPS), flush=True)

    ic_store = {}
    results, rows, wdf = run_arms(ctx, Z_full, sleeves, floor, tag, args, ic_store)
    wdf.to_csv(et.OUT / f"tpa_weights_{tag}.csv", index=False)
    pd.DataFrame(ic_store).rename_axis("target_month").to_csv(et.OUT / f"tpa_monthly_ic_{tag}.csv")
    paired_ic = {a: lc.paired(ic_store[a], ic_store["tpa_eq"]) for a in ("tpa_erc", "tpa_ms")}
    print("\nPaired monthly universe-IC differences vs tpa_eq:", json.dumps(paired_ic, indent=1), flush=True)
    (et.OUT / f"tpa_results_{tag}.json").write_text(json.dumps(et._clean({"results": results, "paired_ic_vs_eq": paired_ic}), indent=2))
    summ = pd.DataFrame(rows)
    summ.to_csv(et.OUT / f"tpa_summary_{tag}.csv", index=False)
    cols = ["arm", "portfolio", "ir_gross", "ir_net", "sharpe_net", "cagr_net_pct", "beta", "roll_beta_min", "roll_beta_max",
            "turnover_one_way_pct_gross", "max_dd_net_pct", "universe_rank_ic", "universe_rank_ic_t"]
    print(f"\n{'=' * 78}\nSUMMARY TPA (floor ${floor:,.0f}M)\n{'=' * 78}")
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
