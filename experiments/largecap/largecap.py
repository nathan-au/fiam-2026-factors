"""
FIAM 2026 - Large-cap-only strategy, pre-registered (largecap.py).

WHY. docs/PM_ABLATION.md and docs/NEGATIVE_RESULT.md show that the five tree models' edge on the 147 characteristics
lives in sub-$5 / sub-$500M shorts and vanishes on the tradeable book. The one non-negative tradeable row was a
$2B market-cap floor (RF gross IR 0.67, most stable rolling beta) -- but it was one post-hoc row of a sweep and the
model was trained on ALL stocks. This script tests that idea properly: the universe, the factor set, the model
arms, the portfolio variants and the headline are FIXED BELOW BEFORE ANY RESULT WAS SEEN, and the model is TRAINED
and VALIDATED on the large-cap universe only.

PRE-REGISTERED DESIGN (do not change after looking at results; report deviations as such)
  Universe   price >= $5, market cap >= $2,000M, 126d dollar volume >= $10M, both beta_60m and betabab_1260d
             observed (as of the characteristic month). Sensitivity floor: --floor 1000 (labelled as such).
  Factors    18 characteristics chosen BY ECONOMIC GROUP from docs/FACTORS.md, not by IC (docs/FACTOR_FILTER.md
             sec 2: past IC does not persist in the investable universe). No momentum. See FACTOR_GROUPS.
  Arms       et          Extra-Trees on the 18 factors, trained + validated on universe rows only (HEADLINE model)
             comp        equal-weight composite of the same 18 factors with pre-specified signs (no fitting);
                         the honest "does a model beat a simple rule?" benchmark
             et_allrows  same Extra-Trees, trained on ALL stocks, scored on the universe (diagnostic: does
                         universe-restricted training matter?)
  Selection  hyperparameters per fold on VALIDATION mean monthly rank IC (validation rows = universe rows for
             `et`), never on test results. Label winsorized at the training fold's 1/99 percentiles.
  Portfolio  same LP as docs/RF_3.md `pm_*`: dollar-neutral, neutral to BOTH beta_60m and betabab_1260d, net sector
             <= 5% of NAV and gross sector share <= 35%, gross 200%, per-name cap 1%. HEADLINE = `lc_t10`
             (10% one-way turnover budget, from the tips; kept for consistency with earlier docs, not tuned).
             Sensitivities: `lc_free` (no cap), `lc_t20` (20%), `lc_t10_w05` (per-name cap 0.5% -> at least 400
             names, a smoother book than the LP's cap-bound corner solution).
  Reporting  gross AND net of the tiered cost assumptions of docs/PM_ABLATION.md; IR, beta, rolling beta, short
             book market cap; universe-only rank IC with paired tests; decile table; seed noise for the headline
             arm (--seeds).

Reuses the frozen harness in et.py (imported, not modified): data loading, rank transform, walk-forward folds,
rank-IC selection, LP, cost model, performance statistics.

Run:  .venv/bin/python largecap.py [--floor 2000] [--arms et,comp,et_allrows] [--seeds 5] [--smoke]
Outputs (output/): oos_predictions_lc_<arm>.csv, portfolio_{holdings,returns}_<variant>_lc_<arm>.csv,
    lc_feature_importance_<arm>.csv, lc_results.json, lc_summary.csv, lc_monthly_ic.csv, lc_deciles.csv
"""

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

import et  # frozen harness; nothing in et.py is edited

warnings.filterwarnings("ignore")
TARGET = et.TARGET_COL

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
    return et.monthly_rank_ic(p["target_month"].to_numpy(), p["pred"].to_numpy(), p[TARGET].to_numpy())


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
# Arms
# ---------------------------------------------------------------------------


def predict_arm(ctx, arm, floor, args):
    if arm == "comp":
        return composite_scores(ctx, ctx.meta, floor), pd.Series(0.0, index=FEATURES), None
    universe = "tradeable" if arm == "et" else "all"  # ctx.tradeable is overridden with the large-cap mask
    return et.run_walk_forward(ctx, FEATURES, f"lc/{arm}", winsor=not args.raw_target, train_universe=universe)


def run_arm(ctx, arm, floor, tag, variants, args, ic_store):
    print(f"\n{'=' * 78}\nLARGE-CAP | ARM {arm} | floor ${floor:,.0f}M | {len(FEATURES)} factors\n{'=' * 78}", flush=True)
    t0 = time.time()
    pred_vec, imp, fold_info = predict_arm(ctx, arm, floor, args)
    preds = et.preds_frame(ctx, pred_vec)
    preds.to_csv(et.OUT / f"oos_predictions_{tag}_{arm}.csv", index=False)
    if arm != "comp":
        imp.rename("importance").rename_axis("variable").to_csv(et.OUT / f"{tag}_feature_importance_{arm}.csv")

    ic = universe_ic(preds, floor)
    ic_store[arm] = ic
    u = preds[universe_mask(preds, floor)]
    dec = decile_table(preds, floor)
    model_stats = {
        "n_features": len(FEATURES), "universe_stocks_per_month": float(u.groupby("target_month").size().mean()),
        "universe_rank_ic": float(ic.mean()), "universe_rank_ic_t": tstat(ic),
        "universe_rank_ic_share_positive": float((ic > 0).mean()),
        "universe_ic_by_year": {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()},
        "universe_oos_r2_pct": None if arm == "comp" else 100 * et.oos_r2(u[TARGET].to_numpy(), u["pred"].to_numpy()),
        "mean_val_rank_ic": None if fold_info is None else float(np.mean([f["val_ic"] for f in fold_info])),
        "decile_top_minus_bottom_pct_per_month": float(dec.loc[10] - dec.loc[1]),
        "pred_rank_autocorr": et.pred_rank_autocorr(u),
    }
    print(f"  universe: {json.dumps(et._clean(model_stats))}", flush=True)
    print("  deciles (% / month): " + " ".join(f"{v:.2f}" for v in dec.to_numpy()), flush=True)

    out = {"features": FEATURES, "model_stats": model_stats, "fold_selection": fold_info,
           "deciles_pct_per_month": {int(k): float(v) for k, v in dec.items()},
           "top_importance": imp.head(10).to_dict() if arm != "comp" else None, "variants": {}}
    rows = []
    for name, cfg in variants.items():
        res = et.evaluate_variant(preds, name, cfg, tag, arm)
        out["variants"][name] = res
        rows.append(et.summary_row(tag, arm, name, res, {k: v for k, v in model_stats.items()
                                                       if not isinstance(v, dict)}))
        g, n = res["gross"], res["net"]
        print(f"  [{name:10s}] gross IR {g['information_ratio']:.2f} / net IR {n['information_ratio']:.2f} | "
              f"beta {g['beta']:+.2f} (t {g['beta_tstat']:+.2f}) roll12 [{g['rolling_beta_12m_min']:+.2f},"
              f"{g['rolling_beta_12m_max']:+.2f}] | turnover {100 * res['drift_adjusted_one_way_turnover_pct_of_gross']:.0f}% | "
              f"maxDD {100 * g['max_drawdown']:.0f}% | short med mcap ${res['short_book_median_market_cap_musd']:,.0f}M | "
              f"pos {g['min_n_positions']}-{g['max_n_positions']} | relaxed {res['months_turnover_cap_relaxed']}", flush=True)
    print(f"  arm done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out, rows


def run_seeds(ctx, floor, tag, variants, n_seeds, args):
    """Seed noise for the headline arm: refit `et` with other seeds, report universe IC and headline-variant IR."""
    rows = []
    for s in range(et.SEED + 1, et.SEED + n_seeds):
        et.SEED = s
        pred_vec, _, _ = et.run_walk_forward(ctx, FEATURES, f"lc/et seed {s}", winsor=not args.raw_target,
                                             train_universe="tradeable")
        preds = et.preds_frame(ctx, pred_vec)
        ic = universe_ic(preds, floor)
        res = et.evaluate_variant(preds, HEADLINE_VARIANT, variants[HEADLINE_VARIANT], tag, f"et_seed{s}")
        rows.append({"seed": s, "universe_rank_ic": float(ic.mean()), "ir_gross": res["gross"]["information_ratio"],
                     "ir_net": res["net"]["information_ratio"], "beta": res["gross"]["beta"],
                     "roll_beta_max": res["gross"]["rolling_beta_12m_max"]})
        print(f"  seed {s}: {rows[-1]}", flush=True)
    et.SEED = 42
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=2000.0, help="market-cap floor in $M (pre-registered: 2000)")
    ap.add_argument("--arms", default="et,comp,et_allrows")
    ap.add_argument("--seeds", type=int, default=1, help="total ET seeds for the headline arm (1 = seed 42 only)")
    ap.add_argument("--smoke", action="store_true", help="plumbing run: subsampled stocks, 1 fold, scratch output dir")
    ap.add_argument("--raw-target", action="store_true")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    floor = args.floor
    tag = "lc" if floor == 2000.0 else f"lc{int(floor)}"
    if args.smoke:
        et.OUT = et.OUTPUT / "_smoke_lc"
        et.OUT.mkdir(exist_ok=True)

    print("Loading model table...", flush=True)
    df, stock_vars = et.load_model_table()
    missing = [f for f in FEATURES if f not in stock_vars]
    assert not missing, f"factors not in the panel: {missing}"
    if args.smoke:
        df = df[df["permno"] % 3 == 0].reset_index(drop=True)
    print(f"  {len(df):,} stock-month rows", flush=True)
    df = et.cross_sectional_rank_transform(df, stock_vars)
    ctx = et.Context(df, stock_vars, max_folds=1 if args.smoke else None)
    ctx.tradeable = universe_mask(df, floor).to_numpy()  # training/validation universe for the `et` arm
    print(f"  universe rows (all months): {int(ctx.tradeable.sum()):,} of {len(df):,}", flush=True)
    del df

    variants = make_variants(floor)
    results, all_rows, ic_store = {}, [], {}
    for arm in args.arms.split(","):
        results[arm], rows = run_arm(ctx, arm, floor, tag, variants, args, ic_store)
        all_rows += rows
        (et.OUT / f"{tag}_results{args.suffix}.json").write_text(json.dumps(et._clean(results), indent=2))
        pd.DataFrame(all_rows).to_csv(et.OUT / f"{tag}_summary{args.suffix}.csv", index=False)

    pd.DataFrame(ic_store).rename_axis("target_month").to_csv(et.OUT / f"{tag}_monthly_ic{args.suffix}.csv")
    extra = {}
    if "et" in ic_store and "comp" in ic_store:
        extra["paired_ic_et_vs_comp"] = paired(ic_store["et"], ic_store["comp"])
    if "et" in ic_store and "et_allrows" in ic_store:
        extra["paired_ic_et_vs_et_allrows"] = paired(ic_store["et"], ic_store["et_allrows"])
    if extra:
        print("\nPaired monthly universe-IC differences:", json.dumps(extra, indent=1))
    if args.seeds > 1 and "et" in results:
        extra["seed_noise_headline"] = run_seeds(ctx, floor, tag, variants, args.seeds, args)
    results["_extra"] = extra
    (et.OUT / f"{tag}_results{args.suffix}.json").write_text(json.dumps(et._clean(results), indent=2))

    summ = pd.DataFrame(all_rows)
    cols = ["arm", "portfolio", "ir_gross", "ir_net", "sharpe_net", "cagr_net_pct", "beta", "roll_beta_min",
            "roll_beta_max", "turnover_one_way_pct_gross", "max_dd_net_pct", "universe_rank_ic", "universe_rank_ic_t",
            "short_median_mcap_musd"]
    print(f"\n{'=' * 78}\nSUMMARY large-cap (floor ${floor:,.0f}M)\n{'=' * 78}")
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
