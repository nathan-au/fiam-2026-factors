"""
desk_26_multiple_testing - how much of the DEV evidence of this run survives multiple-testing corrections?

1  Rebuild the monthly DEV net-return series of every full-DEV book variant evaluated in desk_19..desk_24 (same code paths, no new variants).
2  Probability of backtest overfitting (Bailey et al. 2017, CSCV, 8 blocks -> 70 splits) over that matrix: if we pick the best-in-sample variant, how often
   is it below the median out of sample?
3  Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of the best DEV variant given all trials; and of the desk_25 monotone GBM (2017-2020 OOS) given
   the model-class trials.
4  Seed robustness of the monotone GBM (seeds 0..4, same fixed hyperparameters): OOS 2017-2020 IC per seed (diagnostic - not a selection).
5  BH-FDR over every paired test in the research ledger that has a paired_net_t (one-sided, H1: variant > its reference).
DEV only.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from fiam_desks import evaluate as E, factors_desk as FD  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH, timing as TM, construct as K  # noqa: E402

OUT = HERE / "output"
EXP = "desk_26_multiple_testing"


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    Tr, Tz, Dr = TH.theme_scores(ctx, "rank"), TH.theme_scores(ctx, "z"), TH.signed_scores(ctx, "rank")
    dtc = ctx.dtc_block(True)
    G = ctx.comp.extras["groups"].copy()
    G["dtc"] = dtc
    scores = {"V0_B1": core.b1_score(ctx), "V1_frozen7": ctx.comp.score, "V2_jkp13_rank": TH.composite_from_themes(Tr), "V3_jkp13_z": TH.composite_from_themes(Tz),
              "V4_jkp13_dtc": TH.composite_from_themes(Tr, extra={"dtc": dtc}), "V5_zoo146": Dr.mean(axis=1).to_numpy()}
    for k, T in (("a", G), ("b", Tz)):
        R = TM.theme_returns(ctx, T)
        scores[f"mom_drop_{k}"] = TM.weighted_composite(ctx, T, TM.momentum_weights(ctx, R, 12, 0.0))
        scores[f"mom_half_{k}"] = TM.weighted_composite(ctx, T, TM.momentum_weights(ctx, R, 12, 0.5))
        scores[f"invvol_{k}"] = TM.weighted_composite(ctx, T, TM.invvol_weights(ctx, R))
    Gf = ctx.comp.extras["groups"]
    qq = pd.Series(np.where(P.u, Gf["quality"], np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
    scores["M1_qualvalue"] = FD.add_groups(ctx.comp, {"qv": np.where(qq > 0.5, Gf["value"].to_numpy(), 0.0)}).score
    scores["M2_profxvalue"] = FD.add_groups(ctx.comp, {"pv": FD.urank(Gf["value"].to_numpy() * Gf["profitability"].to_numpy(), P)}).score
    iv = core.load_chars(ctx, ["ivol_capm_252d"])["ivol_capm_252d"]
    med = iv.where(P.u).groupby(P.df["eom"].to_numpy()).transform("median")
    wm = np.where(np.isfinite(np.clip((med / iv).to_numpy(), 0.5, 2)), np.clip((med / iv).to_numpy(), 0.5, 2), 1.0)
    jobs = [(k, s, "dev", core.b1_kw()) for k, s in scores.items()]
    for sig in ("V0_B1", "V3_jkp13_z"):
        s = scores[sig]
        jobs += [(f"{sig}_C1", s, "dev", core.b1_kw(extras={"w_mult": wm})), (f"{sig}_C2", s, "dev", core.b1_kw(cfg={"max_weight": 0.005})),
                 (f"{sig}_C5", s, "dev", core.b1_kw(si_cap=0.05)), (f"{sig}_C6", s, "dev", core.b1_kw(si_cap=0.15))]
    for x in (0.05, 0.20, 0.40):
        jobs.append((f"to{int(100 * x)}", scores["V0_B1"], "dev", core.b1_kw(cfg={"turnover": x})))
    B = core.run_many(ctx, jobs)
    M = pd.DataFrame({k: r["frame"].set_index("target_month")["net_port_excess_ret"] for k, r in B.items()})
    sc = TM.dispersion_gross_scale(ctx)
    M["disp_a"] = TM.apply_gross_scale(B["V0_B1"], sc)
    M["disp_b"] = TM.apply_gross_scale(B["V3_jkp13_z"], sc)
    M = M.dropna()
    M.to_csv(OUT / "dev_net_return_matrix.csv")
    A = M - core.RF_HALF
    irs = A.apply(ST.ir).sort_values(ascending=False)
    print(f"{M.shape[1]} full-DEV book variants, {M.shape[0]} months. Top 5 DEV net IR:\n", irs.head(5).round(3).to_string())
    pbo = ST.pbo_cscv(A.to_numpy(), 8)
    best = irs.index[0]
    srs = (A.mean() / A.std()).to_numpy()  # per-period SR of active returns
    dsr = ST.deflated_sr(A[best].to_numpy(), srs)
    res = {"n_variants": int(M.shape[1]), "pbo": pbo, "best_variant": best, "best_ir": float(irs.iloc[0]), "dsr_best": dsr,
           "share_variants_ir_gt_0": float((irs > 0).mean()), "ir_quantiles": irs.quantile([0.1, 0.5, 0.9]).round(3).to_dict()}
    print("PBO", pbo, "\nDSR best", dsr)
    # monotone GBM seeds + DSR on 2017-2020
    d25 = load(ROOT / "experiments/desk_25_model_classes/desk_25_model_classes.py")
    import xgboost as xgb
    D = Dr.to_numpy()
    u, tm = P.u, pd.DatetimeIndex(P.months)
    yr = pd.Series(P.y).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy() * 2 - 1
    oos = (tm.year >= 2017) & (tm.year <= 2020)
    seed_ic = {}
    seed_pred = {}
    for seed in range(5):
        pred = np.zeros(len(P.df))
        for Y in d25.YEARS:
            tr = u & (tm <= pd.Timestamp(f"{Y - 1}-11-30")) & np.isfinite(yr)
            te = u & (tm.year == Y)
            mdl = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.5, colsample_bytree=0.5, random_state=seed, n_jobs=6,
                                   monotone_constraints="(" + ",".join(["1"] * D.shape[1]) + ")", tree_method="hist")
            mdl.fit(D[tr], yr[tr])
            pred[te] = mdl.predict(D[te])
        ic = E.ic_series(P, pred, "dev", oos)
        seed_ic[seed] = {"ic": float(ic.mean()), "t": ST.tstat(ic), **{str(y): float(ic[str(y)].mean()) for y in d25.YEARS}}
        seed_pred[seed] = pred
        print("seed", seed, seed_ic[seed], flush=True)
    res["gbm_seed_ic_oos"] = seed_ic
    avg = np.mean([seed_pred[s] for s in seed_pred], axis=0)
    ic = E.ic_series(P, avg, "dev", oos)
    res["gbm_seed_avg_ic_oos"] = {"ic": float(ic.mean()), "t": ST.tstat(ic)}
    b = core.run_many(ctx, [("gbm0", np.where(oos, seed_pred[0], 0.0), "dev", core.b1_kw())])
    g = b["gbm0"]["frame"].set_index("target_month")["net_port_excess_ret"]["2017-01-31":] - core.RF_HALF
    # trials of the model-class family on the same window = the 5 rows of desk_25 (active-return IR / sqrt(12) = per-period SR)
    t25 = pd.read_csv(ROOT / "experiments/desk_25_model_classes/output/models_oos_2017_2020.csv")
    res["gbm_dsr_2017_2020"] = ST.deflated_sr(g.to_numpy(), (t25["ir_net_oos"] / np.sqrt(12)).to_numpy())
    # and the harsher version: trials = every DEV variant of the run (per-period SRs of the full-DEV matrix) + the 5 model rows
    res["gbm_dsr_2017_2020_all_trials"] = ST.deflated_sr(g.to_numpy(), np.concatenate([srs, (t25["ir_net_oos"] / np.sqrt(12)).to_numpy()]))
    res["gbm_boot_ir_2017_2020"] = ST.boot_ir(g.to_numpy())
    print("GBM 2017-2020 book:", res["gbm_boot_ir_2017_2020"], "DSR", res["gbm_dsr_2017_2020"])
    # BH over ledger paired tests
    Lg = pd.read_csv(ledger.PATH)
    Lg = Lg[pd.to_numeric(Lg["paired_net_t_vs_base"], errors="coerce").notna() & (Lg["eval_window"].astype(str).str.contains("dev"))]
    p = Lg["paired_net_t_vs_base"].astype(float).map(ST.p_from_t).to_numpy()
    Lg = Lg.assign(p=p, bh_q10=ST.bh_fdr(p, 0.10), bh_q20=ST.bh_fdr(p, 0.20))
    Lg[["experiment", "variant", "paired_net_t_vs_base", "p", "bh_q10", "bh_q20", "decision"]].to_csv(OUT / "ledger_bh.csv", index=False)
    res["ledger_paired_tests"] = int(len(Lg))
    res["bh_q10_discoveries"] = Lg.loc[Lg["bh_q10"], ["experiment", "variant", "paired_net_t_vs_base"]].to_dict("records")
    res["bh_q20_discoveries"] = Lg.loc[Lg["bh_q20"], ["experiment", "variant", "paired_net_t_vs_base"]].to_dict("records")
    print("BH q=0.10 discoveries:", res["bh_q10_discoveries"], "\nBH q=0.20:", res["bh_q20_discoveries"])
    ledger.record(EXP, "multiple_testing", "audit", "PBO/DSR/FDR over the run", {"n_variants": int(M.shape[1])}, "dev", "n/a", "dev", 1, {}, decision="diagnostic")
    E.dump(OUT / "results.json", res)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
