"""
desk_28_posthoc_diagnostics - POST-HOC (after the desk_27 TEST look): explain the confirmation results. Selects nothing; adds no TEST arm (it reads the
saved desk_27 frames and re-runs the SAME A0/A1/A5 books only to get holdings).
  P1  A5 (JKP13 z) vs A0 (B1): is the Sharpe difference over DEV+TEST (139 months) distinguishable? paired block bootstrap of SR(a) - SR(b), variance ratio
  P2  where A5's lower volatility comes from: holdings style exposures, returns-based style loadings, short-leg vol, by period
  P3  why the 5% SI cap reversed: P&L of shorts with SI in (5%, 10%] by year (these are the names A1 bars and A0 may short)
  P4  the regime-flip table: every DEV finding of this run next to its TEST counterpart
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from fiam_desks import config as C, evaluate as E  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_28_posthoc_diagnostics"
D27 = ROOT / "experiments/desk_27_confirmation_test/output"


def fr(name):
    f = pd.read_csv(D27 / f"frame_{name}.csv", parse_dates=["target_month"]).set_index("target_month")
    return f


def main():
    t0 = time.time()
    res = {}
    a = pd.concat([fr("A5_dev"), fr("A5_test")])
    b = pd.concat([fr("A0_dev"), fr("A0_test")])
    na, nb = a["net_port_excess_ret"], b["net_port_excess_ret"]
    sr = lambda x: float(np.sqrt(12) * x.mean() / x.std())
    idx = ST.boot_idx(len(na), 5000, 4, 7)
    A, Bv = na.to_numpy(), nb.to_numpy()
    d = np.array([sr(pd.Series(A[i])) - sr(pd.Series(Bv[i])) for i in idx])
    res["P1_sharpe_diff_139m"] = {"sr_A5": sr(na), "sr_A0": sr(nb), "diff": sr(na) - sr(nb), "ci90": [float(np.quantile(d, .05)), float(np.quantile(d, .95))],
                                  "p_le_0": float((d <= 0).mean()), "vol_A5": float(na.std() * np.sqrt(12)), "vol_A0": float(nb.std() * np.sqrt(12)),
                                  "mean_A5": float(12 * na.mean()), "mean_A0": float(12 * nb.mean()), "corr": float(na.corr(nb)),
                                  "by_period": {p: {"sr_A5": sr(fr(f"A5_{p}")["net_port_excess_ret"]), "sr_A0": sr(fr(f"A0_{p}")["net_port_excess_ret"]),
                                                    "vol_A5": float(fr(f"A5_{p}")["net_port_excess_ret"].std() * np.sqrt(12)), "vol_A0": float(fr(f"A0_{p}")["net_port_excess_ret"].std() * np.sqrt(12))}
                                                for p in ("dev", "test")}}
    print("P1", json.dumps(res["P1_sharpe_diff_139m"], indent=1))
    # P2 / P3 need holdings: re-run the same books (identical to desk_27 arms; deterministic)
    ctx = core.research_ctx(text=False)
    P = ctx.P
    b1 = core.b1_score(ctx)
    j13 = TH.theme_scores(ctx, "z").mean(axis=1).to_numpy()
    jobs = [(f"{k}_{p}", s, p, core.b1_kw()) for k, s in (("A0", b1), ("A5", j13)) for p in ("dev", "test")]
    B = core.run_many(ctx, jobs)
    for k in ("A0_dev", "A0_test"):
        chk = abs(B[k]["frame"]["net_port_excess_ret"].sum() - fr(k)["net_port_excess_ret"].sum()) < 1e-9
        assert chk, "re-run differs from desk_27"
    p2 = {}
    for k, r in B.items():
        s = b1 if k.startswith("A0") else j13
        m = core.metrics(ctx, r, s, k.split("_")[1], boot=False)
        f_ = r["frame"]
        p2[k] = {"holdings_style_exposure": {x: round(v, 3) for x, v in m["holdings_style_exposure"].items()}, "style_loadings": m["style_attr"]["loadings"],
                 "style_alpha_t": m["style_attr"]["alpha_t"], "style_r2": m["style_attr"]["r2"], "long_leg_vol": float(f_["long_leg_ret"].std() * np.sqrt(12)),
                 "short_leg_vol": float(f_["short_leg_ret"].std() * np.sqrt(12)), "legs_corr": float(f_["long_leg_ret"].corr(f_["short_leg_ret"]))}
        print("P2", k, p2[k], flush=True)
    res["P2"] = p2
    # P3: shorts with SI in (5%, 10%] in the A0 book
    sir = pd.Series(ctx.S["sir"])
    key = P.df[["permno", "eom"]].assign(sir=sir.to_numpy())
    p3 = {}
    for p in ("dev", "test"):
        h = B[f"A0_{p}"]["holdings"]  # carries the formation-month SI ratio used by the LP (column `sir`)
        sh = h[(h["weight"] < 0)]
        band = sh[(sh["sir"] > 0.05) & (sh["sir"] <= 0.10)]
        pnl = (band["weight"] * band[C.TARGET_COL]).groupby(band["target_month"].dt.year).sum()
        allsh = (sh["weight"] * sh[C.TARGET_COL]).groupby(sh["target_month"].dt.year).sum()
        gw = band["weight"].abs().groupby(band["target_month"]).sum().mean()
        p3[p] = {"pnl_by_year_si_5_10_shorts": pnl.round(4).to_dict(), "pnl_all_shorts_by_year": allsh.round(4).to_dict(), "avg_gross_weight_in_band": float(gw),
                 "avg_ret_of_band_shorts": float(band[C.TARGET_COL].mean()), "avg_ret_other_shorts": float(sh[~sh.index.isin(band.index)][C.TARGET_COL].mean())}
        top = (band.assign(pnl=band["weight"] * band[C.TARGET_COL]).groupby(["ticker"])["pnl"].sum().sort_values())
        p3[p]["worst5"] = top.head(5).round(4).to_dict()
        p3[p]["best5"] = top.tail(5).round(4).to_dict()
        print("P3", p, p3[p], flush=True)
    res["P3"] = p3
    # P4 regime-flip table (numbers from the experiment outputs of this run)
    flips = [
        ("composite (B1) universe IC", "+0.013 (t 0.87)", "+0.045 (t 2.39)", "same sign, 3.5x larger on TEST", "desk_17/27"),
        ("value group IC", "-0.016", "+0.050 (2021-22 +0.08)", "FLIP", "desk_17"),
        ("dtc block IC", "-0.001 (fresh 29m)", "+0.035 (feat_short_interest)", "FLIP / not replicated", "desk_14"),
        ("B1 IC high-SI minus low-SI tercile", "-0.050 (t -2.46)", "+0.050 (t +3.03)", "FLIP (significant both ways)", "desk_24/27 S4"),
        ("SI cap 5% vs 10% (paired net t)", "+3.19", "-2.00", "FLIP", "desk_20/27 A1"),
        ("SI cap 10% vs none", "+2.12 (fresh 29m)", "IR 0.584 -> 0.637, DD -18.9% -> -9.8% (feat_si_followup)", "same direction (risk)", "desk_14"),
        ("novneg_max IC, $0.5-2B band (filers)", "-0.022 (t -1.49, $1-2B)", "+0.038 (t +2.89)", "FLIP", "desk_02/27 S1"),
        ("novneg_max IC, universe", "+0.007 (t 1.64)", "+0.010 (t 2.87)", "same sign, small", "desk_12/23"),
        ("novneg_max -> next-month idio risk (partial ivol)", "+0.030 (t 4.65)", "+0.029 (t 4.71)", "SAME (robust)", "desk_23/27 S3"),
        ("monotone GBM IC", "+0.023 (2017-20, t 1.41)", "+0.017 (t 1.12)", "same sign, weak", "desk_25/27"),
        ("JKP13z vs B1 net Sharpe", "+0.21 vs -0.15", "+1.29 vs +0.99", "SAME (A5 better both)", "desk_19/27"),
        ("cross-sectional dispersion -> IC slope", "-0.048 (t -5.4)", "-0.024 (t -3.1)", "SAME", "desk_17"),
    ]
    res["P4_regime_flip_table"] = [dict(zip(["finding", "DEV", "TEST", "verdict", "source"], f)) for f in flips]
    pd.DataFrame(res["P4_regime_flip_table"]).to_csv(OUT / "regime_flip_table.csv", index=False)
    ledger.record(EXP, "posthoc", "diagnostics", "explain desk_27 (post-hoc, no selection)", {}, "n/a", "n/a", "dev+test (saved frames)", 1, {}, decision="descriptive",
                  reason="post-hoc diagnostics of the single TEST look; no new arm")
    E.dump(OUT / "results.json", res)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
