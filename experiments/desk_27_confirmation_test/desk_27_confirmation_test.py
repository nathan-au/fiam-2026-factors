"""
desk_27_confirmation_test - the ONE pre-registered TEST confirmation of this research run. Arms, tests and rules: PREREGISTRATION.md (written first).
Run: .venv/bin/python experiments/desk_27_confirmation_test/desk_27_confirmation_test.py --confirm
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from fiam_desks import evaluate as E, lp as L, text_desk as TD  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_27_confirmation_test"


def walk_forward(ctx, D, years, kind, seeds=(0,)):
    import xgboost as xgb
    P = ctx.P
    u, tm = P.u, pd.DatetimeIndex(P.months)
    yr = pd.Series(P.y).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy() * 2 - 1
    pred = np.zeros(len(P.df))
    X = D.to_numpy()
    for Y in years:
        tr = u & (tm <= pd.Timestamp(f"{Y - 1}-11-30")) & np.isfinite(yr)
        te = u & (tm.year == Y)
        if kind == "ridge":
            n_m = len(np.unique(tm[tr]))
            b = np.linalg.solve(X[tr].T @ X[tr] + 1e4 * n_m * np.eye(X.shape[1]), X[tr].T @ yr[tr])
            pred[te] = X[te] @ b
        else:
            ps = []
            for sd in seeds:
                m = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.5, colsample_bytree=0.5, random_state=sd, n_jobs=7,
                                     monotone_constraints="(" + ",".join(["1"] * X.shape[1]) + ")", tree_method="hist")
                m.fit(X[tr], yr[tr])
                ps.append(m.predict(X[te]))
            pred[te] = np.mean(ps, axis=0)
        print(f"  {kind} {Y}: train rows {int(tr.sum())}", flush=True)
    return pred


def band_ic(P, raw_df, score_raw, sign, lo, hi, period):
    """IC of sign*rank(score) within a market-cap band [lo, hi) using the universe's other screens (price, dollar volume, betas)."""
    df = P.df
    band = ((df["raw_prc"].abs() >= 5) & (df["raw_dolvol_126d"] >= 1e7) & df["raw_beta_60m"].notna() & df["raw_betabab_1260d"].notna()
            & (df["raw_me"] >= lo) & (df["raw_me"] < hi)).to_numpy() & (P.period == period) & np.isfinite(score_raw)
    d = pd.DataFrame({"m": P.months[band], "s": sign * score_raw[band], "y": P.y[band]})
    ic = d.groupby("m").apply(lambda g: np.corrcoef(rankdata(g["s"]), rankdata(g["y"]))[0, 1] if len(g) > 20 else np.nan, include_groups=False).dropna()
    return {"ic": float(ic.mean()), "t": ST.tstat(ic), "n_months": int(len(ic)), "avg_names": float(d.groupby("m").size().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    if not ap.parse_args().confirm:
        sys.exit("--confirm required: this is the single pre-registered TEST look")
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P, raw = ctx.P, ctx.raw
    Dr, Tz = TH.signed_scores(ctx, "rank"), TH.theme_scores(ctx, "z")
    years = list(range(2017, 2027))
    gbm = walk_forward(ctx, Dr, years, "gbm", seeds=(0, 1, 2, 3, 4))
    rdg = walk_forward(ctx, Dr, years, "ridge")
    b1 = core.b1_score(ctx)
    j13 = Tz.mean(axis=1).to_numpy()
    sir = ctx.S["sir"]
    bar_both = np.nan_to_num(sir, nan=0.0) > 0.10
    arms = {"A0": (b1, {}), "A1": (b1, {"si_cap": 0.05}), "A2": (b1, {"extras": {"veto_long": bar_both}}), "A3": (gbm, {}), "A4": (rdg, {}), "A5": (j13, {}),
            "A6": (gbm, {"si_cap": 0.05})}
    jobs = [(f"{a}_{p}", s, p, core.b1_kw(**kw)) for a, (s, kw) in arms.items() for p in ("dev", "test")]
    B = core.run_many(ctx, jobs)
    net = {k: r["frame"].set_index("target_month")["net_port_excess_ret"] for k, r in B.items()}
    walk = {"A3", "A4", "A6"}
    Lg = pd.read_csv(ledger.PATH)
    trial_srs = pd.to_numeric(Lg["ir_net"], errors="coerce").dropna().to_numpy() / np.sqrt(12)
    rows = []
    a0t = net["A0_test"]
    m0 = core.metrics(ctx, B["A0_test"], b1, "test", boot=False)
    for a, (s, kw) in arms.items():
        rt = B[f"{a}_test"]
        m = core.metrics(ctx, rt, s, "test", boot=True)
        ft = net[f"{a}_test"]
        pn = ST.paired(ft, a0t)
        bd = ST.boot_ir_diff((ft - core.RF_HALF).to_numpy(), (a0t - core.RF_HALF).reindex(ft.index).to_numpy())
        start = "2017-01-31" if a in walk else "2015-01-31"
        comb = pd.concat([net[f"{a}_dev"][start:], ft])
        comb0 = pd.concat([net["A0_dev"][start:], a0t])
        sh, sh0 = float(np.sqrt(12) * comb.mean() / comb.std()), float(np.sqrt(12) * comb0.mean() / comb0.std())
        cons = L.constraint_report(rt)
        ok_c = all(v for k, v in cons.items())
        is_signal = a in ("A3", "A4", "A5", "A6")
        rule = {"a_paired_t_ge_1": bool(pn["t"] >= 1.0) if a != "A0" else None,
                "b_ic_ok": bool(m["ic"] >= m0["ic"] - 0.005) if is_signal else True,
                "c_dd_and_constraints": bool(m["max_dd_net"] >= m0["max_dd_net"] - 0.05 and ok_c),
                "d_combined_sharpe_ge_A0": bool(sh >= sh0)}
        passed = a != "A0" and all(v for v in rule.values() if v is not None)
        dsr = ST.deflated_sr((ft - core.RF_HALF).to_numpy(), np.append(trial_srs, (ft - core.RF_HALF).mean() / (ft - core.RF_HALF).std()))
        row = {"arm": a, "test_ic": m["ic"], "test_ic_t": m["ic_t"], "test_ir_net": m["ir_net"], "test_ir_gross": m["ir_gross"], "test_sharpe": m["sharpe_net"],
               "test_max_dd": m["max_dd_net"], "test_beta_vw": m["beta_vw_mkt"], "test_turnover": m["one_way_turnover"], "positions": f"{m['min_positions']}-{m['max_positions']}",
               "paired_net_t": pn["t"] if a != "A0" else np.nan, "ir_diff_ci90": bd["ci90"] if a != "A0" else None, "combined_window_start": start, "combined_sharpe": sh,
               "combined_sharpe_A0": sh0, "dev_part_ir_net": ST.ir(net[f"{a}_dev"][start:] - core.RF_HALF), "boot_ir_ci90": m["boot_ir_net"]["ci90"], "dsr_all_trials": dsr["dsr"],
               "calendar_year_net": m["calendar_year_net"], "style_alpha_t": m["style_attr"]["alpha_t"], "style_r2": m["style_attr"]["r2"], "constraints": cons, "rule": rule,
               "decision": "reference" if a == "A0" else ("PASS" if passed else "fail")}
        rows.append(row)
        ledger.record(EXP, "confirmation", a, "pre-registered TEST confirmation (PREREGISTRATION.md)", {"arm": a, **{k: str(v) for k, v in kw.items() if k != "extras"}},
                      "dev", "dev", "test", len(arms), m, paired_t=row["paired_net_t"], decision=row["decision"], reason="single pre-registered TEST look")
        print(f"{a}: IC {m['ic']:+.4f} (t {m['ic_t']:+.2f}) | TEST IR net {m['ir_net']:+.3f} SR {m['sharpe_net']:+.2f} DD {100 * m['max_dd_net']:.1f}% | paired t {row['paired_net_t']:+.2f} "
              f"| IR diff CI {row['ir_diff_ci90']} | combined SR {sh:+.3f} vs A0 {sh0:+.3f} (from {start}) | DSR {dsr['dsr']:.3f} | {row['decision']} {rule}", flush=True)
    # A6 interpretation condition
    dec = {r["arm"]: r["decision"] for r in rows}
    a6_valid = dec["A1"] == "PASS" and dec["A3"] == "PASS"
    passing = [r for r in rows if r["decision"] == "PASS" and (r["arm"] != "A6" or a6_valid)]
    cand = max(passing, key=lambda r: r["combined_sharpe"])["arm"] if passing else "A0 (B1, no arm passed)"
    # standalone
    nov = raw["txt_v2_novel_negative_max"]
    S = {}
    S["S1_novneg_band_0.5_2B"] = band_ic(P, None, nov, -1, 500.0, 2000.0, "test")
    s_nov = TD.signal_score(P, raw, "novneg_max")
    ic = E.ic_series(P, s_nov, "test", P.terc == 2)
    S["S2_novneg_large_tercile"] = {"ic": float(ic.mean()), "t": ST.tstat(ic)}
    m = P.mask("test") & np.isfinite(nov) & (np.nan_to_num(raw["txt_n_filings"]) > 0)
    dev_ = pd.Series(P.y).groupby(P.df["eom"].to_numpy()).transform("mean").to_numpy()
    d = pd.DataFrame({"e": P.df["eom"].to_numpy()[m], "x": nov[m], "v": np.abs(P.y - dev_)[m], "iv": P.df["x_vol"].to_numpy()[m]})
    vv = []
    for e, g in d.groupby("e"):
        Z = np.column_stack([np.ones(len(g)), rankdata(g["iv"]) / len(g), rankdata(g["x"]) / len(g)])
        vv.append(np.linalg.lstsq(Z, rankdata(g["v"]) / len(g), rcond=None)[0][2])
    S["S3_novneg_risk_partial_ivol"] = {"coef": float(np.mean(vv)), "t": ST.tstat(pd.Series(vv))}
    sq = pd.Series(np.where(P.u, sir, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
    ia, ib = E.ic_series(P, b1, "test", sq > 2 / 3), E.ic_series(P, b1, "test", sq <= 1 / 3)
    dd = (ia - ib).dropna()
    S["S4_b1_ic_highSI_minus_lowSI"] = {"ic_high": float(ia.mean()), "ic_low": float(ib.mean()), "diff": float(dd.mean()), "t": ST.tstat(dd)}
    ic = E.ic_series(P, gbm, "test")
    S["S5_gbm_test_ic"] = {"ic": float(ic.mean()), "t": ST.tstat(ic), "by_year": E.by_year(ic)}
    rep = {"S1_novneg_band_0.5_2B": lambda v: v["ic"] > 0 and v["t"] >= 2, "S2_novneg_large_tercile": lambda v: v["ic"] > 0 and v["t"] >= 2,
           "S3_novneg_risk_partial_ivol": lambda v: v["coef"] > 0 and v["t"] >= 2, "S4_b1_ic_highSI_minus_lowSI": lambda v: v["diff"] < 0 and v["t"] <= -2,
           "S5_gbm_test_ic": lambda v: v["ic"] > 0 and v["t"] >= 2}
    for k, v in S.items():
        v["replicated"] = bool(rep[k](v))
        ledger.record(EXP, "confirmation_standalone", k, "pre-registered replication", {}, "dev", "dev", "test", len(S), {"ic": v.get("ic", v.get("coef", v.get("diff"))), "ic_t": v["t"]},
                      decision="replicated" if v["replicated"] else "not replicated", reason="single pre-registered TEST look")
        print(k, v, flush=True)
    for k, r in B.items():
        r["frame"].to_csv(OUT / f"frame_{k}.csv", index=False)
    pd.DataFrame(rows).drop(columns=["constraints", "rule", "calendar_year_net"]).to_csv(OUT / "confirmation_test.csv", index=False)
    np.save(OUT / "gbm_pred.npy", gbm)
    E.dump(OUT / "results.json", {"arms": rows, "standalone": S, "candidate": cand, "a6_interpretable": a6_valid})
    print("CANDIDATE:", cand, f"| {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
