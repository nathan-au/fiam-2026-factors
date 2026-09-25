"""
desk_32 - SECOND pre-registered TEST look (PREREGISTRATION.md, written first): text-conditional factor weighting (Y1, Y2) + replications R1-R3.
R4 (LLM on TEST) is conditional on desk_31 H3 and is handled by desk_31 if it applies. Run with --confirm.
"""
import argparse, sys, importlib.util, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from fiam_desks import evaluate as E, lp as L  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402

OUT = HERE / "output"
EXP = "desk_32_text_confirmation"
spec = importlib.util.spec_from_file_location("fy", ROOT / "experiments/desk_30_text_untested_uses/followup_y1.py"); FY = importlib.util.module_from_spec(spec); spec.loader.exec_module(FY)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--confirm", action="store_true")
    if not ap.parse_args().confirm:
        sys.exit("--confirm required (second pre-registered TEST look)")
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P, raw = ctx.P, ctx.raw
    b1 = core.b1_score(ctx)
    S, fresh, merger = FY.y_scores(ctx)
    arms = {"A0": b1, "Y1": S["Y1_fresh_surprise_x2"], "Y2": S["Y2_Y1_plus_merger_noview"]}
    B = core.run_many(ctx, [(f"{k}_{p}", s, p, core.b1_kw()) for k, s in arms.items() for p in ("dev", "test")], workers=2)
    net = {k: r["frame"].set_index("target_month")["net_port_excess_ret"] for k, r in B.items()}
    m0 = core.metrics(ctx, B["A0_test"], b1, "test", boot=False)
    comb = lambda k: pd.concat([net[f"{k}_dev"], net[f"{k}_test"]])
    sh = lambda x: float(np.sqrt(12) * x.mean() / x.std())
    rows = []
    for k, s in arms.items():
        m = core.metrics(ctx, B[f"{k}_test"], s, "test", boot=True)
        pn = ST.paired(net[f"{k}_test"], net["A0_test"]) if k != "A0" else {"t": np.nan}
        bd = ST.boot_ir_diff((net[f"{k}_test"] - core.RF_HALF).to_numpy(), (net["A0_test"] - core.RF_HALF).to_numpy()) if k != "A0" else {"ci90": None}
        cons = all(L.constraint_report(B[f"{k}_test"]).values())
        rule = {"a_paired_t_ge_1.3": bool(pn["t"] >= 1.3), "b_ic": bool(m["ic"] >= m0["ic"] - 0.005), "c_dd_constraints": bool(m["max_dd_net"] >= m0["max_dd_net"] - 0.05 and cons),
                "d_combined_sharpe": bool(sh(comb(k)) >= sh(comb("A0")))}
        dec = "reference" if k == "A0" else ("PASS" if all(rule.values()) else "fail")
        rows.append({"arm": k, "test_ic": m["ic"], "test_ic_t": m["ic_t"], "test_ir_net": m["ir_net"], "test_sharpe": m["sharpe_net"], "test_max_dd": m["max_dd_net"],
                     "paired_net_t": pn["t"], "ir_diff_ci90": bd["ci90"], "combined_sharpe": sh(comb(k)), "dev_sharpe": sh(net[f"{k}_dev"]), "boot_ir_ci90": m["boot_ir_net"]["ci90"],
                     "rule": rule, "decision": dec})
        ledger.record(EXP, "second_look", k, "text-conditional factor weighting (PREREGISTRATION.md)", {"arm": k}, "dev", "dev", "test", 3, m, paired_t=pn["t"], decision=dec,
                      reason="second pre-registered TEST look (Bonferroni x2)")
        print(k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items()}, flush=True)
    # replications
    G = ctx.comp.extras["groups"]
    rep = {}
    ia, ib = E.ic_series(P, G["surprise"].to_numpy(), "test", fresh), E.ic_series(P, G["surprise"].to_numpy(), "test", ~fresh)
    d = (ia - ib).dropna(); rep["R1_fresh_minus_stale_surprise_ic"] = {"ic_fresh": float(ia.mean()), "ic_stale": float(ib.mean()), "diff": float(d.mean()), "t": ST.tstat(d)}
    ia, ib = E.ic_series(P, b1, "test", ~merger), E.ic_series(P, b1, "test", merger)
    d = (ia - ib).dropna(); rep["R2_nonmerger_minus_merger_b1_ic"] = {"ic_other": float(ia.mean()), "ic_merger": float(ib.mean()), "diff": float(d.mean()), "t": ST.tstat(d)}
    h = B["A0_test"]["holdings"]
    nq = pd.Series(np.where(P.u, np.nan_to_num(raw["txt_v2_novel_negative_max"], nan=0.0), np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True)
    hk = h.merge(P.df[["permno", "eom"]].assign(nq=nq.to_numpy()), on=["permno", "eom"], how="left")
    ex = (hk["weight"].abs() * hk["nq"]).groupby(hk["target_month"]).sum() / hk["weight"].abs().groupby(hk["target_month"]).sum()
    z = pd.concat([ex.rename("x"), net["A0_test"].abs().rename("v")], axis=1).dropna()
    rc = float(np.corrcoef(rankdata(z.x), rankdata(z.v))[0, 1])
    rep["R3_book_text_risk"] = {"rank_corr": rc, "t": rc * np.sqrt((len(z) - 2) / (1 - rc ** 2))}
    for k, v in rep.items():
        v["replicated"] = bool((v.get("diff", v.get("rank_corr")) > 0) and v["t"] >= 2.3)
        ledger.record(EXP, "second_look_replication", k, "conditional text structure", {}, "dev", "dev", "test", 3, {"ic": v.get("diff", v.get("rank_corr")), "ic_t": v["t"]},
                      decision="replicated" if v["replicated"] else "not replicated", reason="second pre-registered TEST look")
        print(k, v, flush=True)
    passing = [r for r in rows if r["decision"] == "PASS"]
    cand = max(passing, key=lambda r: r["combined_sharpe"])["arm"] if passing else "A0 (B1)"
    E.dump(OUT / "results.json", {"arms": rows, "replications": rep, "candidate": cand})
    for k, r in B.items():
        r["frame"].to_csv(OUT / f"frame_{k}.csv", index=False)
    print("CANDIDATE", cand, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
