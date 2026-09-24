"""
desk_14_si_history_extension - data extension + fresh out-of-sample test of the two short-interest components of the baseline.

fiam_desks/si.py starts at 2020-06 because only those FINRA files were cached. FINRA's CDN serves mid-month files from 2018-01 (earlier: HTTP 403),
downloaded 2026-09-24. Days-to-cover (dtc) and the 10% SI cap were discovered/adopted on 2020-06..2026 data (feat_short_interest, feat_si_followup),
so formation months 2018-01..2020-05 (target months 2018-02..2020-06, 29 months) are data NO earlier experiment has seen for these components.

PRE-REGISTERED (written before running):
  H1  dtc standalone universe IC < 0 direction holds (i.e. -rank(dtc) block IC > 0) on the 29 fresh months: 'replicates' iff block IC > 0 and t >= 1.
  H2  dtc as 8th group raises the composite's IC on the fresh months: paired gain t >= 1 -> 'supports', t <= -1 -> 'contradicts', else 'inconclusive'.
  H3  The SI cap does not hurt the composite+dtc book on the fresh months: paired monthly net return (cap - nocap) t > -1; and reduces the short leg's worst month.
  Report also the full-DEV books: B0 (frozen, dtc from 2020-06) vs B0x (extended, dtc from 2018-01).
Data used for discovery: none (components fixed); evaluation: DEV fresh months only. TEST not touched.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, factors_desk as FD  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402

OUT = HERE / "output"
EXP = "desk_14_si_history_extension"
FRESH = (pd.Timestamp("2018-02-28"), pd.Timestamp("2020-06-30"))


def main():
    t0 = time.time()
    ctx = core.Ctx(text=False)
    P = ctx.P
    res = {}
    # coverage of the extended file per year (universe rows with a dtc value)
    u = P.u
    cov = pd.Series(np.isfinite(ctx.S["dtc"][u])).groupby(pd.DatetimeIndex(P.df["eom"].to_numpy()[u]).year).mean()
    res["coverage_universe_by_eom_year"] = {int(k): round(float(v), 3) for k, v in cov.items()}
    # consistency: extended == frozen where frozen exists
    both = np.isfinite(ctx.S0["dtc"])
    res["extended_equals_frozen_on_overlap"] = bool(np.allclose(ctx.S0["dtc"][both], ctx.S["dtc"][both]) and np.allclose(ctx.S0["sir"][both], ctx.S["sir"][both], equal_nan=True))
    print("coverage", res["coverage_universe_by_eom_year"], "| overlap identical:", res["extended_equals_frozen_on_overlap"])

    fresh = (P.months >= np.datetime64(FRESH[0])) & (P.months <= np.datetime64(FRESH[1]))
    dtcb = ctx.dtc_block(extended=True)
    comp = ctx.comp.score
    comp8 = FD.add_groups(ctx.comp, {"dtc": dtcb}).score
    ic_b = E.ic_series(P, dtcb, "dev", fresh)
    ic_c = E.ic_series(P, comp, "dev", fresh)
    ic_k = E.ic_series(P, comp8, "dev", fresh)
    res["H1_dtc_block_ic_fresh"] = E.ic_summary(ic_b)
    res["H2_paired_gain_fresh"] = E.paired(ic_k, ic_c)
    res["H2_comp_ic_fresh"], res["H2_comp8_ic_fresh"] = E.ic_summary(ic_c), E.ic_summary(ic_k)
    ri, cor = E.resid_ic(P, np.where(fresh, dtcb, 0), comp, "dev")
    ri = ri[(ri.index >= FRESH[0]) & (ri.index <= FRESH[1])]
    res["H1_residual_ic_fresh"] = {"ic": float(ri.mean()), "t": E.tstat(ri)}
    # SI-ratio sort on fresh months: mean next-month return of top-SI decile minus rest (universe)
    sir = ctx.S["sir"]
    d = pd.DataFrame({"m": P.months, "sir": sir, "y": P.y})[P.u & fresh & np.isfinite(sir)]
    d["hi"] = d["sir"] > 0.10
    spread = d.groupby("m").apply(lambda g: g.loc[g["hi"], "y"].mean() - g.loc[~g["hi"], "y"].mean(), include_groups=False)
    res["sir_gt10_minus_rest_fresh"] = {"mean": float(spread.mean()), "t": ST.tstat(spread), "share_names_hi": float(d["hi"].mean())}
    print("H1 dtc block IC fresh", res["H1_dtc_block_ic_fresh"], "resid", res["H1_residual_ic_fresh"])
    print("H2 paired gain fresh", res["H2_paired_gain_fresh"], "comp", res["H2_comp_ic_fresh"]["ic"], "-> comp8", res["H2_comp8_ic_fresh"]["ic"])
    print("SIR>10% minus rest", res["sir_gt10_minus_rest_fresh"])

    books = {"B0_frozen": (ctx.baseline_score(False), True, False), "B0x_extended": (comp8, True, True), "B0x_nocap": (comp8, False, True),
             "comp_only_nocap": (comp, False, True), "comp_only_cap_ext": (comp, True, True)}
    frames = {}
    for name, (sc, cap, ext) in books.items():
        r = core.book(ctx, sc, "dev", si_cap=cap, extended_si=ext)
        m = core.metrics(ctx, r, sc, "dev", boot=False)
        fr = r["frame"].set_index("target_month")
        frames[name] = fr
        fm = fr.loc[FRESH[0]:FRESH[1]]
        m["fresh_ir_net"] = ST.ir(fm["net_port_excess_ret"] - core.RF_HALF)
        m["fresh_short_leg_worst"] = float(fm["short_leg_ret"].min())
        m["fresh_cum_net"] = float((1 + fm["net_port_excess_ret"]).prod() - 1)
        res[name] = m
        print(f"{name:18s} DEV {core.short(m)} | fresh IR {m['fresh_ir_net']:+.3f} cum {100 * m['fresh_cum_net']:+.1f}% worst short-leg {100 * m['fresh_short_leg_worst']:.1f}%", flush=True)
        ledger.record(EXP, "si_extension", name, "dtc/SI cap on unseen 2018-2020 months", {"cap": cap, "extended": ext}, "none (fixed)", "none", "dev", len(books), m)
    f = lambda a: frames[a].loc[FRESH[0]:FRESH[1], "net_port_excess_ret"]
    res["H3_cap_vs_nocap_fresh"] = ST.paired(f("B0x_extended"), f("B0x_nocap"))
    res["B0x_vs_comp_only_nocap_fresh"] = ST.paired(f("B0x_nocap"), f("comp_only_nocap"))
    res["B0x_vs_B0_fullDEV"] = ST.paired(frames["B0x_extended"]["net_port_excess_ret"], frames["B0_frozen"]["net_port_excess_ret"])
    print("H3 cap-nocap fresh", res["H3_cap_vs_nocap_fresh"], "| dtc book vs comp (nocap) fresh", res["B0x_vs_comp_only_nocap_fresh"])
    h1 = res["H1_dtc_block_ic_fresh"]
    res["verdict"] = {"H1": "replicates" if (h1["ic"] > 0 and h1["t"] >= 1) else "not replicated",
                      "H2": "supports" if res["H2_paired_gain_fresh"]["t"] >= 1 else ("contradicts" if res["H2_paired_gain_fresh"]["t"] <= -1 else "inconclusive"),
                      "H3": "holds" if res["H3_cap_vs_nocap_fresh"]["t"] > -1 else "fails"}
    print("VERDICT", res["verdict"], f"{time.time() - t0:.0f}s")
    E.dump(OUT / "results.json", res)


if __name__ == "__main__":
    main()
