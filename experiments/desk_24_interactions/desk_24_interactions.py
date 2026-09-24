"""
desk_24_interactions - is the factor signal CONDITIONALLY useful where it is weak on average?

Economic hypotheses, pre-listed (DEV, universe rows, B1 = composite + dtc). Conditional IC = IC of B1 within a subgroup; the test is the monthly paired
difference between two complementary subgroups (t, one-sided p in the predicted direction); BH-FDR q = 0.10 over I1-I6.
  I1  SI: B1 works better among HIGH-SI names (short-sale impediments concentrate mispricing: Stambaugh-Yu-Yuan, Drechsler)       high - low SI tercile > 0
  I2  text: B1 works better WITHOUT fresh adverse text (desk_23 C1 hint; news resets the characteristic's meaning)                no-top-novneg - top-novneg > 0
  I3  size: B1 works better among the smaller names of the universe (limits to arbitrage; McLean-Pontiff)                           small - large tercile > 0
  I4  liquidity: B1 works better among less liquid names (Amihud rank)                                                              illiquid - liquid tercile > 0
  I5  volatility: B1 works better among LOW idiosyncratic-vol names (signal-to-noise)                                               low - high ivol tercile > 0
  I6  analyst/attention proxy: B1 works better among names with FEW 8-K filings in the trailing 12 months (low attention)           low - high filing-intensity > 0
Two multiplicative signals (candidates for the book, pre-registered as a family of 2; rule as desk_19):
  M1  quality-value: value score only among above-median quality names (Piotroski / Asness QMJ: cheap AND good), else 0, added as an 8th group to the 7 frozen
  M2  profitability x value product (Novy-Marx 2013: gross profitability complements value), added as an 8th group
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, factors_desk as FD  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402

OUT = HERE / "output"
EXP = "desk_24_interactions"


def tercile(P, x):
    return pd.Series(np.where(P.u, x, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P, raw = ctx.P, ctx.raw
    b1 = core.b1_score(ctx)
    has = np.nan_to_num(raw["txt_n_filings"]) > 0
    nov = raw["txt_v2_novel_negative_max"]
    tr = lambda x: tercile(P, x)
    q_si, q_size, q_ami, q_iv = tr(ctx.S["sir"]), tr(P.df["raw_me"].to_numpy()), tr(P.df["ami_126d"].to_numpy()), tr(P.df["x_vol"].to_numpy())
    q_nov = tr(np.where(has, nov, np.nan))
    q_att = tr(raw["txt_filings_trailing12_mean"])
    tests = {
        "I1_SI_high_vs_low": (q_si > 2 / 3, q_si <= 1 / 3),
        "I2_noadverse_vs_adverse_text": (~(q_nov > 2 / 3), q_nov > 2 / 3),
        "I3_small_vs_large": (q_size <= 1 / 3, q_size > 2 / 3),
        "I4_illiquid_vs_liquid": (q_ami > 2 / 3, q_ami <= 1 / 3),  # ami_126d rank: higher = more illiquid (panel rank of Amihud)
        "I5_lowvol_vs_highvol": (q_iv <= 1 / 3, q_iv > 2 / 3),
        "I6_lowattention_vs_high": (q_att <= 1 / 3, q_att > 2 / 3),
    }
    rows = []
    for k, (a, b) in tests.items():
        ia, ib = E.ic_series(P, b1, "dev", a), E.ic_series(P, b1, "dev", b)
        d = (ia - ib).dropna()
        rows.append({"test": k, "ic_A": float(ia.mean()), "ic_B": float(ib.mean()), "diff": float(d.mean()), "t": ST.tstat(d), "n": len(d), "p": ST.p_from_t(ST.tstat(d))})
    D = pd.DataFrame(rows)
    D["bh_q10"] = ST.bh_fdr(D["p"].to_numpy(), 0.10)
    print(D.round(4).to_string(), flush=True)
    D.to_csv(OUT / "conditional_ic_dev.csv", index=False)
    for _, r in D.iterrows():
        ledger.record(EXP, "conditional_ic", r.test, "economically motivated conditional IC", {}, "dev", "dev", "dev", 6, {"ic": r["diff"], "ic_t": r.t},
                      decision="discovery" if r.bh_q10 else "no evidence")
    G = ctx.comp.extras["groups"]
    val, qual, prof = G["value"].to_numpy(), G["quality"].to_numpy(), G["profitability"].to_numpy()
    qq = tercile(P, qual)
    M1 = np.where(qq > 0.5, val, 0.0)
    M2 = FD.urank(val * prof + 0.0 * val, P)  # product of the two group scores, re-ranked
    V = {"V1_frozen7": ctx.comp.score, "M1_qualvalue": FD.add_groups(ctx.comp, {"qv": M1}).score, "M2_profxvalue": FD.add_groups(ctx.comp, {"pv": M2}).score}
    B = core.run_many(ctx, [(k, s, "dev", core.b1_kw()) for k, s in V.items()])
    ic0 = E.ic_series(P, V["V1_frozen7"], "dev")
    f0 = B["V1_frozen7"]["frame"].set_index("target_month")["net_port_excess_ret"]
    m0 = core.metrics(ctx, B["V1_frozen7"], V["V1_frozen7"], "dev", boot=False)
    out = []
    for k, s in V.items():
        m = core.metrics(ctx, B[k], s, "dev", boot=False)
        ic = E.ic_series(P, s, "dev")
        d = ic - ic0
        f = B[k]["frame"].set_index("target_month")["net_port_excess_ret"]
        pt = ST.paired(f, f0)["t"] if k != "V1_frozen7" else np.nan
        ok = k != "V1_frozen7" and ST.tstat(d) >= 1 and d[:"2017-12-31"].mean() > 0 and d["2018-01-31":].mean() > 0 and m["ir_net"] >= m0["ir_net"] + 0.10
        dec = "reference" if k == "V1_frozen7" else ("promising" if ok else ("rejected" if ST.tstat(d) <= -1 else "no evidence"))
        out.append({"variant": k, "ic": m["ic"], "paired_ic_gain": float(d.mean()), "paired_ic_t": ST.tstat(d), "ir_net": m["ir_net"], "max_dd": m["max_dd_net"], "paired_net_t": pt, "decision": dec})
        if k != "V1_frozen7":
            ledger.record(EXP, "interaction_signal", k, "multiplicative quality/profitability x value", {}, "dev", "dev", "dev", 2, m, paired_t=pt, decision=dec)
    print(pd.DataFrame(out).round(4).to_string())
    E.dump(OUT / "results.json", {"conditional": rows, "signals": out})
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
