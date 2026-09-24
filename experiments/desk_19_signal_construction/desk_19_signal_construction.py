"""
desk_19_signal_construction - can the factors-desk SIGNAL be built better, without fitting anything?

Literature motivation: desk_16 L1 (JKP 13 themes with published directions), L5 (value beyond book equity), L10 (residual momentum), L14 (expect decay).
Reference V0 = B1 signal (frozen 7 groups + dtc, extended SI). PM fixed = B1 (lc_t10 + 10% SI cap). Research panel = PanelX (desk_15).

PRE-REGISTERED candidate family (5, written before running):
  V1  frozen 7 groups, no dtc                       (is dtc helping on DEV at all?)
  V2  JKP 13 themes, equal weight, signed universe ranks, theme = mean of members
  V3  as V2 with winsorised z-scores instead of ranks
  V4  V2 + dtc as a 14th group
  V5  all 146 signed ranks equal-weighted (no theme balancing: 'zoo mean')
Promotion rule ('promising - requires further validation'): paired DEV IC gain vs V0 t >= 1 AND gain > 0 in both DEV halves (2015-02..2017-12,
2018-01..2020-12) AND DEV book net IR >= V0 + 0.10 AND max DD not worse by more than 5 pp. Otherwise 'no evidence' (or 'rejected' if paired t <= -1).
Diagnostics (NOT used for selection): per-theme IC by DEV year; leave-one-theme-out IC; style-residual IC (score residualised each month on 7 style
ranks); holdings style exposures. DEV only.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, factors_desk as FD  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_19_signal_construction"
H1 = ("2015-02-28", "2017-12-31")
H2 = ("2018-01-31", "2020-12-31")


def style_resid_ic(ctx, score, period="dev"):
    P = ctx.P
    m = P.mask(period)
    X = np.column_stack([P.df[c].to_numpy()[m] for c in ("x_size", "be_me", "x_mom", "x_vol", "qmj", "gp_at", "at_gr1")])
    X = np.where(np.isfinite(X), X, 0.0)
    s, y, mo = np.asarray(score)[m], P.y[m], P.months[m]
    out = {}
    from scipy.stats import rankdata
    for t in np.unique(mo):
        i = mo == t
        Z = np.column_stack([np.ones(i.sum()), X[i]])
        b = np.linalg.lstsq(Z, s[i], rcond=None)[0]
        r = s[i] - Z @ b
        out[t] = np.corrcoef(rankdata(r), rankdata(y[i]))[0, 1]
    return pd.Series(out)


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    Tr = TH.theme_scores(ctx, "rank")
    Tz = TH.theme_scores(ctx, "z")
    D = TH.signed_scores(ctx, "rank")
    dtc = ctx.dtc_block(True)
    V = {"V0_B1": core.b1_score(ctx), "V1_frozen7": ctx.comp.score, "V2_jkp13_rank": TH.composite_from_themes(Tr),
         "V3_jkp13_z": TH.composite_from_themes(Tz), "V4_jkp13_dtc": TH.composite_from_themes(Tr, extra={"dtc": dtc}), "V5_zoo146": D.mean(axis=1).to_numpy()}
    res = {"themes": {k: int(v) for k, v in TH.jkp_map().groupby("cluster").size().items()}}
    # diagnostics: theme IC by year on DEV
    tic = {th: E.ic_series(P, Tr[th].to_numpy(), "dev") for th in Tr.columns}
    TIC = pd.DataFrame(tic)
    TICY = TIC.groupby(TIC.index.year).mean().round(4).T
    TICY["dev_mean"], TICY["dev_t"] = TIC.mean().round(4), TIC.apply(ST.tstat).round(2)
    TICY.to_csv(OUT / "theme_ic_by_year_dev.csv")
    print(TICY.to_string(), flush=True)
    # leave-one-theme-out (diagnostic)
    base_ic = E.ic_series(P, V["V2_jkp13_rank"], "dev")
    loo = {}
    for th in Tr.columns:
        s = TH.composite_from_themes(Tr.drop(columns=[th]))
        ic = E.ic_series(P, s, "dev")
        loo[th] = {"ic_without": float(ic.mean()), "delta_vs_all13": float((ic - base_ic).mean()), "t": ST.tstat(ic - base_ic)}
    res["leave_one_theme_out_dev"] = loo
    pd.DataFrame(loo).T.round(4).to_csv(OUT / "leave_one_theme_out_dev.csv")
    # books (parallel)
    jobs = [(k, v, "dev", core.b1_kw()) for k, v in V.items()]
    books = core.run_many(ctx, jobs)
    ic0 = E.ic_series(P, V["V0_B1"], "dev")
    f0 = books["V0_B1"]["frame"].set_index("target_month")["net_port_excess_ret"]
    rows = []
    for k, s in V.items():
        m = core.metrics(ctx, books[k], s, "dev", boot=False)
        ic = E.ic_series(P, s, "dev")
        d = ic - ic0
        h1, h2 = d[H1[0]:H1[1]], d[H2[0]:H2[1]]
        sr = style_resid_ic(ctx, s)
        f = books[k]["frame"].set_index("target_month")["net_port_excess_ret"]
        pn = ST.paired(f, f0)
        row = {"variant": k, "ic": m["ic"], "ic_t": m["ic_t"], "ic_h1": float(ic[H1[0]:H1[1]].mean()), "ic_h2": float(ic[H2[0]:H2[1]].mean()),
               "paired_ic_gain": float(d.mean()), "paired_ic_t": ST.tstat(d), "gain_h1": float(h1.mean()), "gain_h2": float(h2.mean()),
               "style_resid_ic": float(sr.mean()), "style_resid_ic_t": ST.tstat(sr), "ir_net": m["ir_net"], "ir_gross": m["ir_gross"], "max_dd": m["max_dd_net"],
               "paired_net_t": pn["t"], "turnover": m["one_way_turnover"], "beta_vw": m["beta_vw_mkt"], **{f"expo_{a}": b for a, b in m["holdings_style_exposure"].items()},
               **{f"y{a}": b for a, b in m["calendar_year_net"].items()}}
        if k != "V0_B1":
            ok = (row["paired_ic_t"] >= 1 and row["gain_h1"] > 0 and row["gain_h2"] > 0 and row["ir_net"] >= rows[0]["ir_net"] + 0.10 and row["max_dd"] >= rows[0]["max_dd"] - 0.05)
            row["decision"] = "promising" if ok else ("rejected" if row["paired_ic_t"] <= -1 else "no evidence")
        else:
            row["decision"] = "reference"
        rows.append(row)
        print(f"{k:15s} {core.short(m)} | IC h1 {row['ic_h1']:+.4f} h2 {row['ic_h2']:+.4f} | gain {row['paired_ic_gain']:+.4f} (t {row['paired_ic_t']:+.2f}) | "
              f"style-resid IC {row['style_resid_ic']:+.4f} (t {row['style_resid_ic_t']:+.2f}) | paired net t {pn['t']:+.2f} -> {row['decision']}", flush=True)
        ledger.record(EXP, "signal_construction", k, "JKP literature-signed themes vs frozen composite", {"variant": k}, "dev", "dev", "dev", len(V) - 1, m,
                      paired_t=pn["t"], decision=row["decision"])
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "variants_dev.csv", index=False)
    res["variants"] = rows
    E.dump(OUT / "results.json", res)
    print(pd.DataFrame(loo).T.round(4).to_string(), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
