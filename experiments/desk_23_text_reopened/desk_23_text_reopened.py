"""
desk_23_text_reopened - does 8-K text have conditional, portfolio-construction or risk-management value on the tradeable universe?

Motivation: desk_16 L15 (Lazy Prices: CHANGES matter, exec/litigation sections most informative, slow accrual), L16 (LM tone ~ risk), L9 (short-side
concentration, informed shorts); desk_09/12 (novneg_max small same-sign in both periods). DEV only (target months 2015-02..2020-12), universe rows.
PRE-REGISTERED family (signals; sign fixed a priori: adverse text -> lower return):
  T1  novneg_max                          two-sided universe rank (reference; desk_12)
  T2  novneg_max one-sided                only the adverse half penalised (asymmetry)
  T3  tone_change = neg_mean minus the firm's own mean neg_mean over its filing months in the previous 12 months (within-company change, Lazy Prices)
  T4  novneg_exec_lit = novneg_max only where the month has a 5.02 or 8.01 filing or litigation language (filing-type conditioning), else no view
  T5  novneg_max restricted to the top SI-ratio tercile (text x informed shorts), else no view
IC t-stats -> one-sided p -> BH-FDR q = 0.10 over the 5. Conditional analyses (not in the FDR family, descriptive): (C1) B1 IC among names with a filing
and top-tercile novneg vs the rest (text as confidence); (C2) novneg IC by universe size tercile; (C3) text as RISK: rank correlation of novneg_max / neg_mean
/ novelty with next-month |residual return| (return minus month mean) vs the same for no-filing names; (C4) if C3 holds (t >= 3): B1 book with the per-name
cap halved for top-decile novneg names (risk overlay) - rule 'promising' iff DEV net IR >= base + 0.15 and paired t >= 1.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, factors_desk as FD, text_desk as TD  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402

OUT = HERE / "output"
EXP = "desk_23_text_reopened"


def firm_trailing_mean(P, x, months=12):
    """Mean of x over the same permno's previous `months` calendar months where x is defined (strictly before t)."""
    df = pd.DataFrame({"permno": P.df["permno"].to_numpy(), "eom": P.df["eom"].to_numpy(), "x": x})
    W = df.pivot(index="eom", columns="permno", values="x")
    W = W.reindex(pd.date_range(W.index.min(), W.index.max(), freq="ME"))
    S = W.shift(1).rolling(months, min_periods=1).mean()
    r, c = S.index.get_indexer(df["eom"]), S.columns.get_indexer(df["permno"])
    return S.to_numpy()[r, c]


def ic_on(P, s, mask=None):
    ic = E.ic_series(P, s, "dev", mask)
    return {"ic": float(ic.mean()), "t": ST.tstat(ic), "n_months": int(ic.notna().sum())}


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P, raw = ctx.P, ctx.raw
    u = P.u
    nov = raw["txt_v2_novel_negative_max"]
    has = np.nan_to_num(raw["txt_n_filings"]) > 0
    res = {"coverage_universe_dev": float((has & P.mask("dev")).sum() / P.mask("dev").sum())}
    T1 = TD.signal_score(P, raw, "novneg_max")
    T2 = TD.signal_score(P, raw, "novneg_max", "one_sided")
    neg = raw["txt_v2_neg_mean"]
    tc = neg - firm_trailing_mean(P, neg)
    T3 = -FD.urank(tc, P)
    execlit = (np.nan_to_num(raw["txt_item_5_02"]) + np.nan_to_num(raw["txt_item_8_01"]) + np.nan_to_num(raw["txt_v2_n_801_litigation"])) > 0
    T4 = -FD.urank(np.where(execlit, nov, np.nan), P)
    sir = ctx.S["sir"]
    sq = pd.Series(np.where(u, sir, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
    T5 = -FD.urank(np.where(sq > 2 / 3, nov, np.nan), P)
    fam = {"T1_novneg": T1, "T2_novneg_onesided": T2, "T3_tone_change": T3, "T4_novneg_exec_lit": T4, "T5_novneg_highSI": T5}
    rows = []
    for k, s in fam.items():
        nz = (np.asarray(s) != 0)
        r = ic_on(P, s)
        r_nz = ic_on(P, s, nz)  # IC among names where the signal has a view
        rows.append({"signal": k, **r, "ic_where_view": r_nz["ic"], "t_where_view": r_nz["t"], "share_with_view": float((nz & P.mask("dev")).sum() / P.mask("dev").sum()),
                     "p_one_sided": ST.p_from_t(r["t"])})
    D = pd.DataFrame(rows)
    D["bh_q10_discovery"] = ST.bh_fdr(D["p_one_sided"].to_numpy(), 0.10)
    print(D.round(4).to_string(), flush=True)
    for _, r in D.iterrows():
        ledger.record(EXP, "text_family", r.signal, "text conditional/asymmetric/change", {}, "dev", "dev", "dev", 5, {"ic": r.ic, "ic_t": r.t},
                      decision="discovery" if r.bh_q10_discovery else "no evidence")
    res["family"] = rows
    # C1 text as confidence for the factor score
    b1 = core.b1_score(ctx)
    nq = pd.Series(np.where(u & has, nov, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
    hi = nq > 2 / 3
    res["C1_b1_ic_by_text"] = {"top_tercile_novneg": ic_on(P, b1, hi), "filing_not_top": ic_on(P, b1, has & ~hi), "no_filing": ic_on(P, b1, ~has)}
    print("C1", res["C1_b1_ic_by_text"])
    # C2 by size tercile
    res["C2_novneg_ic_by_size"] = {lab: ic_on(P, T1, P.terc == i) for i, lab in enumerate(("small", "mid", "large"))}
    print("C2", res["C2_novneg_ic_by_size"])
    # C3 text as risk predictor
    m = P.mask("dev")
    y = P.y
    dev_ = pd.Series(y).groupby(P.df["eom"].to_numpy()).transform("mean").to_numpy()
    absres = np.abs(y - dev_)
    c3 = {}
    for nm, x in (("novneg_max", nov), ("neg_mean", neg), ("novelty_max", raw["txt_v2_novelty_max"]), ("n_filings", raw["txt_n_filings"])):
        ok = m & np.isfinite(x) & has
        d = pd.DataFrame({"e": P.df["eom"].to_numpy()[ok], "x": x[ok], "v": absres[ok]})
        v = d.groupby("e").apply(lambda g: np.corrcoef(rankdata(g["x"]), rankdata(g["v"]))[0, 1] if len(g) > 20 else np.nan, include_groups=False)
        # control: does it survive the stock's own trailing vol? partial on ivol rank
        c3[nm] = {"rank_corr_with_abs_resid": float(v.mean()), "t": ST.tstat(v)}
    okf = m
    d = pd.DataFrame({"e": P.df["eom"].to_numpy()[okf], "f": has[okf].astype(float), "v": absres[okf], "iv": P.df["x_vol"].to_numpy()[okf]})
    c3["has_filing_vs_abs_resid_partial_ivol"] = {}
    vv = []
    for e, g in d.groupby("e"):
        Z = np.column_stack([np.ones(len(g)), rankdata(g["iv"]), g["f"]])
        b = np.linalg.lstsq(Z, rankdata(g["v"]), rcond=None)[0]
        vv.append(b[2])
    c3["has_filing_vs_abs_resid_partial_ivol"] = {"coef_rank_units": float(np.mean(vv)), "t": ST.tstat(pd.Series(vv))}
    # novneg partial on ivol among filers
    ok = m & has & np.isfinite(nov)
    d = pd.DataFrame({"e": P.df["eom"].to_numpy()[ok], "x": nov[ok], "v": absres[ok], "iv": P.df["x_vol"].to_numpy()[ok]})
    vv = []
    for e, g in d.groupby("e"):
        Z = np.column_stack([np.ones(len(g)), rankdata(g["iv"]) / len(g), rankdata(g["x"]) / len(g)])
        vv.append(np.linalg.lstsq(Z, rankdata(g["v"]) / len(g), rcond=None)[0][2])
    c3["novneg_partial_ivol"] = {"coef": float(np.mean(vv)), "t": ST.tstat(pd.Series(vv))}
    res["C3_text_as_risk"] = c3
    print("C3", c3)
    # C4 risk overlay if C3 holds
    if c3["novneg_partial_ivol"]["t"] >= 3:
        q = pd.Series(np.where(u & has, nov, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy()
        wm = np.where(q > 0.9, 0.5, 1.0)
        B = core.run_many(ctx, [("base", b1, "dev", core.b1_kw()), ("overlay", b1, "dev", core.b1_kw(extras={"w_mult": wm}))])
        mb, mo = core.metrics(ctx, B["base"], b1, "dev", boot=False), core.metrics(ctx, B["overlay"], b1, "dev", boot=False)
        pt = ST.paired(B["overlay"]["frame"].set_index("target_month")["net_port_excess_ret"], B["base"]["frame"].set_index("target_month")["net_port_excess_ret"])
        dec = "promising" if (mo["ir_net"] >= mb["ir_net"] + 0.15 and pt["t"] >= 1) else "no evidence"
        res["C4_risk_overlay"] = {"base_ir": mb["ir_net"], "overlay_ir": mo["ir_net"], "base_dd": mb["max_dd_net"], "overlay_dd": mo["max_dd_net"], "paired": pt, "decision": dec}
        ledger.record(EXP, "text_risk_overlay", "C4_halfcap_top_decile_novneg", "text as risk-management", {}, "dev", "dev", "dev", 1, mo, paired_t=pt["t"], decision=dec)
        print("C4", res["C4_risk_overlay"])
    E.dump(OUT / "results.json", res)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
