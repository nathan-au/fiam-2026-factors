"""
desk_20_portfolio_construction - with the SIGNAL fixed, does a different construction extract more of it?

Signal fixed: V0 = B1 (frozen composite + dtc) and, as a robustness column, V3 = JKP13 z (desk_19's best book, not promoted). Research panel PanelX.
Motivation: desk_16 L11 (Grinold alpha = IC x vol x score: the LP ignores stock risk; 2020's loss came from high-ivol shorts), L12 (transfer coefficient),
L9 (short-leg concentration), L13 (hold bands).
PRE-REGISTERED family (6 alternatives vs C0 = frozen lc_t10 LP + 10% SI cap):
  C1  LP, per-name cap scaled by inverse idiosyncratic vol: cap_i = 1% x clip(median_t(s) / s_i, 0.5, 2), s = ivol_capm_252d
  C2  LP, per-name cap 0.5% (twice the names)
  C3  QP risk-aware: max alpha'w - lam (factor + idio risk), alpha = 0.02 s z, lam = 20 (a priori scale where the idio penalty of a 1% position ~ its
      alpha at |z| = 1), candidates = score quartiles with hold band
  C4  QP score-tracking (weights ~ z, projected on the constraints), same candidates
  C5  SI cap 5%       C6  SI cap 15%    (level sensitivity of the one confirmed component)
Rule: 'promising' iff DEV net IR >= C0 + 0.15 (5 x the desk_18 jitter sd) AND paired monthly net t >= 1 AND max DD not worse by > 5 pp AND all FIAM constraints
pass. Reported for both signals; promotion requires both signals to satisfy the rule (construction must not be signal-specific). DEV only.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, lp as L  # noqa: E402
from fiam_research import core, construct as K, ledger, stats as ST, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_20_portfolio_construction"
_G = {}


def job(item):
    import warnings
    warnings.filterwarnings("ignore")
    name, sig, arm = item
    ctx, S, fm, wm = _G["ctx"], _G["S"][sig], _G["fm"], _G["wm"]
    if arm == "C0":
        r = core.book(ctx, S, "dev", **core.b1_kw())
    elif arm == "C1":
        r = core.book(ctx, S, "dev", **core.b1_kw(extras={"w_mult": wm}))
    elif arm == "C2":
        r = core.book(ctx, S, "dev", **core.b1_kw(cfg={"max_weight": 0.005}))
    elif arm == "C3":
        r = K.book_qp(ctx, S, "dev", "qp_risk", 20.0, fm=fm)
    elif arm == "C4":
        r = K.book_qp(ctx, S, "dev", "qp_track", fm=fm)
    elif arm == "C5":
        r = core.book(ctx, S, "dev", **core.b1_kw(si_cap=0.05))
    elif arm == "C6":
        r = core.book(ctx, S, "dev", **core.b1_kw(si_cap=0.15))
    return name, r


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    iv = core.load_chars(ctx, ["ivol_capm_252d"])["ivol_capm_252d"]
    med = iv.where(P.u).groupby(P.df["eom"].to_numpy()).transform("median")
    wm = np.clip((med / iv).to_numpy(), 0.5, 2.0)
    wm = np.where(np.isfinite(wm), wm, 1.0)
    S = {"V0": core.b1_score(ctx), "V3": TH.composite_from_themes(TH.theme_scores(ctx, "z"))}
    _G.update(ctx=ctx, S=S, fm=K.style_fm_returns(ctx), wm=wm)
    arms = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    items = [(f"{sig}_{a}", sig, a) for sig in S for a in arms]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(7) as pool:
        R = dict(pool.map(job, items, chunksize=1))
    rows = []
    for sig in S:
        base = R[f"{sig}_C0"]
        f0 = base["frame"].set_index("target_month")["net_port_excess_ret"]
        for a in arms:
            r = R[f"{sig}_{a}"]
            m = core.metrics(ctx, r, S[sig], "dev", boot=False)
            f = r["frame"].set_index("target_month")["net_port_excess_ret"]
            pn = ST.paired(f, f0)
            cons = L.constraint_report(r)
            tc = K.transfer_coefficient(ctx, r, S[sig], "dev")
            row = {"signal": sig, "arm": a, "ir_net": m["ir_net"], "ir_gross": m["ir_gross"], "sharpe": m["sharpe_net"], "max_dd": m["max_dd_net"], "beta_vw": m["beta_vw_mkt"],
                   "turnover": m["one_way_turnover"], "pos": f"{m['min_positions']}-{m['max_positions']}", "cost_bp": m["trade_cost_bp_month"] + m["borrow_cost_bp_month"],
                   "long_ann": m["long_contrib_ann"], "short_ann": m["short_contrib_ann"], "tc": tc, "paired_net_t": pn["t"], "relaxed_months": m["months_turnover_relaxed"],
                   "constraints_ok": all(v for k, v in cons.items() if k not in ("dollar_neutral",)), **{f"expo_{k}": v for k, v in m["holdings_style_exposure"].items()}}
            rows.append(row)
    D = pd.DataFrame(rows)
    for sig in S:
        b = D[(D.signal == sig) & (D.arm == "C0")].iloc[0]
        for i in D.index[(D.signal == sig) & (D.arm != "C0")]:
            r = D.loc[i]
            D.loc[i, "rule_ok"] = bool(r.ir_net >= b.ir_net + 0.15 and r.paired_net_t >= 1 and r.max_dd >= b.max_dd - 0.05 and r.constraints_ok)
    for a in arms[1:]:
        ok = bool(D[(D.arm == a)]["rule_ok"].all())
        D.loc[D.arm == a, "decision"] = "promising" if ok else ("rejected" if (D[D.arm == a]["paired_net_t"] <= -1).all() else "no evidence")
    D.loc[D.arm == "C0", "decision"] = "reference"
    for _, r in D.iterrows():
        ledger.record(EXP, "construction", f"{r.signal}_{r.arm}", "construction with fixed signal", {"arm": r.arm, "signal": r.signal}, "dev", "dev", "dev", 6,
                      {"ir_net": r.ir_net, "ir_gross": r.ir_gross, "max_dd_net": r.max_dd}, paired_t=r.paired_net_t, decision=r.decision)
    D.to_csv(OUT / "construction_dev.csv", index=False)
    pd.set_option("display.width", 250)
    print(D[["signal", "arm", "ir_net", "ir_gross", "sharpe", "max_dd", "beta_vw", "turnover", "pos", "cost_bp", "long_ann", "short_ann", "tc", "paired_net_t", "relaxed_months", "constraints_ok", "decision"]].round(3).to_string())
    print(D[["signal", "arm"] + [c for c in D.columns if c.startswith("expo_")]].round(2).to_string())
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
