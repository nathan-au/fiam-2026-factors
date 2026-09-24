"""
desk_21_regime_timing - can the 2015-2020 degradation be mitigated with CAUSAL, a-priori regime rules?

Motivation: desk_17 (DEV loss = multi-year value / low-vol drawdowns; dispersion predicts lower IC), desk_16 L2/L3 (time-series factor momentum),
L6 (Stivers-Sun dispersion), L7 (vol-managed caveats), tpa (ERC suggestive).
Bases: B1 groups (7 frozen groups + dtc, PM = B1) and J13z (JKP 13 themes, z-scores = desk_19 V3). All parameters fixed a priori from the literature:
  R1a/R2a  factor momentum, L = 12 months, losers dropped (off = 0)             [Ehsani-Linnainmaa 12m sign]
  R1b/R2b  factor momentum, L = 12, losers at half weight (off = 0.5)          [shrinkage toward equal weight]
  R3a/R3b  inverse-vol group weights, 24m window                                [risk balancing]
  R4a/R4b  dispersion gross scaling: gross x 0.5 when formation dispersion > expanding 80th pct (>= 24 months history)  [Stivers-Sun; desk_17]
(a = B1 groups, b = J13z.) 8 variants, DEV only.
Rule 'promising': DEV net IR >= base + 0.15 AND paired monthly net t >= 1 AND mean net difference > 0 in both 2016-02..2017-12 and 2018-01..2020-12
AND max DD not worse by > 5 pp. Discovery caveat: the dispersion-IC relation was found on DEV and its IC slope was also seen on TEST (desk_17).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH, timing as TM  # noqa: E402

OUT = HERE / "output"
EXP = "desk_21_regime_timing"


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    G = ctx.comp.extras["groups"].copy()
    G["dtc"] = ctx.dtc_block(True)
    Tz = TH.theme_scores(ctx, "z")
    bases = {"a": G, "b": Tz}
    scores, info = {}, {}
    for k, T in bases.items():
        R = TM.theme_returns(ctx, T)
        R.to_csv(OUT / f"theme_returns_{k}.csv")
        scores[f"base_{k}"] = T.mean(axis=1).to_numpy()
        for lab, off in (("mom_drop", 0.0), ("mom_half", 0.5)):
            W = TM.momentum_weights(ctx, R, 12, off)
            scores[f"{lab}_{k}"] = TM.weighted_composite(ctx, T, W)
            info[f"{lab}_{k}"] = {"avg_weight": W[W.index <= "2020-12-31"].mean().round(2).to_dict()}
        W = TM.invvol_weights(ctx, R)
        scores[f"invvol_{k}"] = TM.weighted_composite(ctx, T, W)
        info[f"invvol_{k}"] = {"avg_weight": W[W.index <= "2020-12-31"].mean().round(2).to_dict()}
    books = core.run_many(ctx, [(k, s, "dev", core.b1_kw()) for k, s in scores.items()])
    scale = TM.dispersion_gross_scale(ctx)
    scale.to_csv(OUT / "dispersion_scale.csv")
    info["dispersion_scaled_months_dev"] = [str(d.date()) for d in scale[(scale < 1) & (scale.index <= "2020-12-31")].index]
    rows = []
    act = lambda s: s - core.RF_HALF
    for k in ("a", "b"):
        base = books[f"base_{k}"]
        f0 = base["frame"].set_index("target_month")["net_port_excess_ret"]
        mb = core.metrics(ctx, base, scores[f"base_{k}"], "dev", boot=False)
        rows.append({"variant": f"base_{k}", "ir_net": mb["ir_net"], "max_dd": mb["max_dd_net"], "ic": mb["ic"], "paired_t": np.nan, "d_h1": np.nan, "d_h2": np.nan, "decision": "reference"})
        variants = {f"{v}_{k}": books[f"{v}_{k}"]["frame"].set_index("target_month")["net_port_excess_ret"] for v in ("mom_drop", "mom_half", "invvol")}
        variants[f"disp_{k}"] = TM.apply_gross_scale(base, scale)
        for name, f in variants.items():
            d = (f - f0).dropna()
            ir = ST.ir(act(f))
            dd = float(((1 + f).cumprod() / (1 + f).cumprod().cummax() - 1).min())
            ic = float(E.ic_series(P, scores[name], "dev").mean()) if name in scores else mb["ic"]
            h1, h2 = d["2016-02-28":"2017-12-31"].mean(), d["2018-01-31":"2020-12-31"].mean()
            ok = ir >= mb["ir_net"] + 0.15 and ST.tstat(d) >= 1 and h1 > 0 and h2 > 0 and dd >= mb["max_dd_net"] - 0.05
            row = {"variant": name, "ir_net": ir, "max_dd": dd, "ic": ic, "paired_t": ST.tstat(d), "d_h1": float(h1), "d_h2": float(h2),
                   "decision": "promising" if ok else ("rejected" if ST.tstat(d) <= -1 else "no evidence")}
            rows.append(row)
            ledger.record(EXP, "regime_timing", name, "causal a-priori regime rule", {"variant": name}, "dev", "dev", "dev", 8, {"ir_net": ir, "max_dd_net": dd, "ic": ic},
                          paired_t=row["paired_t"], decision=row["decision"])
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "timing_dev.csv", index=False)
    print(D.round(4).to_string())
    for k, v in info.items():
        print(k, v)
    E.dump(OUT / "results.json", {"rows": rows, "info": info})
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
