"""
desk_22_signal_decay - how long does each signal stay useful, and does a slower / faster rebalance help after costs?

The panel is MONTHLY (no daily returns): next-day / 5-day / 10-day / 20-day horizons cannot be measured; the finest horizon is 1 month (= the target).
  D1  universe rank IC of each signal formed at eom t against the excess return of month t+h, h = 1..12 (and the cumulative 3-month return t+1..t+3),
      for the 7 frozen groups, dtc, B1, the 13 JKP themes (z) and the text signal novneg_max. Universe and signal fixed at t (no future information:
      only the EVALUATION return moves forward; a stock that delists before t+h drops out of that horizon's IC, disclosed).
  D2  B1 book with one-way turnover budget 5% / 10% (base) / 20% / 40% per month (slower vs faster rebalancing), DEV.
Pre-registered reading: a signal is 'slow' if IC(h=6) >= 0.5 IC(h=1) with the same sign. D2 rule: an alternative budget is 'promising' only if DEV net IR >= base + 0.15
and paired t >= 1. DEV for D2; D1 is reported on DEV (decision data) and not on TEST.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, text_desk as TD  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_22_signal_decay"


def lead_returns(P, H=12):
    df = P.df[["permno", "eom", "y"]]
    W = df.pivot(index="eom", columns="permno", values="y").sort_index()
    idx = pd.date_range(W.index.min(), W.index.max(), freq="ME")
    W = W.reindex(idx)
    r = W.index.get_indexer(df["eom"])
    c = W.columns.get_indexer(df["permno"])
    A = W.to_numpy()
    out = {}
    for h in range(1, H + 1):
        L = np.full(len(df), np.nan)
        ok = r + h - 1 < len(A)
        L[ok] = A[r[ok] + h - 1, c[ok]]
        out[h] = L
    out["cum3"] = (1 + np.nan_to_num(out[1])) * (1 + out[2]) * (1 + out[3]) - 1
    return out


def ic_h(P, score, yh, period="dev"):
    m = P.mask(period) & np.isfinite(yh)
    d = pd.DataFrame({"e": P.df["eom"].to_numpy()[m], "s": np.asarray(score)[m], "y": yh[m]})
    v = d.groupby("e").apply(lambda g: np.corrcoef(rankdata(g["s"]), rankdata(g["y"]))[0, 1] if len(g) > 20 else np.nan, include_groups=False)
    return float(v.mean()), ST.tstat(v)


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P = ctx.P
    Y = lead_returns(P)
    G = ctx.comp.extras["groups"].copy()
    G["dtc"] = ctx.dtc_block(True)
    Tz = TH.theme_scores(ctx, "z")
    sig = {f"grp:{c}": G[c].to_numpy() for c in G.columns}
    sig["B1"] = core.b1_score(ctx)
    sig.update({f"jkp:{c}": Tz[c].to_numpy() for c in Tz.columns})
    sig["J13z"] = Tz.mean(axis=1).to_numpy()
    sig["text:novneg_max"] = TD.signal_score(P, ctx.raw, "novneg_max")
    rows = []
    for k, s in sig.items():
        row = {"signal": k}
        for h in list(range(1, 13)) + ["cum3"]:
            row[f"ic_h{h}"], row[f"t_h{h}"] = ic_h(P, s, Y[h])
        i1, i6 = row["ic_h1"], row["ic_h6"]
        row["slow"] = bool(np.sign(i6) == np.sign(i1) and abs(i6) >= 0.5 * abs(i1))
        rows.append(row)
        print(f"{k:28s} " + " ".join(f"{row[f'ic_h{h}']:+.4f}" for h in (1, 2, 3, 6, 9, 12)) + f" | cum3 {row['ic_hcum3']:+.4f} (t {row['t_hcum3']:+.2f}) slow={row['slow']}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "ic_by_horizon_dev.csv", index=False)
    # D2 turnover budget
    s = sig["B1"]
    jobs = [(f"to{int(100 * x)}", s, "dev", core.b1_kw(cfg={"turnover": x})) for x in (0.05, 0.10, 0.20, 0.40)]
    B = core.run_many(ctx, jobs)
    f0 = B["to10"]["frame"].set_index("target_month")["net_port_excess_ret"]
    rr = []
    for k, r in B.items():
        m = core.metrics(ctx, r, s, "dev", boot=False)
        f = r["frame"].set_index("target_month")["net_port_excess_ret"]
        pt = ST.paired(f, f0)["t"] if k != "to10" else np.nan
        dec = "reference" if k == "to10" else ("promising" if (m["ir_net"] >= core.metrics(ctx, B["to10"], s, "dev", boot=False)["ir_net"] + 0.15 and pt >= 1) else "no evidence")
        rr.append({"budget": k, "ir_net": m["ir_net"], "ir_gross": m["ir_gross"], "turnover": m["one_way_turnover"], "cost_bp": m["trade_cost_bp_month"] + m["borrow_cost_bp_month"],
                   "max_dd": m["max_dd_net"], "paired_t": pt, "relaxed": m["months_turnover_relaxed"], "decision": dec})
        ledger.record(EXP, "turnover_budget", k, "rebalance speed", {"turnover": k}, "dev", "dev", "dev", 3, m, paired_t=pt, decision=dec)
    R = pd.DataFrame(rr)
    R.to_csv(OUT / "turnover_budget_dev.csv", index=False)
    print(R.round(4).to_string())
    ledger.record(EXP, "decay", "ic_by_horizon", "signal decay 1-12m", {}, "dev", "n/a", "dev", 1, {}, decision="diagnostic")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
