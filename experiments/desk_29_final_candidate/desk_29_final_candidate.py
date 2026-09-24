"""
desk_29_final_candidate - FIAM-format outputs and audits for
  B1  the properly validated candidate (desk_27 rule 3: no arm passed) = committed system + audit corrections (delisting rows kept, SI from 2018)
  A5  the documented alternative ('promising - requires further validation'): JKP 13 themes, z-scores, equal weight, literature signs, same PM
Both are arms already evaluated in desk_27; this adds no TEST arm (ledgered as output generation).
Audits: FIAM constraints month by month, determinism (two runs), truncation invariance of the new signal code (themes / z-scores) at 3 dates.
"""
import sys, time, types
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import audit as A, evaluate as E, lp as L, system as S  # noqa: E402
from fiam_research import core, ledger, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_29_final_candidate"


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    sig = {"B1": core.b1_score(ctx), "A5_jkp13z": TH.theme_scores(ctx, "z").mean(axis=1).to_numpy()}
    res = {}
    for name, s in sig.items():
        r = core.book(ctx, s, "test", **core.b1_kw())
        r2 = core.book(ctx, s, "test", **core.b1_kw())
        hh = [A.frame_hash(x["holdings"][["target_month", "permno", "weight"]]) for x in (r, r2)]
        out = {"result": r}
        h = S.holdings_fiam(out)
        ret = S.returns_csv(out)
        h.to_csv(OUT / f"holdings_{name}.csv", index=False)
        ret.to_csv(OUT / f"returns_{name}.csv", index=False)
        m = core.metrics(ctx, r, s, "test")
        fr = r["frame"]
        res[name] = {"metrics": {k: v for k, v in m.items()}, "constraints": L.constraint_report(r), "determinism": hh[0] == hh[1],
                     "monthly_extremes": {"positions": [int(fr.n_positions.min()), int(fr.n_positions.max())], "gross": [float(fr.gross_exposure.min()), float(fr.gross_exposure.max())],
                                          "net_abs_max": float(fr.net_exposure.abs().max()), "beta_abs_max": float(fr.beta_exposure.abs().max())},
                     "top10_long_avg": h[h.WEIGHT > 0].groupby(["TICKER", "COMPANY NAME"])["WEIGHT"].sum().div(h.Date.nunique()).nlargest(10).round(3).to_dict(),
                     "top10_short_avg": h[h.WEIGHT < 0].groupby(["TICKER", "COMPANY NAME"])["WEIGHT"].sum().div(h.Date.nunique()).nsmallest(10).round(3).to_dict()}
        print(name, core.short(m), "| constraints", all(res[name]["constraints"].values()), "| deterministic", res[name]["determinism"], flush=True)
        ledger.record(EXP, "final_outputs", name, "FIAM-format outputs (arm already in desk_27)", {}, "dev", "dev", "test", 1, m, decision="output",
                      reason="output generation of a desk_27 arm; no selection")
    # truncation invariance of theme z-scores (new code): scores at date t from a panel cut at t must equal full-panel scores
    Tfull = TH.theme_scores(ctx, "z")
    rng = np.random.default_rng(3)
    eoms = np.sort(P.df["eom"].unique())
    dates = sorted(rng.choice(eoms[24:], 3, replace=False))
    diffs = []
    for t in dates:
        Pt = P.truncated(t)
        c2 = types.SimpleNamespace(P=Pt)
        J = TH.jkp_map()
        X = core.load_chars(c2, list(J["col"]))
        Z = np.column_stack([r_["direction"] * TH._zscore(X[r_["col"]].to_numpy(float), c2) for _, r_ in J.iterrows()])
        Zf = pd.DataFrame(Z, columns=J["col"])
        Tt = pd.DataFrame({th: Zf[list(g["col"])].mean(axis=1) for th, g in J.groupby("cluster")})
        rows = (Pt.df["eom"] == t).to_numpy()
        full_rows = (P.df["eom"] == t).to_numpy()
        diffs.append(float(np.abs(Tt.loc[rows, Tfull.columns].to_numpy() - Tfull.loc[full_rows].to_numpy()).max()))
    res["truncation_invariance_themes"] = {"dates": [str(pd.Timestamp(d).date()) for d in dates], "max_abs_diff": diffs, "passed": bool(max(diffs) < 1e-10)}
    print("truncation", res["truncation_invariance_themes"])
    E.dump(OUT / "summary.json", res)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
