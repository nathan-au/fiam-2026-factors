"""
FIAM 2026 - Factors desk candidates, DEV period only (desk_05_factors_desk_dev). Test window not read.

The frozen composite has IC 0.0106 (t 0.68) on 2015-2020 against 0.0366 on 2021-2026: the desk that anchors the system is weak in the development period. desk_03 found
by accident that plain short-term reversal adds to it on dev. Here the factors-desk candidates are declared and tested properly (3 tests, Bonferroni 3):
  rev_group     an 8th economic group 'reversal' = mean(-rank(ret_1_0), -rank(ret_60_12)), both signs from docs/FACTORS.md sec 5 (fixed before results)
  rev_1m        ret_1_0 only  (decomposition of the above)
  sector_neutral  the frozen 7-group composite ranked WITHIN GICS sector instead of over the universe (replacement composite)
Also descriptive: dev IC of each of the 7 groups (never used for selection: experiments/factor_filter showed past IC does not persist).
Adoption rule (pre-registered): a candidate enters the default factors desk only if the repo's rule says PASS on dev (paired t >= 2, Bonferroni p <= 0.05, not small-only);
'inconclusive' candidates are carried as OPTIONS to one logged confirmation run on the test window; 'kill' are dropped.
LP books (lc_t10 on dev months, fresh start) are reported for the composite and each candidate with the paired net-return t vs the composite.
Run:  .venv/bin/python experiments/desk_05_factors_desk_dev/desk_05_factors_desk_dev.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, lp as L  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
PERIOD = "dev"


def main():
    t0 = time.time()
    P = Panel()
    comp = FD.composite(P)
    G = comp.extras["groups"]
    print("descriptive dev IC by factor group:", flush=True)
    grp_ic = {g: E.ic_summary(E.ic_series(P, G[g].to_numpy(), PERIOD)) for g in G.columns}
    for g, v in grp_ic.items():
        print(f"  {g:32s} IC {v['ic']:+.4f} (t {v['t']:+.2f})", flush=True)
    print(f"  composite {E.ic_summary(E.ic_series(P, comp.score, PERIOD))}", flush=True)

    r1, r60 = FD.urank(P.df["raw_ret_1_0"].to_numpy(), P), FD.urank(P.df["raw_ret_60_12"].to_numpy(), P)
    blocks = {"rev_group": -(r1 + r60) / 2, "rev_1m": -r1}
    rows, full, scores = [], {}, {"comp": comp.score}
    for name, b in blocks.items():
        rep = E.block_report(P, comp, b, PERIOD, name, n_perm=200, seed=3)
        full[name] = rep
        scores[name] = (G.sum(axis=1).to_numpy() + b) / 8
        rows.append({"candidate": name, "kind": "8th group", "ic_alone": rep["block"]["ic"], "t_alone": rep["block"]["t"], "comp_plus_ic": rep["comp_plus_block"]["ic"],
                     "paired_gain": rep["paired_gain"]["mean_diff"], "paired_t": rep["paired_gain"]["t"], "resid_ic": rep["residual_ic"], "perm_p": rep["perm"]["p_value"],
                     "bonf_p": min(1, 3 * rep["perm"]["p_value"]), "gain_small": rep["tercile_small"]["paired_gain"], "gain_mid": rep["tercile_mid"]["paired_gain"],
                     "gain_large": rep["tercile_large"]["paired_gain"], "verdict": E.kill_pass(rep, n_tests=3)})
    sn = FD.composite(P, "sector_neutral")
    scores["sector_neutral"] = sn.score
    ic_sn, ic_c = E.ic_series(P, sn.score, PERIOD), E.ic_series(P, comp.score, PERIOD)
    pr = E.paired(ic_sn, ic_c)
    rows.append({"candidate": "sector_neutral", "kind": "replacement", "ic_alone": ic_sn.mean(), "t_alone": E.tstat(ic_sn), "comp_plus_ic": np.nan, "paired_gain": pr["mean_diff"],
                 "paired_t": pr["t"], "verdict": "kill" if pr["t"] < 1 else "inconclusive"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "candidates_dev.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    print("\nLP books on dev months (lc_t10, fresh start)", flush=True)
    frames, prow = {}, []
    for name, s in scores.items():
        r = L.evaluate_book(P.preds_frame(s, PERIOD), C.LP_BASE, keep_frames=True)
        frames[name] = r["frame"]
        prow.append({"book": name, "ir_gross": r["ir_gross"], "ir_net": r["ir_net"], "beta": r["beta"], "max_dd_net_pct": 100 * r["max_dd_net"], "turnover": r["one_way_turnover"],
                     "n_months": r["n_months"], "long_leg_cagr": r["long_leg_cagr"], "short_leg_cagr": r["short_leg_cagr"]})
    for r in prow:
        r["paired_net_t_vs_comp"] = E.paired_net(frames[r["book"]], frames["comp"])["t"] if r["book"] != "comp" else np.nan
    pdf = pd.DataFrame(prow)
    pdf.to_csv(OUT / "lp_dev.csv", index=False)
    print(pdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    E.dump(OUT / "results.json", {"group_ic_dev": grp_ic, "candidates": full, "lp": prow})
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
