"""
FIAM 2026 - Does a momentum group help the factors desk? DEV period only (desk_07_factors_momentum_dev). Test window not read.

Why: desk_05/06 showed the frozen composite's book loses on 2015-2020 for signal reasons (value and accrual groups had negative IC, the value drought), not because of what the LP
holds. The frozen factor set excluded momentum by design (crash risk). One economic group from docs/FACTORS.md sec 4, declared here BEFORE looking: momentum = mean(rank ret_12_1,
rank resff3_12_1), sign +1, added as an 8th equal-weight group. A SINGLE test (Bonferroni 1). Because the prior on this is regime-dependent (momentum did well pre-2020 and unwound
in 2025-26), the result on dev alone can never justify adoption: adoption rule = PASS on dev AND non-negative paired gain on test at ONE logged confirmation (desk_09).
Run:  .venv/bin/python experiments/desk_07_factors_momentum_dev/desk_07_factors_momentum_dev.py
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
    P = Panel()
    comp = FD.composite(P)
    G = comp.extras["groups"]
    mom = (FD.urank(P.df["raw_ret_12_1"].to_numpy(), P) + FD.urank(P.df["raw_resff3_12_1"].to_numpy(), P)) / 2
    rep = E.block_report(P, comp, mom, PERIOD, "momentum_group", n_perm=200, seed=4)
    print(f"momentum group: IC {rep['block']['ic']:+.4f} (t {rep['block']['t']:+.2f}); composite+mom gain {rep['paired_gain']['mean_diff']:+.4f} (t {rep['paired_gain']['t']:+.2f}); resid {rep['residual_ic']:+.4f}; "
          f"perm p {rep['perm']['p_value']:.3f}; terciles gain S/M/L {rep['tercile_small']['paired_gain']:+.4f}/{rep['tercile_mid']['paired_gain']:+.4f}/{rep['tercile_large']['paired_gain']:+.4f}; "
          f"IC by year {rep['block_ic_by_year']} -> {E.kill_pass(rep, 1)}", flush=True)
    s8 = (G.sum(axis=1).to_numpy() + mom) / 8
    rows, frames = [], {}
    for name, s in (("comp", comp.score), ("comp_plus_momentum", s8), ("momentum_only", mom)):
        r = L.evaluate_book(P.preds_frame(s, PERIOD), C.LP_BASE, keep_frames=True)
        frames[name] = r["frame"]
        cy = r["calendar_year_net"]
        rows.append({"book": name, "ir_gross": r["ir_gross"], "ir_net": r["ir_net"], "max_dd_net_pct": 100 * r["max_dd_net"], "beta": r["beta"], **{f"net_{y}": cy.get(y, np.nan) for y in range(2015, 2021)}})
    df = pd.DataFrame(rows)
    df["paired_net_t_vs_comp"] = [E.paired_net(frames[b], frames["comp"])["t"] if b != "comp" else np.nan for b in df["book"]]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(OUT / "momentum_dev.csv", index=False)
    E.dump(OUT / "results.json", {"block": rep, "books": rows})


if __name__ == "__main__":
    main()
