"""
FIAM 2026 - Deterministic PM hardening: the pre-existing neutralisation dial, replicated on the DEV period (desk_06_pm_neutral_dial_dev). Test window not read.

desk_05 showed the frozen composite's lc_t10 book LOSES on 2015-2020 (net IR about -0.9; -30% in 2020 alone, short leg -50%: the shorts are high-idiosyncratic-vol names that
rallied). experiments/neutral_dial (test period) defined a dial D1..D4 of exact-neutrality constraints added to the same LP: D2 +log size, D3 +12-1 momentum, D4 +idio vol +quality.
No new PM idea is invented here: the dial and its columns are taken as defined there (fixed before this run). Question: does deeper neutralisation fix the dev-period loss, and at what
cost - i.e. is the failure a property of the signal or of what the unconstrained LP lets it hold? Also `w05` (per-name cap 0.5%, at least 400 names) from experiments/largecap.
Run:  .venv/bin/python experiments/desk_06_pm_neutral_dial_dev/desk_06_pm_neutral_dial_dev.py
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
    base = C.LP_BASE
    variants = {
        "D1_beta_sector (current lc_t10)": base,
        "D2_plus_size": {**base, "extra_neutral": ["x_size"]},
        "D3_plus_size_mom": {**base, "extra_neutral": ["x_size", "x_mom"]},
        "D4_plus_size_mom_vol_qual": {**base, "extra_neutral": ["x_size", "x_mom", "x_vol", "x_qual"]},
        "D1_w05": {**base, "max_weight": 0.005},
        "D4_w05": {**base, "max_weight": 0.005, "extra_neutral": ["x_size", "x_mom", "x_vol", "x_qual"]},
    }
    preds = P.preds_frame(comp.score, PERIOD)
    rows, frames = [], {}
    for name, cfg in variants.items():
        r = L.evaluate_book(preds, cfg, keep_frames=True)
        frames[name] = r["frame"]
        cy = r["calendar_year_net"]
        rows.append({"variant": name, "ir_gross": r["ir_gross"], "ir_net": r["ir_net"], "cagr_net_pct": 100 * r["cagr_net"], "beta": r["beta"], "roll_beta_min": r["roll_beta_min"],
                     "roll_beta_max": r["roll_beta_max"], "max_dd_net_pct": 100 * r["max_dd_net"], "one_way_turnover": r["one_way_turnover"], "months_relaxed": r["months_turnover_relaxed"],
                     "min_positions": r["min_positions"], **{f"net_{y}": cy.get(y, np.nan) for y in range(2015, 2021)}, "long_leg_cagr": r["long_leg_cagr"], "short_leg_cagr": r["short_leg_cagr"]})
        print(f"  {name:34s} IR gross {r['ir_gross']:+.2f} net {r['ir_net']:+.2f} | maxDD {100 * r['max_dd_net']:.0f}% | 2019 {cy.get(2019, np.nan):+.3f} 2020 {cy.get(2020, np.nan):+.3f} | "
              f"beta {r['beta']:+.2f} | relaxed {r['months_turnover_relaxed']}", flush=True)
    df = pd.DataFrame(rows)
    ctrl = "D1_beta_sector (current lc_t10)"
    df["paired_net_t_vs_D1"] = [E.paired_net(frames[v], frames[ctrl])["t"] if v != ctrl else np.nan for v in df["variant"]]
    df.to_csv(OUT / "dial_dev.csv", index=False)
    print("\n" + df[["variant", "ir_gross", "ir_net", "paired_net_t_vs_D1", "max_dd_net_pct", "net_2019", "net_2020", "beta"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    E.dump(OUT / "results.json", rows)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
