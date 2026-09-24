"""
FIAM 2026 - Deterministic PM integration modes, DEV period only (desk_08_pm_integration_dev). Test window not read.

Arms (all declared before running; parameters are round numbers fixed in docs/DESKS.md, not tuned):
  A0 factors_only       frozen composite (baseline)
  A1 blend              text as an 8th equal-weight group (w = 1)                      text score = T_A
  A2 veto_long          longs barred where the adverse flag fired                       flag = T_B
  A3 tilt               pred = factors - 0.25 * flag                                    flag = T_B
  A4 dial               per-name cap x (1 + 0.5 * sign(factors) * sign(text))           text score = T_A
  A5 judge              shorts must be convicted: >= 5 of 7 factor groups agree OR flag   flag = T_B
  A6 judge_factors_only same without the text flag (ablation: is any effect the text or just the consensus filter?)
  A7 agree_dial         per-name cap x (0.5 + agreement of the 7 factor groups)         (desk-internal confidence, no text)
Text desk T_A = equal-weight mean of signed [novelty_max, novneg_max, abrupt_exit] (two-sided); T_B flag = any of novel_distress, hard_abrupt, litigation.
For each text-dependent arm a PLACEBO: the text desk output is shuffled within (month, sector) 20 times (seeds 0..19) and the same arm is re-run; the real arm is placed in that
null distribution of net IR. Judged on: universe IC of the PM score (paired vs A0), LP net IR and paired net-return t vs A0, drawdown, constraint compliance.
Run:  .venv/bin/python experiments/desk_08_pm_integration_dev/desk_08_pm_integration_dev.py [--placebos 20]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, lp as L, pm as PM, text_desk as TD  # noqa: E402
from fiam_desks.factors_desk import DeskOutput  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
PERIOD = "dev"
ARMS = {
    "A0_factors_only": ("factors_only", {}),
    "A1_blend_w1": ("blend", {"w": 1.0}),
    "A2_veto_long": ("veto_long", {}),
    "A3_tilt_0.25": ("tilt", {"lam": 0.25}),
    "A4_dial_0.5": ("dial", {"kappa": 0.5}),
    "A5_judge": ("judge", {"a_min": 5, "use_text": True}),
    "A6_judge_factors_only": ("judge", {"a_min": 5, "use_text": False}),
    "A7_agree_dial": ("agree_dial", {"lo": 0.5, "hi": 1.5}),
}
TEXT_DEP = ["A1_blend_w1", "A2_veto_long", "A3_tilt_0.25", "A4_dial_0.5", "A5_judge"]


def run_arm(P, fac, txt, name, sweep_frame=False):
    mode, kw = ARMS[name]
    dec = PM.decide(fac, txt, mode, **kw)
    r = L.evaluate_book(PM.lp_frame(P, dec, PERIOD), C.LP_BASE, keep_frames=True)
    return dec, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebos", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    P = Panel()
    fac = FD.composite(P)
    v1, v2 = TD.load_blocks()
    raw = TD.raw_arrays(P, v1, v2)
    txt = TD.desk(P, raw, {"signals": ["novelty_max", "novneg_max", "abrupt_exit"], "mode": "two_sided", "flags": ["novel_distress", "hard_abrupt", "litigation"]})
    print(f"text desk T_A IC (dev universe) {E.ic_summary(E.ic_series(P, txt.score, PERIOD))}; flagged share of universe rows {txt.extras['flag'][P.mask(PERIOD)].mean():.3f}", flush=True)
    ic0 = E.ic_series(P, fac.score, PERIOD)
    rows, frames, decs = [], {}, {}
    for name in ARMS:
        dec, r = run_arm(P, fac, txt, name)
        frames[name], decs[name] = r["frame"], dec
        ic = E.ic_series(P, dec["pred"], PERIOD)
        cy = r["calendar_year_net"]
        rows.append({"arm": name, "ic_pm_score": ic.mean(), "ic_paired_vs_A0_t": E.paired(ic, ic0)["t"], "ir_gross": r["ir_gross"], "ir_net": r["ir_net"], "max_dd_net_pct": 100 * r["max_dd_net"],
                     "beta": r["beta"], "net_2019": cy.get(2019, np.nan), "net_2020": cy.get(2020, np.nan), "min_positions": r["min_positions"], "months_veto_dropped": r["veto_dropped_months"],
                     "turnover": r["one_way_turnover"], "constraints_ok": all(L.constraint_report(r).values())})
        print(f"  {name:24s} PM-score IC {rows[-1]['ic_pm_score']:+.4f} | IR gross {r['ir_gross']:+.2f} net {r['ir_net']:+.2f} | maxDD {100 * r['max_dd_net']:.0f}% | 2020 {cy.get(2020, np.nan):+.3f} | "
              f"constraints {'ok' if rows[-1]['constraints_ok'] else 'VIOLATED'}", flush=True)
    df = pd.DataFrame(rows)
    df["paired_net_t_vs_A0"] = [E.paired_net(frames[a], frames["A0_factors_only"])["t"] if a != "A0_factors_only" else np.nan for a in df["arm"]]

    print(f"\nplacebos ({args.placebos} shuffles each, text output shuffled within month x sector)", flush=True)
    pl = {}
    for name in TEXT_DEP:
        mode, kw = ARMS[name]
        irs = []
        for k in range(args.placebos):
            sc = E.shuffle_within(P, txt.score, seed=k)
            fl = E.shuffle_within(P, txt.extras["flag"], seed=1000 + k)
            t2 = DeskOutput("text_placebo", sc, {"flag": fl, "coverage": txt.extras["coverage"]}, {})
            dec = PM.decide(fac, t2, mode, **kw)
            irs.append(L.evaluate_book(PM.lp_frame(P, dec, PERIOD), C.LP_BASE)["ir_net"])
        real = float(df.loc[df["arm"] == name, "ir_net"].iloc[0])
        pl[name] = {"placebo_ir_net_mean": float(np.mean(irs)), "placebo_ir_net_sd": float(np.std(irs, ddof=1)), "real": real, "share_placebos_ge_real": float((np.array(irs) >= real).mean()),
                    "placebo_irs": [float(x) for x in irs]}
        print(f"  {name:22s} real net IR {real:+.3f} | placebo {np.mean(irs):+.3f} +- {np.std(irs, ddof=1):.3f} | share of placebos >= real {pl[name]['share_placebos_ge_real']:.2f}", flush=True)
    df["placebo_share_ge_real"] = df["arm"].map(lambda a: pl.get(a, {}).get("share_placebos_ge_real", np.nan))
    df.to_csv(OUT / "integration_dev.csv", index=False)
    E.dump(OUT / "results.json", {"arms": rows, "placebo": pl})
    print("\n" + df[["arm", "ic_pm_score", "ir_gross", "ir_net", "paired_net_t_vs_A0", "max_dd_net_pct", "net_2020", "placebo_share_ge_real"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
