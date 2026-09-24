"""
FIAM 2026 - The single confirmation run on the TEST window (desk_09_confirmation_test). Read PREREGISTRATION.md first: arms, statistics and decision rules were fixed before this ran.
Every arm is appended to experiments/desk_test_ledger.csv. Requires --confirm (guard against accidental test-window looks).
Run:  .venv/bin/python experiments/desk_09_confirmation_test/desk_09_confirmation_test.py --confirm
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, ledger, lp as L, pm as PM, si, text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
PERIOD = "test"
EXP = "desk_09_confirmation_test"
TEXT_SPEC = {"signals": ["novelty_max", "novneg_max", "abrupt_exit"], "mode": "two_sided", "flags": ["novel_distress", "hard_abrupt", "litigation"]}
MODES = {"blend": ("blend", {"w": 1.0}), "veto_long": ("veto_long", {}), "tilt": ("tilt", {"lam": 0.25}), "dial": ("dial", {"kappa": 0.5}), "judge": ("judge", {"a_min": 5, "use_text": True})}
DEV = json.loads((HERE.parent / "desk_08_pm_integration_dev/output/results.json").read_text())
DEV_ARMS = {a["arm"].split("_", 1)[1]: a for a in DEV["arms"]}
DEV_PAIRED = {"blend": DEV_ARMS["blend_w1"], "veto_long": DEV_ARMS["veto_long"], "tilt": DEV_ARMS["tilt_0.25"], "dial": DEV_ARMS["dial_0.5"], "judge": DEV_ARMS["judge"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    if not args.confirm:
        sys.exit("refusing to read the test window without --confirm (see PREREGISTRATION.md)")
    t0 = time.time()
    P = Panel()
    comp = FD.composite(P)
    S = si.features(P)
    v1, v2 = TD.load_blocks()
    raw = TD.raw_arrays(P, v1, v2)
    txt = TD.desk(P, raw, TEXT_SPEC)
    dtc_block = -FD.urank(S["dtc"], P)
    rev1 = -FD.urank(P.df["raw_ret_1_0"].to_numpy(), P)
    F0 = comp
    F1 = FD.add_groups(comp, {"dtc": dtc_block}, "factors+dtc")
    F1r = FD.add_groups(comp, {"dtc": dtc_block, "rev_1m": rev1}, "factors+dtc+rev1m")
    cfg_nocap, cfg_cap = C.LP_BASE, PM.cfg_with_si_cap()
    extras = {"sir": S["sir"]}

    def book(name, fac, cfg, mode="factors_only", **kw):
        dec = PM.decide(fac, txt if mode != "factors_only" else None, mode, **kw)
        r = L.evaluate_book(PM.lp_frame(P, dec, PERIOD, extras), cfg, keep_frames=True)
        ledger.record(EXP, name, {"cfg": cfg, "mode": mode, "kw": kw, "factors": fac.name, "text": TEXT_SPEC if mode != "factors_only" else None})
        ic = E.ic_series(P, dec["pred"], PERIOD)
        cy = r["calendar_year_net"]
        row = {"arm": name, "ic_pm_score": ic.mean(), "ic_t": E.tstat(ic), "ir_gross": r["ir_gross"], "ir_net": r["ir_net"], "sharpe_net": r["sharpe_net"], "cagr_net_pct": 100 * r["cagr_net"],
               "beta": r["beta"], "roll_beta_min": r["roll_beta_min"], "roll_beta_max": r["roll_beta_max"], "max_dd_net_pct": 100 * r["max_dd_net"], "net_2025": cy.get(2025, np.nan),
               "turnover": r["one_way_turnover"], "min_pos": r["min_positions"], "max_pos": r["max_positions"], "months_veto_dropped": r["veto_dropped_months"],
               "constraints_ok": all(L.constraint_report(r).values())}
        print(f"  {name:34s} IC {row['ic_pm_score']:+.4f} | IR gross {r['ir_gross']:+.3f} net {r['ir_net']:+.3f} | maxDD {row['max_dd_net_pct']:.1f}% | 2025 {row['net_2025']:+.3f} | "
              f"pos {r['min_positions']}-{r['max_positions']} | {'ok' if row['constraints_ok'] else 'CONSTRAINT VIOLATION'}", flush=True)
        return row, r["frame"], ic

    rows, frames, ics = [], {}, {}
    print("factors-desk bases (reproduce published rows)", flush=True)
    for name, fac, cfg in (("F0_composite", F0, cfg_nocap), ("F0_cap", F0, cfg_cap), ("F1_dtc", F1, cfg_nocap), ("F1_cap (DEFAULT base)", F1, cfg_cap)):
        row, fr, ic = book(name, fac, cfg)
        rows.append(row); frames[name] = fr; ics[name] = ic
    pub = {"F0_composite": (0.615, 0.541), "F0_cap": (0.590, 0.508), "F1_dtc": (0.659, 0.584), "F1_cap (DEFAULT base)": (0.721, 0.637)}
    repro = {k: {"published_gross_net": v, "got": (rows[i]["ir_gross"], rows[i]["ir_net"]), "ok": abs(rows[i]["ir_gross"] - v[0]) < 5e-3 and abs(rows[i]["ir_net"] - v[1]) < 5e-3} for i, (k, v) in enumerate(pub.items())}
    print("  reproduction of published rows:", {k: v["ok"] for k, v in repro.items()}, flush=True)

    print("\ntext modes layered on the default base F1_cap", flush=True)
    base = "F1_cap (DEFAULT base)"
    for m, (mode, kw) in MODES.items():
        row, fr, ic = book(f"F1_cap + text:{m}", F1, cfg_cap, mode, **kw)
        rows.append(row); frames[row["arm"]] = fr; ics[row["arm"]] = ic
    print("\nrobustness: candidates on the base without dtc (F0_cap)", flush=True)
    for m in ("tilt", "judge"):
        mode, kw = MODES[m]
        row, fr, ic = book(f"F0_cap + text:{m}", F0, cfg_cap, mode, **kw)
        rows.append(row); frames[row["arm"]] = fr; ics[row["arm"]] = ic
    print("\nfactors option carried from dev: rev_1m", flush=True)
    row, fr, ic = book("F1_cap + rev_1m", F1r, cfg_cap)
    rows.append(row); frames[row["arm"]] = fr; ics[row["arm"]] = ic

    df = pd.DataFrame(rows)
    base_of = lambda a: "F0_cap" if a.startswith("F0_cap +") else base
    df["paired_net_t_vs_base"] = [E.paired_net(frames[a], frames[base_of(a)])["t"] if (" + " in a) else np.nan for a in df["arm"]]
    df["paired_ic_t_vs_base"] = [E.paired(ics[a], ics[base_of(a)])["t"] if (" + " in a) else np.nan for a in df["arm"]]
    b_dd = float(df.loc[df["arm"] == base, "max_dd_net_pct"].iloc[0])
    dec = []
    for a in df["arm"]:
        if " + text:" not in a or a.startswith("F0_cap"):
            dec.append(""); continue
        m = a.split("text:")[1]
        d = DEV_PAIRED[m]
        r = df[df["arm"] == a].iloc[0]
        a_ok = bool(d["paired_net_t_vs_A0"] >= 0.5 and (d.get("placebo_share_ge_real") is not None and d["placebo_share_ge_real"] <= 0.05)) if "paired_net_t_vs_A0" in d else None
        dev_t = next(x for x in DEV["arms"] if x["arm"] == d["arm"])
        rr = json.loads((HERE.parent / "desk_08_pm_integration_dev/output/results.json").read_text())
        pl = rr["placebo"].get(d["arm"], {}).get("share_placebos_ge_real", 1.0)
        pt = pd.read_csv(HERE.parent / "desk_08_pm_integration_dev/output/integration_dev.csv").set_index("arm").loc[d["arm"], "paired_net_t_vs_A0"]
        a_ok = bool(pt >= 0.5 and pl <= 0.05)
        b_ok = bool(r["paired_net_t_vs_base"] >= 1.0 and r["ic_pm_score"] >= float(df.loc[df["arm"] == base, "ic_pm_score"].iloc[0]))
        c_ok = bool(r["constraints_ok"] and r["max_dd_net_pct"] >= b_dd - 2.0)
        dec.append(f"a(dev)={'Y' if a_ok else 'n'} b(test)={'Y' if b_ok else 'n'} c(risk)={'Y' if c_ok else 'n'} -> {'ADOPT' if (a_ok and b_ok and c_ok) else 'advisory'}")
    df["decision_rule2"] = dec
    df.to_csv(OUT / "confirmation_test.csv", index=False)

    print("\nstandalone text signals on TEST (rule 3: replicated iff pre-registered sign and t >= 2)", flush=True)
    sa = []
    for n in list(TD.PRIMARY):
        s = TD.signal_score(P, raw, n)
        ic = E.ic_series(P, s, PERIOD)
        ledger.record(EXP, f"standalone:{n}", {"signal": n})
        sa.append({"signal": n, "ic": ic.mean(), "t": E.tstat(ic), "replicated": bool(ic.mean() > 0 and E.tstat(ic) >= 2), "ic_by_year": E.by_year(ic)})
    ic = E.ic_series(P, txt.score, PERIOD)
    ledger.record(EXP, "standalone:T_A", TEXT_SPEC)
    sa.append({"signal": "T_A (3-signal desk)", "ic": ic.mean(), "t": E.tstat(ic), "replicated": bool(ic.mean() > 0 and E.tstat(ic) >= 2), "ic_by_year": E.by_year(ic)})
    for r in sa:
        print(f"  {r['signal']:22s} IC {r['ic']:+.4f} (t {r['t']:+.2f}) -> {'replicated' if r['replicated'] else 'not replicated'}  by year {r['ic_by_year']}", flush=True)

    print("\n" + df[["arm", "ic_pm_score", "ir_gross", "ir_net", "paired_net_t_vs_base", "paired_ic_t_vs_base", "max_dd_net_pct", "net_2025", "decision_rule2"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    E.dump(OUT / "results.json", {"books": rows, "reproduction": repro, "standalone_text": sa, "decisions": dict(zip(df["arm"], dec))})
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
