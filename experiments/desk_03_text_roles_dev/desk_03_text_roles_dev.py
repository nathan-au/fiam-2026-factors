"""
FIAM 2026 - Other roles for the text desk, DEV period only (desk_03_text_roles_dev). Test window not read.

desk_01/02: as a plain 8th ranking group the text signals add nothing on the tradeable universe in 2015-2020. Change of approach - three different roles:
  F1 persistence   causal exponentially-decayed intensity (half-life 6 months, the value fixed by experiments/feat_8k_502_followup) of 5.02 filings (txt_v1),
                   abrupt exits, hard exits, novel-distress. Replicates the user's 2021-26 finding `mgmt_5_02_decayed_hl6` on EARLIER data.   sign -1
  F2 news-gated    reversal vs continuation gated by novelty: rev_nonovel = -rank(ret_1_0) for stocks WITHOUT a novel (>= 0.5) filing this month (sign +);
                   drift_novel = +rank(ret_1_0) for stocks WITH a novel filing (sign +). Hypothesis: novelty separates information-driven moves (drift)
                   from liquidity-driven ones (reversal); feat_8k_meta killed the cruder 'any 8-K' gate.
  F3 tail overlay  does a flag select next-month TAIL losses (y <= -15%) even if it does not move the mean? monthly flagged-minus-unflagged tail rate, t over months.
Tests F1+F2 = 7 (Bonferroni 7). F3 is descriptive (7 flags x 2 statistics), judged by t and direction only.
Run:  .venv/bin/python experiments/desk_03_text_roles_dev/desk_03_text_roles_dev.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
PERIOD = "dev"


def tail_table(P, flag, name, thr=-0.15):
    m = P.mask(PERIOD)
    d = pd.DataFrame({"m": P.months[m], "f": flag[m], "tail": (P.y[m] <= thr).astype(float), "y": P.y[m]})
    g = d.groupby(["m", "f"]).agg(tail=("tail", "mean"), y=("y", "mean"), n=("y", "size")).unstack("f")
    ok = g["n"][True].fillna(0) >= 5
    dt = (g["tail"][True] - g["tail"][False])[ok]
    dy = (g["y"][True] - g["y"][False])[ok]
    return {"flag": name, "months_ok": int(ok.sum()), "flagged_per_month": float(g["n"][True][ok].mean()), "tail_rate_flagged": float(g["tail"][True][ok].mean()),
            "tail_rate_unflagged": float(g["tail"][False][ok].mean()), "tail_diff": float(dt.mean()), "tail_diff_t": E.tstat(dt),
            "mean_ret_diff_pct": float(100 * dy.mean()), "mean_ret_diff_t": E.tstat(dy)}


def main():
    t0 = time.time()
    P = Panel()
    comp = FD.composite(P)
    v1, v2 = TD.load_blocks()
    raw = TD.raw_arrays(P, v1, v2)
    raw["ret_1_0"] = P.df["raw_ret_1_0"].to_numpy()
    print(f"composite dev IC {E.ic_summary(E.ic_series(P, comp.score, PERIOD))}", flush=True)

    # ---- F1 persistence
    blocks = {}
    for name, col in (("m502_hl6", "txt_item_5_02"), ("abrupt_hl6", "txt_v2_n_502_abrupt_exit"), ("hard_abrupt_hl6", "txt_v2_n_502_hard_abrupt"),
                      ("novel_distress_hl6", "txt_v2_any_novel_distress")):
        x = np.nan_to_num(raw[col], nan=0.0)
        blocks[name] = -FD.urank(TD.decay(P, x, 6.0), P)
    # ---- F2 news-gated
    nov = raw["txt_v2_novelty_max"]
    novel = np.nan_to_num(nov, nan=-1.0) >= 0.5
    r = FD.urank(raw["ret_1_0"], P)
    blocks["rev_nonovel"] = -r * (~novel)
    blocks["drift_novel"] = r * novel
    blocks["_control_rev_all"] = -r  # reference only, not a test
    tests = [k for k in blocks if not k.startswith("_")]
    rows, full = [], {}
    for name, b in blocks.items():
        rep = E.block_report(P, comp, b, PERIOD, name, n_perm=200, seed=2)
        full[name] = rep
        rows.append({"block": name, "nonzero_share": rep["nonzero_share_universe"], "ic": rep["block"]["ic"], "t": rep["block"]["t"], "resid_ic": rep["residual_ic"],
                     "resid_t": rep["residual_ic_t"], "corr_comp": rep["rank_corr_with_comp"], "paired_gain": rep["paired_gain"]["mean_diff"],
                     "paired_t": rep["paired_gain"]["t"], "perm_p": rep["perm"]["p_value"], "bonf_p": min(1, rep["perm"]["p_value"] * len(tests)),
                     "gain_small": rep["tercile_small"]["paired_gain"], "gain_mid": rep["tercile_mid"]["paired_gain"], "gain_large": rep["tercile_large"]["paired_gain"],
                     "verdict": "control" if name.startswith("_") else E.kill_pass(rep, n_tests=len(tests))})
        r_ = rows[-1]
        print(f"  {name:20s} IC {r_['ic']:+.4f} (t {r_['t']:+.2f}) resid {r_['resid_ic']:+.4f} gain {r_['paired_gain']:+.4f} (t {r_['paired_t']:+.2f}) p {r_['perm_p']:.3f} -> {r_['verdict']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "roles_ic.csv", index=False)

    # ---- F3 tail overlay
    print("\nF3 tail overlay: flagged-minus-unflagged next-month tail-loss rate (y <= -15%), universe, monthly, dev", flush=True)
    flags = {n: TD.flag_array(P, raw, [n]) for n in ("abrupt_exit", "hard_abrupt", "distress_801", "litigation", "novel_distress", "financing", "v1_any_distress")}
    top = np.nan_to_num(FD.urank(raw["txt_v2_novel_negative_max"], P), nan=0.0) > 0.8
    flags["novneg_top20pct_of_filers"] = top & P.u
    trows = [tail_table(P, f, n) for n, f in flags.items()]
    tdf = pd.DataFrame(trows)
    tdf.to_csv(OUT / "tail_overlay.csv", index=False)
    print(tdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    E.dump(OUT / "results.json", {"f1_f2": full, "tail": trows})
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
