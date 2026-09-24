"""
FIAM 2026 - Text desk, raw signal study on the DEV period only (desk_01_text_signals_dev).

Which of the teammate's txt_v1 / txt_v2 signals carry information on the tradeable (>= $2B) universe, on 2015-02..2020-12 target months, with the
sign fixed in advance, after the frozen composite? The FIAM scoring window (2021-2026) is NOT read here (Panel period 'test' is never selected).

PRE-REGISTERED (before any return was looked at)
  primary family     novelty_max, novneg_max, abrupt_exit                     (the three the text lane headlines) -> Bonferroni 3
  secondary family   the rest of fiam_desks.text_desk.SIGNALS + one-sided variants of the rank signals -> Bonferroni over ALL signals tested here
  rule               docs/EXPERIMENTS.md kill/pass rule, adding the signal as an 8th equal-weight group to the frozen composite
  extra evidence     residual IC after the composite, 6-month-lag placebo (an event signal should decay), size terciles, permutation null 200
Run:  .venv/bin/python experiments/desk_01_text_signals_dev/desk_01_text_signals_dev.py
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


def all_stock_ic(P, score):
    m = P.mask(PERIOD, universe=False)
    return E.monthly_rank_ic(P.months[m], np.asarray(score)[m], P.y[m])


def main():
    t0 = time.time()
    P = Panel()
    comp = FD.composite(P)
    v1, v2 = TD.load_blocks()
    raw = TD.raw_arrays(P, v1, v2)
    lag6 = lambda v: v.assign(eom=v["eom"] + pd.offsets.MonthEnd(6))
    raw_l6 = TD.raw_arrays(P, lag6(v1), lag6(v2))
    print(f"composite dev IC {E.ic_summary(E.ic_series(P, comp.score, PERIOD))}", flush=True)

    tests = [(n, "two_sided") for n in TD.SIGNALS] + [(n, "one_sided") for n in TD.SIGNALS if TD.SIGNALS[n][1] == "rank"]
    rows, full = [], {}
    for name, mode in tests:
        s = TD.signal_score(P, raw, name, mode)
        s6 = TD.signal_score(P, raw_l6, name, mode)
        r = E.block_report(P, comp, s, PERIOD, f"{name}/{mode}", w=1.0, n_perm=200, lag_block=s6, seed=1)
        r["all_stock_ic"] = E.ic_summary(all_stock_ic(P, s))
        full[f"{name}/{mode}"] = r
        rows.append({
            "signal": name, "mode": mode, "family": "primary" if (name in TD.PRIMARY and mode == "two_sided") else "secondary",
            "nonzero_share": r["nonzero_share_universe"], "ic_universe": r["block"]["ic"], "t_universe": r["block"]["t"],
            "ic_all_stocks": r["all_stock_ic"]["ic"], "t_all_stocks": r["all_stock_ic"]["t"],
            "resid_ic": r["residual_ic"], "resid_t": r["residual_ic_t"], "corr_comp": r["rank_corr_with_comp"],
            "paired_gain": r["paired_gain"]["mean_diff"], "paired_t": r["paired_gain"]["t"], "perm_p": r["perm"]["p_value"],
            "ic_lag6": r["placebo_lag"]["ic"], "t_lag6": r["placebo_lag"]["t"],
            "gain_small": r["tercile_small"]["paired_gain"], "gain_mid": r["tercile_mid"]["paired_gain"], "gain_large": r["tercile_large"]["paired_gain"],
            "ic_small": r["tercile_small"]["block_ic"], "ic_mid": r["tercile_mid"]["block_ic"], "ic_large": r["tercile_large"]["block_ic"],
        })
        print(f"  {name:16s} {mode:9s} IC {rows[-1]['ic_universe']:+.4f} (t {rows[-1]['t_universe']:+.2f}) all-stock {rows[-1]['ic_all_stocks']:+.4f} "
              f"| resid {rows[-1]['resid_ic']:+.4f} | gain {rows[-1]['paired_gain']:+.4f} (t {rows[-1]['paired_t']:+.2f}) p {rows[-1]['perm_p']:.3f} "
              f"| lag6 IC {rows[-1]['ic_lag6']:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    n_all = len(df)
    df["bonf_p"] = np.where(df["family"] == "primary", np.minimum(1, df["perm_p"] * 3), np.minimum(1, df["perm_p"] * n_all))
    df["verdict"] = [E.kill_pass(full[f"{r.signal}/{r.mode}"], n_tests=(3 if r.family == "primary" else n_all)) for r in df.itertuples()]
    df.to_csv(OUT / "signals_dev.csv", index=False)
    E.dump(OUT / "results.json", {"composite_dev_ic": E.ic_summary(E.ic_series(P, comp.score, PERIOD)), "n_signals_tested": n_all, "signals": full})
    print("\n" + df[["signal", "mode", "family", "ic_universe", "t_universe", "ic_all_stocks", "resid_ic", "paired_gain", "paired_t", "bonf_p", "ic_lag6", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
