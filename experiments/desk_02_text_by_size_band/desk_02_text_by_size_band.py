"""
FIAM 2026 - Where does the text edge live? (desk_02_text_by_size_band). DEV period only (2015-02..2020-12 target months); the test window is not read.

desk_01 found that the text lane's headline IC replicates on ALL stocks conditional on a filing (novelty -0.012, t -2.5) but is ~0 in the >= $2B universe.
Question: is there a market-cap band that PM screens still admit (price >= $5, $10M dollar volume, both betas) where text carries signal, so that the
desk system's universe floor should be lower than the frozen $2B?

Design: (a) raw signal IC among stocks WITH a filing, by market-cap band (all price/dollar-volume/beta screens on); (b) for floors 500 / 1000 / 2000 ($M) the
composite's dev IC and the paired gain of adding each primary text signal as an 8th group (frozen kill/pass rule), permutation null 200.
Floors were fixed BEFORE looking: they are the two lower floors already used in this project (pm_ablation $500M, largecap sensitivity $1B) plus the frozen $2B.
Run:  .venv/bin/python experiments/desk_02_text_by_size_band/desk_02_text_by_size_band.py
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
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 1e9)]
SIGS = list(TD.PRIMARY) + ["novelty_mean", "novel_distress"]


def main():
    t0 = time.time()
    v1, v2 = TD.load_blocks()
    res = {"bands": {}, "floors": {}}
    P = Panel(floor=0.0)  # screens except the market-cap floor; bands are cut below
    raw = TD.raw_arrays(P, v1, v2)
    df = P.df
    screens = ((df["raw_prc"].abs() >= C.MIN_PRICE) & (df["raw_dolvol_126d"] >= C.MIN_DOLVOL) & df["raw_beta_60m"].notna() & df["raw_betabab_1260d"].notna()).to_numpy()
    dev = P.mask(PERIOD, universe=False) & screens
    me = df["raw_me"].to_numpy()
    print("(a) raw signal IC among filers, by market-cap band (signed: + = predicted direction; dev; screens on)", flush=True)
    rows = []
    for col, name in (("txt_v2_novelty_max", "novelty_max"), ("txt_v2_novel_negative_max", "novneg_max"), ("txt_v2_neg_mean", "neg_mean")):
        x = -raw[col]  # sign -1
        for lo, hi in BANDS:
            m = dev & (me >= lo) & (me < hi) & np.isfinite(x)
            ic = E.monthly_rank_ic(P.months[m], x[m], P.y[m])
            rows.append({"signal": name, "band": f"{lo}-{hi if hi < 1e8 else 'inf'}", "ic": ic.mean(), "t": E.tstat(ic), "n_months": ic.notna().sum(), "rows_per_month": m.sum() / ic.notna().sum()})
    bd = pd.DataFrame(rows)
    print(bd.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    bd.to_csv(OUT / "band_ic.csv", index=False)
    res["bands"] = bd.to_dict("records")

    print("\n(b) floor sweep: composite dev IC and paired gain of each text signal (8th group), dev", flush=True)
    frows = []
    for floor in (500.0, 1000.0, 2000.0):
        Pf = Panel(floor=floor)
        comp = FD.composite(Pf)
        rawf = TD.raw_arrays(Pf, v1, v2)
        ic_c = E.ic_series(Pf, comp.score, PERIOD)
        print(f"  floor ${floor:,.0f}M: universe rows/month {Pf.mask(PERIOD).sum() / ic_c.notna().sum():.0f}; composite dev IC {ic_c.mean():+.4f} (t {E.tstat(ic_c):+.2f})", flush=True)
        for n in SIGS:
            s = TD.signal_score(Pf, rawf, n)
            r = E.block_report(Pf, comp, s, PERIOD, n, n_perm=200, seed=1)
            frows.append({"floor": floor, "signal": n, "n_univ_month": Pf.mask(PERIOD).sum() / ic_c.notna().sum(), "comp_ic": ic_c.mean(), "comp_t": E.tstat(ic_c),
                          "block_ic": r["block"]["ic"], "block_t": r["block"]["t"], "resid_ic": r["residual_ic"], "gain": r["paired_gain"]["mean_diff"],
                          "gain_t": r["paired_gain"]["t"], "perm_p": r["perm"]["p_value"], "gain_small": r["tercile_small"]["paired_gain"],
                          "gain_mid": r["tercile_mid"]["paired_gain"], "gain_large": r["tercile_large"]["paired_gain"], "verdict": E.kill_pass(r, n_tests=len(SIGS) * 3)})
            f = frows[-1]
            print(f"    {n:14s} block IC {f['block_ic']:+.4f} (t {f['block_t']:+.2f}) resid {f['resid_ic']:+.4f} gain {f['gain']:+.4f} (t {f['gain_t']:+.2f}) p {f['perm_p']:.3f} -> {f['verdict']}", flush=True)
    fd = pd.DataFrame(frows)
    fd.to_csv(OUT / "floor_sweep.csv", index=False)
    res["floors"] = fd.to_dict("records")
    E.dump(OUT / "results.json", res)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
