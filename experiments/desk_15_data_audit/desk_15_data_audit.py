"""
desk_15_data_audit - data-quality and implementation audit of the baseline, with one correction experiment.

Checks (descriptive): universe size by year; missingness of the 18 composite inputs by year (median-filled -> rank 0); the S&P 500 series coverage
(FRED SP500 starts 2016-09, so harness beta/alpha on DEV silently use 2016-10.. only); FINRA SI join coverage; target alignment (desk_00).
Correction experiment (C1): the frozen Panel drops stock-months whose NEXT-month return is missing BEFORE forming the universe. All such universe
rows are the permno's last observation (delisting / acquisition), so the frozen universe uses the future fact 'this stock survives month t+1'.
C1 keeps them (y = 0, a neutral delisting-return assumption; no CRSP delisting returns are available) and re-runs the baseline on DEV and TEST, then
stresses held delisting rows (long -30%, short +20%: acquisition-premium / failure worst cases).
Pre-registered decision: if |net IR change| < 0.05 on both periods under y=0, the leak is immaterial and the frozen Panel stays the research panel
(documented); otherwise all later experiments use PanelX.
TEST is evaluated because this is a correction of the baseline, not a selection among alternatives (ledgered).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402
from fiam_research.panel_ext import PanelX  # noqa: E402

OUT = HERE / "output"
EXP = "desk_15_data_audit"


def main():
    t0 = time.time()
    res = {}
    base = core.Ctx(text=False)
    PX = PanelX(0.0)
    ctxx = core.Ctx(text=False, panel=PX)
    P = base.P
    # descriptive checks
    u = P.u
    yr = pd.DatetimeIndex(P.df["eom"].to_numpy()).year
    res["universe_per_month_by_year"] = {int(k): int(v) for k, v in (pd.Series(u).groupby(yr).sum() / pd.Series(1, index=range(len(u))).groupby(yr).apply(lambda s: 1).reindex(range(2015, 2027)).fillna(1) / 12).round(0).items()} if False else \
        {int(y): round(float(u[yr == y].sum() / len(np.unique(P.df['eom'].to_numpy()[yr == y]))), 0) for y in range(2015, 2027)}
    sp = pd.read_csv(C.CACHE / "SP500.csv")
    res["sp500_series_first_date"] = str(sp["observation_date"].iloc[0])
    res["delist_rows_universe"] = int((PX.delist & PX.u).sum())
    res["delist_rows_universe_by_year"] = {int(k): int(v) for k, v in pd.Series((PX.delist & PX.u)).groupby(pd.DatetimeIndex(PX.df["eom"].to_numpy()).year).sum().items()}
    print(res)
    for period in ("dev", "test"):
        s0 = base.baseline_score(False)
        r0 = core.book(base, s0, period)
        sx = ctxx.baseline_score(False)
        rx = core.book(ctxx, sx, period)
        h = rx["holdings"].merge(PX.df[["permno", "eom", "__delist"]], on=["permno", "eom"], how="left")
        hd = h[h["__delist"]]
        m0, mx = core.metrics(base, r0, s0, period, boot=False), core.metrics(ctxx, rx, sx, period, boot=False)
        # stress: held delisting rows get long -30% / short +20% instead of 0 (gross P&L adjustment, per target month)
        adj = (np.where(hd["weight"] > 0, -0.30, 0.20) * hd["weight"]).groupby(hd["target_month"]).sum()
        net_x = rx["frame"].set_index("target_month")["net_port_excess_ret"]
        stressed = net_x.add(adj, fill_value=0.0)
        res[period] = {"frozen": {k: m0[k] for k in ("ir_net", "ir_gross", "ic", "max_dd_net")}, "with_delist_rows_y0": {k: mx[k] for k in ("ir_net", "ir_gross", "ic", "max_dd_net")},
                       "held_delist_positions": int(len(hd)), "held_delist_long": int((hd["weight"] > 0).sum()), "held_delist_short": int((hd["weight"] < 0).sum()),
                       "held_delist_abs_weight_sum": float(hd["weight"].abs().sum()), "stressed_ir_net": ST.ir(stressed - core.RF_HALF),
                       "ir_net_change_y0": mx["ir_net"] - m0["ir_net"]}
        print(period, res[period], flush=True)
        ledger.record(EXP, "audit_delist", "baseline_with_delist_rows", "target availability must not define the universe", {"fill": 0.0}, "n/a", "n/a", period, 1, mx,
                      decision="correction", reason="audit correction of the baseline, no selection")
    res["decision"] = "immaterial: keep frozen Panel" if all(abs(res[p]["ir_net_change_y0"]) < 0.05 for p in ("dev", "test")) else "material: use PanelX"
    print("DECISION", res["decision"], f"{time.time() - t0:.0f}s")
    E.dump(OUT / "results.json", res)


if __name__ == "__main__":
    main()
