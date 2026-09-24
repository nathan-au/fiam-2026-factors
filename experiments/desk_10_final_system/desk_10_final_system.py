"""
FIAM 2026 - The assembled default system, end to end (desk_10_final_system).

Runs fiam_desks.system.build with the DEFAULT config (factors = composite + days-to-cover; PM = lc_t10 LP + 10% short-interest cap; text desk advisory) and writes the FIAM-format
deliverables: holdings.csv (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT in % of NAV), returns.csv, rationale.csv (one row per position-month: what every desk said + a templated reason),
summary.json. Then audits it: reproduces the published F1_cap row, FIAM hard constraints month by month, determinism (two runs, identical holdings), truncation invariance of every desk
output including the short-interest features, and the same system on the DEV period for disclosure (SI data do not exist before 2020-06, so there the cap never binds).
The test-period config is identical to arm `F1_cap (DEFAULT base)` of desk_09 (already in the ledger); this run adds no new test-period arm.
Run:  .venv/bin/python experiments/desk_10_final_system/desk_10_final_system.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import audit as A, config as C, evaluate as E, factors_desk as FD, lp as L, si as SI, system as S, text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"


def main():
    t0 = time.time()
    P = Panel()
    parts = S.desks(P)
    res = {"config": S.DEFAULT}

    out = S.build(P, "test", parts=parts)
    r = out["result"]
    h, ret, rat = S.holdings_fiam(out), S.returns_csv(out), S.rationale(P, out)
    h.to_csv(OUT / "holdings.csv", index=False)
    ret.to_csv(OUT / "returns.csv", index=False)
    rat.to_csv(OUT / "rationale.csv", index=False)
    summ = {k: v for k, v in r.items() if k not in ("frame", "holdings")}
    summ["constraints"] = L.constraint_report(r)
    summ["reproduces_published_F1_cap"] = bool(abs(r["ir_gross"] - 0.721) < 5e-3 and abs(r["ir_net"] - 0.637) < 5e-3)
    res["test"] = summ
    print(f"TEST  : IR gross {r['ir_gross']:+.3f} net {r['ir_net']:+.3f} | beta {r['beta']:+.3f} (t {r['beta_t']:+.2f}) | maxDD {100 * r['max_dd_net']:.1f}% | positions {r['min_positions']}-{r['max_positions']} "
          f"| turnover {100 * r['one_way_turnover']:.1f}% | constraints {summ['constraints']} | reproduces {summ['reproduces_published_F1_cap']}", flush=True)
    print(f"        files: holdings {len(h):,} rows, months {h['Date'].nunique()}, returns {len(ret)} rows, rationale {len(rat):,} rows", flush=True)
    print("        sample rationale:", rat.iloc[0]["reason"], flush=True)

    fr = r["frame"]
    monthly = pd.DataFrame({"month": fr["target_month"], "n_positions": fr["n_positions"], "gross": fr["gross_exposure"], "net": fr["net_exposure"], "beta_60m_exposure": fr["beta_exposure"],
                            "betabab_exposure": fr["betabab_exposure"], "max_abs_sector_net": fr["max_abs_sector_net"], "max_sector_gross_share": fr["max_sector_gross_share"]})
    monthly.to_csv(OUT / "monthly_constraints.csv", index=False)
    res["monthly_constraint_extremes"] = {"positions": [int(monthly["n_positions"].min()), int(monthly["n_positions"].max())], "gross": [float(monthly["gross"].min()), float(monthly["gross"].max())],
                                          "net_abs_max": float(monthly["net"].abs().max()), "beta_abs_max": float(monthly["beta_60m_exposure"].abs().max()), "betabab_abs_max": float(monthly["betabab_exposure"].abs().max())}
    print("        monthly extremes:", res["monthly_constraint_extremes"], flush=True)

    out2 = S.build(P, "test", parts=parts)
    hh = [A.frame_hash(o["result"]["holdings"][["target_month", "permno", "weight"]]) for o in (out, out2)]
    res["determinism"] = {"hashes": hh, "passed": bool(hh[0] == hh[1])}
    print("DETERMINISM:", res["determinism"], flush=True)

    builders = {
        "factors_composite+dtc": lambda p: S.factors_output(p, FD.composite(p), SI.features(p), "composite+dtc").score,
        "short_interest_ratio": lambda p: SI.features(p)["sir"],
        "text_desk_score": lambda p: S.desks(p)[3].score,
    }
    res["truncation"] = A.desk_truncation(P, builders, n_dates=3, seed=1)
    print("TRUNCATION:", res["truncation"], flush=True)

    dev = S.build(P, "dev", parts=parts)["result"]
    res["dev_disclosure"] = {k: v for k, v in dev.items() if k not in ("frame", "holdings")}
    print(f"DEV (2015-02..2020-12, disclosure only; SI cap inactive, dtc = 0): IR gross {dev['ir_gross']:+.3f} net {dev['ir_net']:+.3f} | maxDD {100 * dev['max_dd_net']:.1f}% | 2020 {dev['calendar_year_net'].get(2020):+.3f}", flush=True)
    E.dump(OUT / "summary.json", res)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
