"""
desk_13_baseline_freeze - freeze the committed default system (commit 6e145be, "Test desks") as the reference every later experiment is compared to.

Baseline B0 = fiam_desks.system.DEFAULT exactly: frozen 7-group composite + days-to-cover 8th group; lc_t10 LP; SI-ratio short cap 10%; text advisory;
FINRA short interest from 2020-06 only (fiam_desks/si.py). Runs DEV (2015-02..2020-12) and TEST (2021-01..2026-08) and writes the full metric suite
(fiam_research.core.metrics), the monthly return series and the holdings, plus hashes of every source file of fiam_desks/.
TEST numbers were already reported in desk_09/desk_10; recomputing them here selects nothing (ledgered as a descriptive baseline freeze).
Run: .venv/bin/python experiments/desk_13_baseline_freeze/desk_13_baseline_freeze.py
"""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from fiam_desks import config as C, evaluate as E, system as S  # noqa: E402
from fiam_research import core, ledger  # noqa: E402

OUT = HERE / "output"
EXP = "desk_13_baseline_freeze"


def main():
    t0 = time.time()
    ctx = core.Ctx()
    P = ctx.P
    score = ctx.baseline_score(extended_si=False)
    res_all = {}
    for period in ("dev", "test"):
        r = core.book(ctx, score, period, si_cap=True, extended_si=False)
        m = core.metrics(ctx, r, score, period)
        res_all[period] = m
        fr = r["frame"]
        fr.to_csv(OUT / f"baseline_{period}_monthly.csv", index=False)
        r["holdings"][["target_month", "permno", "ticker", "sector", "weight", "raw_me", C.TARGET_COL]].to_csv(OUT / f"baseline_{period}_holdings.csv.gz", index=False)
        print(f"{period.upper():5s}", core.short(m), flush=True)
        ledger.record(EXP, "baseline", "B0_default", "freeze the committed default system", {"system": S.DEFAULT}, "n/a", "n/a", period, 1, m,
                      decision="baseline", reason="descriptive baseline freeze; test already seen in desk_09/10" if period == "test" else "descriptive")
    # reproduction check against the published F1_cap row
    res_all["reproduces_desk10"] = bool(abs(res_all["test"]["ir_net"] - 0.637) < 5e-3 and abs(res_all["test"]["ir_gross"] - 0.721) < 5e-3)
    src = {p.name: C.file_sha256(p) for p in sorted((ROOT / "fiam_desks").glob("*.py"))}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = {"commit": commit, "system_default": S.DEFAULT, "lp_base": C.LP_BASE, "si_cap": C.SI_CAP, "factor_groups": C.FACTOR_GROUPS,
           "universe": {"floor_mcap_musd": C.FLOOR_MCAP, "min_price": C.MIN_PRICE, "min_dolvol": C.MIN_DOLVOL},
           "periods": {"dev": C.DEV, "test": C.TEST}, "fiam_desks_sha256": src, "short_interest_months": "2020-06 onward (fiam_desks/si.py)"}
    (HERE / "baseline_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    E.dump(OUT / "baseline_metrics.json", res_all)
    print("reproduces desk_10:", res_all["reproduces_desk10"], f"| {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
