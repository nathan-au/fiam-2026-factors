"""
FIAM 2026 - desk system infrastructure audit (desk_00_infra_audit).

Checks, before any signal is evaluated, that (1) the new fiam_desks package reproduces the frozen composite + LP exactly, (2) the teammate's text tables
are what they claim to be (txt_v1 rebuilt from the raw 8-K file; txt_v2 structural invariants; novelty recomputed independently, past-only vs a
future-inclusive control), (3) the desk pipeline is causal (truncation invariance) and deterministic, (4) the target column is really next month's return.

Run:  .venv/bin/python experiments/desk_00_infra_audit/desk_00_infra_audit.py [--smoke]
Outputs (output/): results.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from fiam_desks import audit as A  # noqa: E402
from fiam_desks import config as C  # noqa: E402
from fiam_desks import evaluate as E  # noqa: E402
from fiam_desks import factors_desk as FD  # noqa: E402
from fiam_desks import lp as L  # noqa: E402
from fiam_desks import text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--novelty-permnos", type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    res = {"config": {"published": C.PUBLISHED, "dev": C.DEV, "test": C.TEST, "floor": C.FLOOR_MCAP}}

    print("[1] reproduce the frozen composite + lc_t10", flush=True)
    P = Panel(smoke=args.smoke)
    comp = FD.composite(P)
    ic = E.ic_series(P, comp.score, "test")
    book = L.evaluate_book(P.preds_frame(comp.score, "test"), C.LP_BASE)
    rep = {"universe_ic": float(ic.mean()), "universe_ic_t": E.tstat(ic), "ir_gross": book["ir_gross"], "ir_net": book["ir_net"],
           "constraints": L.constraint_report(book)}
    rep["passed"] = bool(args.smoke or (abs(rep["universe_ic"] - C.PUBLISHED["comp_universe_ic"]) < 5e-4
                                        and abs(rep["ir_gross"] - C.PUBLISHED["lc_t10_ir_gross"]) < 2e-3
                                        and abs(rep["ir_net"] - C.PUBLISHED["lc_t10_ir_net"]) < 2e-3 and all(rep["constraints"].values())))
    res["reproduce_frozen"] = rep
    print("   ", {k: v for k, v in rep.items() if k != "constraints"}, flush=True)

    print("[2] target alignment", flush=True)
    res["target_alignment"] = A.target_alignment()
    print("   ", res["target_alignment"], flush=True)

    print("[3] text tables", flush=True)
    v1, v2 = TD.load_blocks(verify_hash=True)
    res["text_hashes_verified"] = True
    res["text_v1_vs_raw"] = A.text_v1_vs_raw(v1)
    print("    v1 vs raw:", {k: v for k, v in res["text_v1_vs_raw"].items() if k != "cell_mismatches_by_column"}, flush=True)
    res["text_v2_consistency"] = A.text_v2_consistency(v1, v2)
    print("    v2 invariants:", res["text_v2_consistency"], flush=True)
    res["novelty_recompute"] = A.novelty_recompute(v2, n_permnos=args.novelty_permnos)
    print("    novelty recompute:", res["novelty_recompute"], flush=True)

    print("[4] desk truncation invariance", flush=True)
    raw = TD.raw_arrays(P, v1, v2)

    def text_builder(spec):
        def fn(panel):
            r = TD.raw_arrays(panel, v1[v1["eom"] <= panel.df["eom"].max()], v2[v2["eom"] <= panel.df["eom"].max()])
            return TD.desk(panel, r, spec).score
        return fn

    builders = {
        "factors_composite": lambda p: FD.composite(p).score,
        "text_novelty_two_sided": text_builder({"signals": ["novelty_max"]}),
        "text_novneg_flags_decay6": text_builder({"signals": ["novneg_max", "abrupt_exit"], "half_life": 6.0, "flags": ["novel_distress"]}),
    }
    res["desk_truncation"] = A.desk_truncation(P, builders, n_dates=3)
    print("   ", res["desk_truncation"], flush=True)

    print("[5] determinism", flush=True)
    first = P.preds_frame(comp.score, "dev")
    first = first[first["target_month"] <= "2016-12-31"]
    h = [A.frame_hash(L.build_portfolio(first, C.LP_BASE, verbose=False)[0][["target_month", "permno", "weight"]]) for _ in range(2)]
    res["determinism"] = {"hashes": h, "passed": bool(h[0] == h[1])}
    print("   ", res["determinism"], flush=True)

    res["all_passed"] = bool(all(res[k]["passed"] for k in ("reproduce_frozen", "target_alignment", "text_v1_vs_raw", "text_v2_consistency",
                                                            "novelty_recompute", "desk_truncation", "determinism")))
    res["runtime_min"] = (time.time() - t0) / 60
    E.dump(OUT / ("results_smoke.json" if args.smoke else "results.json"), res)
    print(f"\nALL PASSED: {res['all_passed']}  ({res['runtime_min']:.1f} min)")


if __name__ == "__main__":
    main()
