"""
desk_30 follow-up (SECOND STAGE, motivated by X1 passing FDR and X1b failing; counted as extra trials in the ledger):
  Y1  surprise group weight 2 where a 2.02 was filed in t or t-1 (fresh), weight 1 otherwise (conditional weighting instead of zeroing)
  Y2  Y1 + pending-merger names (X2) set to 'no view' (score 0) instead of vetoed
Same rule as desk_30; Sharpe and paired t reported alongside IR (desk_30 note: IR vs T-bill+4% penalises lower-vol variants when returns are below the hurdle).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import importlib.util
spec = importlib.util.spec_from_file_location("d30", HERE / "desk_30_text_untested_uses.py"); d30 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d30)
from fiam_desks import evaluate as E  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402


def y_scores(ctx):
    P, raw = ctx.P, ctx.raw
    G = ctx.comp.extras["groups"]
    fresh = d30.rolling_any(P, raw["txt_item_2_02"], 2)
    merger = d30.rolling_any(P, raw["txt_v2_n_101_merger"], 6)
    w = np.where(fresh, 2.0, 1.0)
    y1 = (G.drop(columns=["surprise"]).sum(axis=1).to_numpy() + w * G["surprise"].to_numpy() + ctx.dtc_block(True)) / (7.0 + w)
    y2 = np.where(merger, 0.0, y1)
    return {"Y1_fresh_surprise_x2": y1, "Y2_Y1_plus_merger_noview": y2}, fresh, merger


def main():
    ctx = core.research_ctx(text=True)
    P = ctx.P
    b1 = core.b1_score(ctx)
    S, _, _ = y_scores(ctx)
    arms = {"B1": b1, **S}
    B = core.run_many(ctx, [(k, s, "dev", core.b1_kw()) for k, s in arms.items()])
    f0 = B["B1"]["frame"].set_index("target_month")["net_port_excess_ret"]
    m0 = core.metrics(ctx, B["B1"], b1, "dev", boot=False)
    ic0 = E.ic_series(P, b1, "dev")
    rows = []
    for k, s in arms.items():
        m = core.metrics(ctx, B[k], s, "dev", boot=False)
        f = B[k]["frame"].set_index("target_month")["net_port_excess_ret"]
        pt = ST.paired(f, f0)["t"] if k != "B1" else np.nan
        ict = ST.tstat(E.ic_series(P, s, "dev") - ic0) if k != "B1" else np.nan
        ok = k != "B1" and m["ir_net"] >= m0["ir_net"] + 0.15 and pt >= 1 and m["max_dd_net"] >= m0["max_dd_net"] - 0.05
        dec = "reference" if k == "B1" else ("promising" if ok else ("rejected" if pt <= -1 else "no evidence"))
        rows.append({"arm": k, "ic": m["ic"], "paired_ic_t": ict, "ir_net": m["ir_net"], "sharpe": m["sharpe_net"], "max_dd": m["max_dd_net"], "paired_net_t": pt, "decision": dec})
        if k != "B1":
            ledger.record("desk_30_text_untested_uses", "text_followup", k, "second-stage conditional weighting (post X1/X2)", {}, "dev", "dev", "dev", 2, m, paired_t=pt, decision=dec)
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "output/followup_dev.csv", index=False)
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
