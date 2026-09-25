"""
desk_33_agents_on_pipeline - LLM agents on top of the deterministic pipeline, each with a mechanical verifier (fiam_research/agents.py).
  E  explainability agent: 2-sentence rationales for B1 positions (top 5 long + top 5 short in 6 formation months = 60) and the 10 largest average
     longs / shorts over the TEST window (deck page 2); verifier = numbers must come from the record, own ticker only, no forward-looking language.
     Compared with the deterministic template reason of fiam_desks.system.rationale (the baseline 'agent-free' explanation).
  D  devil's-advocate agent: reads desk_27 + desk_32 results (the run's confirmation JSONs), writes 5 attacks citing JSON paths; verifier resolves them.
Local model (fiam_research.llm.MODEL), temperature 0, cached. No output of either agent reaches a position.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from fiam_desks import evaluate as E, factors_desk as FD, system as S  # noqa: E402
from fiam_research import agents as AG, core, ledger, llm  # noqa: E402

OUT = HERE / "output"
EXP = "desk_33_agents_on_pipeline"
MONTHS = ["2021-03-01", "2021-12-01", "2022-06-01", "2023-09-01", "2025-03-01", "2026-06-01"]


def record(r):
    g = {c[2:]: round(float(r[c]), 2) for c in r.index if c.startswith("g_")}
    top = sorted(g.items(), key=lambda kv: kv[1], reverse=r["weight"] > 0)
    rec = {"ticker": r["ticker"], "side": "long" if r["weight"] > 0 else "short", "weight_pct_nav": round(100 * abs(float(r["weight"])), 2),
           "composite_score": round(float(r["factors_score"]), 2), "group_scores": dict(top), "sector_gics2": str(r["sector"])}
    if np.isfinite(r["si_ratio"]):
        rec["short_interest_pct_shares"] = round(100 * float(r["si_ratio"]), 1)
        rec["days_to_cover"] = round(float(r["dtc_raw"]), 1)
    return rec


def flat(rec):
    f = {k: v for k, v in rec.items() if k != "group_scores"}
    f.update({f"g_{k}": v for k, v in rec["group_scores"].items()})
    return f


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P = ctx.P
    fac = FD.add_groups(ctx.comp, {"dtc": ctx.dtc_block(True)})
    r = core.book(ctx, fac.score, "test", **core.b1_kw())
    out = {"result": r, "comp": ctx.comp, "factors": fac, "si": ctx.S, "raw": ctx.raw, "text": ctx.txt}
    rat = S.rationale(P, out)
    rat.to_csv(OUT / "rationale_B1_test.csv.gz", index=False)
    sel = []
    for m in MONTHS:
        x = rat[rat["Date"] == m]
        sel += [x.nlargest(5, "weight"), x.nsmallest(5, "weight")]
    sel = pd.concat(sel)
    rows = []
    for _, rr in sel.iterrows():
        rec = record(rr)
        o = AG.explain_row(rec)
        txt = o.get("rationale", "") if o.get("ok") else ""
        v = AG.verify_explanation(txt, flat(rec))
        rows.append({"Date": rr["Date"], "ticker": rr["ticker"], "weight": rr["weight"], "llm_rationale": txt, "verified": v["ok"], "unsupported_numbers": v["unsupported_numbers"],
                     "foreign_tickers": v["foreign_tickers"], "forward_language": v["forward_language"], "semantic_errors": v["semantic_errors"], "template_reason": rr["reason"], "seconds": o.get("_ms", 0) / 1000})
    X = pd.DataFrame(rows)
    X.to_csv(OUT / "explanations.csv", index=False)
    res = {"E_n": len(X), "E_verified_share": float(X["verified"].mean()), "E_unsupported_number_share": float((X["unsupported_numbers"].map(len) > 0).mean()),
           "E_foreign_ticker_share": float((X["foreign_tickers"].map(len) > 0).mean()), "E_forward_language_share": float((X["forward_language"].map(len) > 0).mean()),
           "E_semantic_error_share": float((X["semantic_errors"].map(len) > 0).mean()),
           "E_median_seconds": float(X["seconds"].median())}
    print(res, flush=True)
    print(X[["Date", "ticker", "verified", "llm_rationale"]].head(6).to_string(), flush=True)
    # D devil's advocate
    J = {"confirmation_desk27": json.loads((ROOT / "experiments/desk_27_confirmation_test/output/results.json").read_text()),
         "second_look_desk32": json.loads((ROOT / "experiments/desk_32_text_confirmation/output/results.json").read_text())}
    for a in J["confirmation_desk27"]["arms"]:
        for k in ("calendar_year_net", "constraints", "rule"):
            a.pop(k, None)
    o = AG.devils_advocate(J)
    atk = o.get("attacks", []) if o.get("ok") else []
    ver = [{**a, **AG.verify_attack(J, a)} for a in atk]
    res["D_n_attacks"], res["D_traceable_share"] = len(ver), (float(np.mean([v["ok"] for v in ver])) if ver else 0.0)
    res["D_attacks"] = ver
    print("DEVIL", json.dumps(ver, indent=1, default=str)[:3000])
    ledger.record(EXP, "agents", "explainer+devil", "LLM agents on top of the pipeline (verified, no capital effect)", {"model": llm.MODEL}, "n/a", "n/a", "test (holdings only)", 1, {},
                  decision="process", reason="agents read formation-month fields / result JSON only; no selection")
    E.dump(OUT / "results.json", res)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
