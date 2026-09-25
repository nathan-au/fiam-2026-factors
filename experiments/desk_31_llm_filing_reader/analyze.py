"""
desk_31 step 2 - does a local LLM reading masked 8-Ks add anything the rules / dictionaries do not? Hypotheses fixed BEFORE reading any output
(this docstring was written while run_reader.py was running):
  H1 comprehension     LLM direction correlates positively with the FILING month's own return (the market's reaction to the same news): t >= 3
  H2 incremental       H1 holds controlling for Loughran-McDonald negative tone and novelty (the rule-based text): LLM coefficient t >= 2
  H3 alpha             LLM direction predicts the NEXT month's return rank, controlling for the B1 factor score, tone and novelty: t >= 2
  H4 risk              LLM surprise (0/1/2) predicts next-month abs residual return beyond novelty and ivol: t >= 2
  H5 look-ahead        attacker probe: share of masked filings whose company (top-1 guess vs any filer alias) or year is identified
Regression unit = filing; y variables are within-month universe percentile ranks in [-1, 1]; standard errors clustered by month. DEV only.
If H3 fails, the LLM is not taken to TEST (no second look spent on it).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E  # noqa: E402
from fiam_research import core, ledger, llm  # noqa: E402

OUT = HERE / "output"
EXP = "desk_31_llm_filing_reader"


def main():
    s = pd.read_parquet(OUT / "sample.parquet")
    rd = pd.read_parquet(OUT / "reads.parquet")
    d = s.merge(rd, on="document_id")
    res = {"n_sample": len(s), "n_read": len(rd), "json_ok_share": float(rd["ok"].mean()), "median_seconds": float(rd["_ms"].median() / 1000)}
    d = d[d["ok"] == True].copy()  # noqa: E712
    d["direction"] = pd.to_numeric(d["direction"], errors="coerce").clip(-2, 2)
    d["surprise"] = pd.to_numeric(d["surprise"], errors="coerce").clip(0, 2)
    res["direction_dist"] = d["direction"].value_counts().sort_index().to_dict()
    res["event_type_dist"] = d["event_type"].value_counts().head(12).to_dict()
    ctx = core.research_ctx(text=True)
    P, raw = ctx.P, ctx.raw
    df = P.df[["permno", "eom"]].copy()
    rk = lambda x: pd.Series(np.where(P.u, x, np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy() * 2 - 1
    df["y_next"] = rk(P.y)
    df["r_same"] = rk(P.df["raw_ret_1_0"].to_numpy())
    dev_ = pd.Series(P.y).groupby(P.df["eom"].to_numpy()).transform("mean").to_numpy()
    df["absres_next"] = rk(np.abs(P.y - dev_))
    df["b1"] = core.b1_score(ctx)
    df["neg"] = rk(raw["txt_v2_neg_mean"])
    df["nov"] = rk(raw["txt_v2_novelty_max"])
    df["ivol"] = rk(P.df["x_vol"].to_numpy())
    d = d.merge(df, on=["permno", "eom"], how="left").dropna(subset=["y_next", "r_same"])
    d[["neg", "nov"]] = d[["neg", "nov"]].fillna(0.0)
    d["m"] = d["eom"].astype(str)
    fit = lambda f: smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d["m"]})
    out = {}
    for name, f, var in (("H1", "r_same ~ direction", "direction"), ("H2", "r_same ~ direction + neg + nov", "direction"),
                         ("H2_dictionary_only", "r_same ~ neg + nov", "neg"),
                         ("H3", "y_next ~ direction + b1 + neg + nov", "direction"), ("H3_raw", "y_next ~ direction", "direction"),
                         ("H4", "absres_next ~ surprise + nov + ivol", "surprise")):
        r = fit(f)
        out[name] = {"coef": float(r.params[var]), "t": float(r.tvalues[var]), "n": int(r.nobs), "r2": float(r.rsquared),
                     "all": {k: [round(float(r.params[k]), 4), round(float(r.tvalues[k]), 2)] for k in r.params.index}}
        print(name, f, "->", out[name]["all"], flush=True)
    res["tests"] = out
    res["verdict"] = {"H1_comprehension": out["H1"]["t"] >= 3, "H2_incremental": out["H2"]["t"] >= 2, "H3_alpha": out["H3"]["t"] >= 2, "H4_risk": out["H4"]["t"] >= 2}
    # H5 attacker
    a = pd.read_parquet(OUT / "attack.parquet") if (OUT / "attack.parquet").exists() else None
    if a is not None:
        import pyarrow.parquet as pq
        from fiam_desks import config as C
        NAMECOLS = ["company_name", "content_company_name", "historical_crsp_company_name", "provider_company_name", "compustat_header_company_name", "ticker", "provider_ticker"]
        al = pq.read_table(C.FILINGS_FILE, columns=["document_id"] + NAMECOLS).to_pandas().set_index("document_id")
        def hit(r):
            g = str(r["guess_company"] or "").lower().strip()
            if len(g) < 2:
                return False
            x = al.loc[r["document_id"]]
            cores = [c.lower() for c in llm.alias_cores([x[c] for c in NAMECOLS[:5]], [x["ticker"], x["provider_ticker"]])]
            return any(c == g or (len(c) >= 4 and (c in g or g in c)) for c in cores)
        a["company_hit"] = a.apply(hit, axis=1)
        a["year_hit"] = pd.to_numeric(a["guess_year"], errors="coerce") == a["true_year"]
        res["H5_attacker"] = {"n": int(len(a)), "company_top1_hit": float(a["company_hit"].mean()), "year_exact": float(a["year_hit"].mean()),
                              "year_base_rate_if_uniform_2015_2020": 1 / 6}
        a.to_csv(OUT / "attack_scored.csv", index=False)
        print("H5", res["H5_attacker"])
    d.to_parquet(OUT / "analysis_frame.parquet")
    for h in ("H1", "H2", "H3", "H4"):
        ledger.record(EXP, "llm_reader", h, "local LLM on masked 8-Ks (qwen3.5 9B q4)", {"model": llm.MODEL, "prompt": llm.PROMPT_VERSION}, "dev", "none", "dev", 4,
                      {"ic": out[h]["coef"], "ic_t": out[h]["t"]}, decision="pass" if res["verdict"].get({"H1": "H1_comprehension", "H2": "H2_incremental", "H3": "H3_alpha", "H4": "H4_risk"}[h]) else "fail")
    E.dump(OUT / "results.json", res)
    print(res["verdict"])


if __name__ == "__main__":
    main()
