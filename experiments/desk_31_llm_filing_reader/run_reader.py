"""
desk_31 step 1 - build the stratified DEV sample of news-bearing 8-Ks from universe stocks and read each with the local LLM (masked).
Sample (fixed before any LLM output was seen): filing months 2015-01..2020-11 (so the next month's return is a DEV target), filer permno in the research
universe at the filing month's end, not an amendment, at least one of items 1.01 1.02 2.01 2.05 2.06 4.01 4.02 5.02 7.01 8.01; 25 filings per month drawn
with numpy seed 0 (fewer if the month has fewer). Then an attacker probe on 150 of them (seed 1).
Resumable: every call is cached (fiam_research/llm.py). Run: .venv/bin/python experiments/desk_31_llm_filing_reader/run_reader.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C  # noqa: E402
from fiam_research import llm  # noqa: E402
from fiam_research.panel_ext import PanelX  # noqa: E402

OUT = HERE / "output"
NEWS = {"1.01", "1.02", "2.01", "2.05", "2.06", "4.01", "4.02", "5.02", "7.01", "8.01"}


def sample():
    f = OUT / "sample.parquet"
    if f.exists():
        return pd.read_parquet(f)
    P = PanelX(0.0)
    uni = P.df.loc[P.u, ["permno", "eom"]].assign(inu=True)
    t = pq.read_table(C.FILINGS_FILE, columns=["document_id", "permno", "filing_date", "items", "is_amendment", "company_name", "word_count"]).to_pandas()
    t["fd"] = pd.to_datetime(t["filing_date"])
    t = t[(t.fd >= "2015-01-01") & (t.fd < "2020-12-01") & (~t.is_amendment.fillna(False))]
    t["eom"] = t.fd + pd.offsets.MonthEnd(0)
    t = t[t["items"].map(lambda a: len(NEWS.intersection(list(a))) > 0)]
    t = t.merge(uni, on=["permno", "eom"], how="inner")
    rng = np.random.default_rng(0)
    parts = [g.iloc[rng.permutation(len(g))[:25]] for _, g in t.sort_values("document_id").groupby("eom")]
    s = pd.concat(parts).reset_index(drop=True)
    s["items"] = s["items"].map(lambda a: ",".join(a))
    s.to_parquet(f)
    return s


def main():
    s = sample()
    print("sample", len(s), "filings,", s.eom.nunique(), "months", flush=True)
    ids = set(s.document_id)
    NAMECOLS = ["company_name", "content_company_name", "historical_crsp_company_name", "provider_company_name", "compustat_header_company_name", "ticker", "provider_ticker"]
    al = pq.read_table(C.FILINGS_FILE, columns=["document_id"] + NAMECOLS).to_pandas()
    al = al[al.document_id.isin(set(s.document_id))].set_index("document_id")
    ALIAS = {d: llm.alias_cores([r[c] for c in NAMECOLS[:5]], [r["ticker"], r["provider_ticker"]]) for d, r in al.iterrows()}
    txt = pq.read_table(C.FILINGS_FILE, columns=["document_id", "text"], filters=[("filing_date", ">=", pd.Timestamp("2015-01-01").date()),
                                                                                     ("filing_date", "<", pd.Timestamp("2020-12-01").date())]).to_pandas()
    txt = txt[txt.document_id.isin(ids)].set_index("document_id")["text"]
    gaz = sorted(set(pq.read_table(C.FILINGS_FILE, columns=["company_name"]).to_pandas().company_name.dropna()))
    rows, t0 = [], time.time()
    for i, r in (s.iloc[::-1].iterrows() if "--reverse" in sys.argv else s.iterrows()):
        m = llm.mask(llm.body(txt[r.document_id]), r.company_name, r.fd.date(), gaz, r.document_id, ALIAS.get(r.document_id, ()))
        out = llm.cached("read", m)
        rows.append({"document_id": r.document_id, "masked_chars": len(m), **{k: out.get(k) for k in ("ok", "event_type", "is_routine", "direction", "surprise", "evidence", "_ms")}})
        if "--reverse" in sys.argv:
            continue
        if i % 50 == 0:
            print(f"{i}/{len(s)} {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_parquet(OUT / "reads.parquet")
    if "--reverse" in sys.argv:
        return
    pd.DataFrame(rows).to_parquet(OUT / "reads.parquet")
    rng = np.random.default_rng(1)
    att = []
    for i in rng.choice(len(s), 150, replace=False):
        r = s.iloc[i]
        m = llm.mask(llm.body(txt[r.document_id]), r.company_name, r.fd.date(), gaz, r.document_id, ALIAS.get(r.document_id, ()))
        out = llm.cached("attack", m)
        att.append({"document_id": r.document_id, "true_company": r.company_name, "true_year": r.fd.year, "guess_company": out.get("company"), "guess_year": out.get("year"), "ok": out.get("ok")})
    pd.DataFrame(att).to_parquet(OUT / "attack.parquet")
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
