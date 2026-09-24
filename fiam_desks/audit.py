"""Reproducibility / look-ahead agent (deterministic script, not an LLM). Independent checks that the inputs and the desk pipeline are causal:

  target_alignment      y at (permno, eom) equals ret_exc at (permno, next eom) on every consecutive pair
  text_v1_vs_raw        txt_v1 counts rebuilt from the raw 8-K file (permno, filing_date, items) match the teammate's table cell for cell
  text_v2_consistency   structural invariants of txt_v2 (novelty only where a filing exists, flag counts <= filings, row set)
  novelty_recompute     our own past-only novelty (5-word phrases vs the firm's previous 24 months) on a seeded permno sample vs the table, and vs a
                        deliberately future-INCLUSIVE variant (if the table matched the future-inclusive one better it would be leaking)
  desk_truncation       every desk output at date t rebuilt from data with eom <= t equals the full-data value (composite, text scores, decay)
  determinism           the same inputs give byte-identical holdings twice
"""

import re
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from . import config as C

ITEM_CODES = ["1.01", "1.02", "2.01", "2.02", "2.05", "2.06", "4.01", "4.02", "5.02", "7.01", "8.01"]


def target_alignment() -> dict:
    d = pd.read_parquet(C.CHARS_FILE, columns=["permno", "eom", "ret_exc", C.TARGET_COL])
    d["eom"] = pd.to_datetime(d["eom"])
    nxt = d[["permno", "eom", "ret_exc"]].copy()
    nxt["eom"] = nxt["eom"] - pd.offsets.MonthEnd(1)
    m = d.merge(nxt, on=["permno", "eom"], how="inner", suffixes=("", "_next"))
    m = m[m[C.TARGET_COL].notna() & m["ret_exc_next"].notna()]
    bad = (np.abs(m[C.TARGET_COL] - m["ret_exc_next"]) > 1e-9).sum()
    return {"pairs_checked": int(len(m)), "mismatches": int(bad), "passed": bool(bad == 0 and len(m) > 100_000)}


def _raw_month_counts() -> pd.DataFrame:
    t = pq.read_table(C.FILINGS_FILE, columns=["permno", "filing_date", "items", "is_amendment"]).to_pandas()
    t["eom"] = pd.to_datetime(t["filing_date"]) + pd.offsets.MonthEnd(0)
    g = t.groupby(["permno", "eom"])
    out = pd.DataFrame({"txt_n_filings": g.size(), "txt_n_amendments": g["is_amendment"].sum()})
    codes = t[["permno", "eom", "items"]].copy()
    codes["items"] = codes["items"].map(lambda x: sorted(set(x)) if x is not None else [])
    ex = codes.explode("items").dropna(subset=["items"])
    for c in ITEM_CODES:
        s = ex[ex["items"] == c].groupby(["permno", "eom"]).size()
        out["txt_item_" + c.replace(".", "_")] = s
    out = out.fillna(0).astype(int).reset_index()
    return out


def text_v1_vs_raw(v1: pd.DataFrame) -> dict:
    raw = _raw_month_counts()
    cols = [c for c in raw.columns if c.startswith("txt_")]
    m = v1[["permno", "eom"] + cols].merge(raw, on=["permno", "eom"], how="outer", suffixes=("", "_raw"), indicator=True)
    only_raw = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    diffs = {c: int((both[c].astype(int) != both[c + "_raw"].astype(int)).sum()) for c in cols}
    # months where the raw file has filings must all be present in v1
    return {"rows_v1": int(len(v1)), "rows_raw_permno_months": int(len(raw)), "raw_months_missing_from_v1": only_raw,
            "cell_mismatches_by_column": diffs, "passed": bool(only_raw == 0 and sum(diffs.values()) == 0)}


def text_v2_consistency(v1: pd.DataFrame, v2: pd.DataFrame) -> dict:
    m = v1[["permno", "eom", "txt_n_filings", "txt_has_filing"]].merge(v2, on=["permno", "eom"], how="outer", indicator=True)
    checks = {"same_row_set": bool((m["_merge"] == "both").all())}
    nf = m["txt_n_filings"].fillna(0)
    checks["novelty_defined_only_with_filing"] = bool(m.loc[nf == 0, "txt_v2_novelty_max"].isna().all())
    checks["n_with_novelty_le_filings"] = bool((m["txt_v2_n_with_novelty"] <= nf).all())
    for c in [c for c in v2.columns if c.startswith("txt_v2_n_") and c != "txt_v2_n_with_novelty" and c != "txt_v2_n_novel_distress"]:
        checks[f"{c}_le_filings"] = bool((m[c] <= nf).all())
    checks["unique_key"] = bool(not v2.duplicated(["permno", "eom"]).any())
    checks["eom_is_month_end"] = bool((v2["eom"] == v2["eom"] + pd.offsets.MonthEnd(0)).all())
    checks["novelty_in_0_1"] = bool(v2["txt_v2_novelty_max"].dropna().between(0, 1).all())
    return {"checks": checks, "passed": bool(all(checks.values()))}


_TOK = re.compile(r"[a-z0-9']+")
_ITEM = re.compile(r"item\s+\d\.\d\d", re.I)


def _body(text: str) -> str:
    m = _ITEM.search(text)
    s = text[m.start():] if m else text
    k = s.lower().rfind("signature")
    return s[:k] if k > 200 else s


def _phrases(text: str, n: int = 5) -> set:
    w = _TOK.findall(_body(text).lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _novelty_series(fil: pd.DataFrame, future_inclusive: bool) -> pd.DataFrame:
    """Novelty per filing. Past-only: phrases seen in the same firm's filings with STRICTLY earlier filing_date within 24 months (same-day filings
    do not see each other). future_inclusive=True additionally counts filings up to 24 months AFTER (a leaking control)."""
    out = []
    for p, g in fil.groupby("permno"):
        g = g.sort_values("filing_date")
        docs = [(d, _phrases(t)) for d, t in zip(g["filing_date"], g["text"])]
        for i, (d, ph) in enumerate(docs):
            lo, hi = d - pd.DateOffset(months=24), d + pd.DateOffset(months=24)
            seen, n_prior = set(), 0
            for j, (d2, ph2) in enumerate(docs):
                if d2 == d and j != i and not future_inclusive:
                    continue
                if (lo <= d2 < d) or (future_inclusive and d2 > d and d2 <= hi):
                    seen |= ph2
                    n_prior += 1
            nov = np.nan if (n_prior == 0 or len(ph) < 20) else len(ph - seen) / len(ph)
            out.append((p, d, nov))
    return pd.DataFrame(out, columns=["permno", "filing_date", "nov"])


def novelty_recompute(v2: pd.DataFrame, n_permnos: int = 60, seed: int = 0) -> dict:
    cnt = pq.read_table(C.FILINGS_FILE, columns=["permno"]).to_pandas()["permno"].value_counts()
    pool = cnt[(cnt >= 30) & (cnt <= 250)].index.to_numpy()
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(pool, size=min(n_permnos, len(pool)), replace=False).tolist())
    fil = pq.read_table(C.FILINGS_FILE, columns=["permno", "filing_date", "text"], filters=[("permno", "in", pick)]).to_pandas()
    fil["filing_date"] = pd.to_datetime(fil["filing_date"])
    res = {}
    tab = {}
    for label, fut in (("past_only", False), ("future_inclusive", True)):
        nv = _novelty_series(fil, fut)
        nv["eom"] = nv["filing_date"] + pd.offsets.MonthEnd(0)
        mx = nv.groupby(["permno", "eom"])["nov"].max().rename("mine").reset_index()
        m = mx.merge(v2[["permno", "eom", "txt_v2_novelty_max"]], on=["permno", "eom"], how="inner").dropna()
        tab[label] = m
        res[label] = {"n_permno_months": int(len(m)), "spearman_vs_table": float(spearmanr(m["mine"], m["txt_v2_novelty_max"])[0]),
                      "median_abs_diff": float((m["mine"] - m["txt_v2_novelty_max"]).abs().median()),
                      "share_within_0.05": float(((m["mine"] - m["txt_v2_novelty_max"]).abs() <= 0.05).mean())}
    res["table_matches_past_only_better"] = bool(res["past_only"]["spearman_vs_table"] > res["future_inclusive"]["spearman_vs_table"])
    res["passed"] = bool(res["table_matches_past_only_better"] and res["past_only"]["spearman_vs_table"] >= 0.85)
    res["sample_permnos"] = len(pick)
    return res


def desk_truncation(panel, builders: dict, n_dates: int = 3, seed: int = 0, tol: float = 1e-9) -> dict:
    """builders: {name: fn(panel) -> array aligned to panel rows}. Rebuild from rows with eom <= t and compare at t (all universe rows)."""
    rng = np.random.default_rng(seed)
    eoms = pd.DatetimeIndex(sorted(pd.to_datetime(panel.df["eom"]).unique()))
    cand = eoms[(eoms >= pd.Timestamp("2018-01-31")) & (eoms <= pd.Timestamp("2026-07-31"))]
    dates = [cand[i] for i in sorted(rng.choice(len(cand), n_dates, replace=False))]
    full = {k: np.asarray(fn(panel), dtype=float) for k, fn in builders.items()}
    worst, n_cmp = {}, 0
    for t in dates:
        sub = panel.truncated(t)
        at_full = (panel.df["eom"] == t).to_numpy() & panel.u
        at_sub = (sub.df["eom"] == t).to_numpy() & sub.u
        for k, fn in builders.items():
            a, b = full[k][at_full], np.asarray(fn(sub), dtype=float)[at_sub]
            assert len(a) == len(b), "row set differs"
            d = np.abs(np.nan_to_num(a, nan=1e9) - np.nan_to_num(b, nan=1e9))
            worst[k] = max(worst.get(k, 0.0), float(d.max()))
            n_cmp += len(a)
    return {"dates": [str(d.date()) for d in dates], "n_values_compared": int(n_cmp), "max_abs_diff": worst,
            "passed": bool(n_cmp > 0 and all(v <= tol for v in worst.values()))}


def frame_hash(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].round(12)
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()).hexdigest()[:16]
