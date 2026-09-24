"""Text desk: turns the teammate's deterministic 8-K tables (cache/text_lane_2026-09-21: txt_v1 item counts / recency, txt_v2.1 rules, disclosure
novelty, Loughran-McDonald tone) into desk outputs. No LLM, nothing fitted. Every raw signal below has a SIGN fixed before any return was looked
at (all are adverse indicators: more of it -> lower next-month return, following the text lane's own hypotheses and the 5.02 result of
experiments/feat_8k_502_followup); the sign is never chosen from data.

Availability: a value at (permno, eom) uses only filings with filing_date <= eom (text lane guarantee G1, audited by experiments/desk_00_infra_audit).
Absent rows: counts -> 0, scores -> NaN (the interface contract). NaN => the desk has no view => score 0 (never filled with 0 BEFORE ranking).
"""

import numpy as np
import pandas as pd

from . import config as C
from .factors_desk import DeskOutput, urank


def load_blocks(verify_hash: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if verify_hash:
        sums = dict(line.split()[::-1] for line in (C.TEXT_DIR / "SHA256SUMS").read_text().splitlines() if line.strip())
        for f in ("text_features_v1.parquet", "text_features_v2.parquet"):
            assert C.file_sha256(C.TEXT_DIR / f) == sums[f], f"{f} does not match SHA256SUMS"
    v1 = pd.read_parquet(C.TEXT_DIR / "text_features_v1.parquet")
    v2 = pd.read_parquet(C.TEXT_DIR / "text_features_v2.parquet")
    for d in (v1, v2):
        d["eom"] = pd.to_datetime(d["eom"])
    return v1, v2


# name -> (source column(s), transform, sign, description)   sign -1: higher = worse
SIGNALS = {
    # -- the three the text lane headlines (primary family) --
    "novelty_max": ("txt_v2_novelty_max", "rank", -1, "share of a filing's 5-word phrases new vs the firm's own prior 24 months (max over month's filings)"),
    "novneg_max": ("txt_v2_novel_negative_max", "rank", -1, "novelty x Loughran-McDonald negative-word share (max over filings)"),
    "abrupt_exit": ("txt_v2_n_502_abrupt_exit", "flag", -1, "5.02 filing with an officer title near an exit verb"),
    # -- secondary family (exploratory; Bonferroni counts them) --
    "novelty_mean": ("txt_v2_novelty_mean", "rank", -1, "mean novelty over the month's filings"),
    "hard_abrupt": ("txt_v2_n_502_hard_abrupt", "flag", -1, "5.02 'effective immediately' / interim CEO-CFO / for cause / investigation / leave"),
    "neg_mean": ("txt_v2_neg_mean", "rank", -1, "Loughran-McDonald negative-word share (mean)"),
    "unc_mean": ("txt_v2_unc_mean", "rank", -1, "Loughran-McDonald uncertainty-word share (mean)"),
    "distress_801": ("txt_v2_n_801_distress", "flag", -1, "8.01 distress language (going concern, default, delisting notice, restatement ...)"),
    "litigation": ("txt_v2_n_801_litigation", "flag", -1, "8.01 / 1.01-2.01 litigation language (complaint, verdict, class action, SEC filed ...)"),
    "financing": ("txt_v2_n_101_financing", "flag", -1, "1.01/1.02/2.01 credit facility, notes, equity raise"),
    "novel_distress": ("txt_v2_any_novel_distress", "flag", -1, "novelty >= 0.5 AND (abrupt exit or 8.01 distress)"),
    # -- txt_v1 controls (the user's feat_8k_meta already killed the metadata versions; kept as controls) --
    "v1_any_distress": ("txt_any_distress", "flag", -1, "any 4.01/4.02/5.02/2.05/2.06 filing in the month (v1 control)"),
    "v1_n_filings": ("txt_n_filings", "rank", -1, "number of 8-Ks in the month (v1 control)"),
}
PRIMARY = ("novelty_max", "novneg_max", "abrupt_exit")


def raw_arrays(panel, v1, v2) -> dict:
    """Raw block columns joined onto the panel rows (counts absent -> 0, scores absent -> NaN)."""
    cols1 = [c for c in v1.columns if c.startswith("txt_") and v1[c].dtype.kind in "iuf"]
    cols2 = [c for c in v2.columns if c.startswith("txt_v2_") and v2[c].dtype.kind in "iuf"]
    count_like = lambda c: ("_n_" in c or c.startswith("txt_item") or c in ("txt_has_filing", "txt_any_distress", "txt_v2_any_novel_distress"))
    a1 = panel.join_wide(v1, cols1, fill_zero=[c for c in cols1 if count_like(c)])
    a2 = panel.join_wide(v2, cols2, fill_zero=[c for c in cols2 if count_like(c)])
    return {**a1, **a2}


def signal_score(panel, raw: dict, name: str, mode: str = "two_sided") -> np.ndarray:
    """Signed score in [-1, 1] (+ = long) for one registry signal.
    rank + two_sided : universe rank of the raw value among stocks WITH a value; NaN -> 0
    rank + one_sided : -max(0, rank)  (only the adverse half of the distribution is penalised; the rest 0)
    flag             : -1 where the flag count > 0, else 0 (an event either happened this month or it did not)"""
    col, kind, sign, _ = SIGNALS[name]
    x = raw[col]
    if kind == "flag":
        return sign * (np.nan_to_num(x, nan=0.0) > 0).astype(float) * panel.u
    r = urank(x, panel)
    if mode == "one_sided":
        r = np.maximum(r, 0.0)
    return sign * r


def flag_array(panel, raw: dict, names) -> np.ndarray:
    """1 where ANY of the named binary/adverse signals fired this month (universe rows only)."""
    f = np.zeros(len(panel.df), dtype=bool)
    for n in names:
        col, kind, _, _ = SIGNALS[n]
        assert kind == "flag", n
        f |= np.nan_to_num(raw[col], nan=0.0) > 0
    return f & panel.u


def decay(panel, x: np.ndarray, half_life: float, min_history: int = 1) -> np.ndarray:
    """Causal exponential decay of a monthly per-permno series: d_t = x_t + 2^(-1/hl) d_{t-1} over calendar months (rows with no filing
    contribute x=0, so the value fades). Uses only months <= t by construction (the recursion runs forward in eom)."""
    df = pd.DataFrame({"permno": panel.df["permno"].to_numpy(), "eom": panel.df["eom"].to_numpy(), "x": np.nan_to_num(np.asarray(x, dtype=float))})
    W = df.pivot(index="eom", columns="permno", values="x")
    idx = pd.date_range(W.index.min(), W.index.max(), freq="ME")
    W = W.reindex(idx).fillna(0.0)
    lam = 2.0 ** (-1.0 / half_life)
    A = W.to_numpy().copy()
    for i in range(1, len(A)):
        A[i] = A[i] + lam * A[i - 1]
    D = pd.DataFrame(A, index=idx, columns=W.columns)
    r = D.index.get_indexer(df["eom"])
    c = D.columns.get_indexer(df["permno"])
    return D.to_numpy()[r, c]


def desk(panel, raw: dict, spec: dict) -> DeskOutput:
    """Build the text desk output from a spec, e.g. {"signals": ["novelty_max", ...], "mode": "two_sided", "weights": None,
    "flags": ["novel_distress", ...], "half_life": None}. `score` = equal-weight mean of the signed signal scores (each in [-1, 1]); `flag` = adverse
    event indicator used by PM overlays; `coverage` = 1 when the stock has any text information this month."""
    names = spec["signals"]
    w = np.asarray(spec.get("weights") or [1.0] * len(names), dtype=float)
    S = np.column_stack([signal_score(panel, raw, n, spec.get("mode", "two_sided")) for n in names])
    if spec.get("half_life"):
        S = np.column_stack([-decay(panel, -S[:, j], spec["half_life"]) for j in range(S.shape[1])])
        S = np.clip(S, -1.0, 1.0)
    score = (S * w).sum(axis=1) / w.sum()
    flag = flag_array(panel, raw, spec["flags"]) if spec.get("flags") else np.zeros(len(panel.df), bool)
    cov = ((np.nan_to_num(raw["txt_n_filings"], nan=0.0) > 0) & panel.u).astype(float)
    return DeskOutput("text", score, {"components": S, "flag": flag, "coverage": cov}, dict(spec))
