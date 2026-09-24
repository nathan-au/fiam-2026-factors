"""The assembled system: Factors desk + Text desk -> Deterministic PM -> LP -> FIAM-format holdings, returns and a per-trade rationale table.

DEFAULT CONFIG (settled by experiments/desk_00..desk_11; docs/DESKS.md sec 5):
  factors desk   frozen 7-group composite + days-to-cover as an 8th group (adopted-conditional, experiments/feat_short_interest)
  PM             lc_t10 LP + short cap on FINRA short-interest ratio <= 10% (adopted, experiments/feat_si_followup)
  text desk      ADVISORY: computed, audited and written to the rationale table; it does not change positions (no text PM mode met the pre-registered adoption rule)
Everything is deterministic. `build` never reads a return of a month later than the one it decides (the LP uses realised returns only to drift last month's book, as in the frozen harness).
"""

import numpy as np
import pandas as pd

from . import config as C
from . import factors_desk as FD
from . import lp as L
from . import pm as PM
from . import si as SI
from . import text_desk as TD

TEXT_SPEC = {"signals": ["novelty_max", "novneg_max", "abrupt_exit"], "mode": "two_sided", "flags": ["novel_distress", "hard_abrupt", "litigation"]}
DEFAULT = {"factors": "composite+dtc", "si_cap": True, "text_mode": "advisory", "text_params": {}, "lp": "lc_t10"}


def desks(panel, need_si=True):
    """All desk outputs on every panel row (no period logic here)."""
    comp = FD.composite(panel)
    S = SI.features(panel) if need_si else {"sir": np.full(len(panel.df), np.nan), "dtc": np.full(len(panel.df), np.nan)}
    v1, v2 = TD.load_blocks()
    raw = TD.raw_arrays(panel, v1, v2)
    txt = TD.desk(panel, raw, TEXT_SPEC)
    return comp, S, raw, txt


def factors_output(panel, comp, S, kind: str):
    if kind == "composite":
        return comp
    if kind == "composite+dtc":
        return FD.add_groups(comp, {"dtc": -FD.urank(S["dtc"], panel)}, "factors+dtc")
    raise ValueError(kind)


def build(panel, period: str, cfg: dict | None = None, parts=None):
    cfg = {**DEFAULT, **(cfg or {})}
    comp, S, raw, txt = parts or desks(panel)
    fac = factors_output(panel, comp, S, cfg["factors"])
    mode = "factors_only" if cfg["text_mode"] == "advisory" else cfg["text_mode"]
    dec = PM.decide(fac, txt if mode != "factors_only" else None, mode, **cfg["text_params"])
    lp_cfg = PM.cfg_with_si_cap() if cfg["si_cap"] else C.LP_BASE
    preds = PM.lp_frame(panel, dec, period, {"sir": S["sir"]})
    res = L.evaluate_book(preds, lp_cfg, keep_frames=True)
    return {"cfg": cfg, "result": res, "decision": dec, "factors": fac, "text": txt, "comp": comp, "si": S, "raw": raw, "lp_cfg": lp_cfg}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------------------------------------------------------
def holdings_fiam(out: dict) -> pd.DataFrame:
    """FIAM holdings CSV: Date (first day of the holding month), PERMNO, TICKER, COMPANY NAME, WEIGHT (% of NAV, + long / - short)."""
    h = out["result"]["holdings"]
    d = pd.DataFrame({"Date": (pd.to_datetime(h["target_month"]) - pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m-%d"), "PERMNO": h["permno"].astype(int),
                      "TICKER": h["ticker"], "COMPANY NAME": h["company_name"], "WEIGHT": (100.0 * h["weight"]).round(6)})
    return d.sort_values(["Date", "WEIGHT"], ascending=[True, False]).reset_index(drop=True)


def returns_csv(out: dict) -> pd.DataFrame:
    fr = out["result"]["frame"]
    rf = fr["rf_monthly"]
    bench = rf + 0.04 / 12.0
    net_total = rf + fr["net_port_excess_ret"]
    return pd.DataFrame({"Date": (pd.to_datetime(fr["target_month"]) - pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m-%d"),
                         "portfolio_return": net_total.round(8), "portfolio_return_gross": (rf + fr["port_excess_ret"]).round(8), "benchmark_return": bench.round(8),
                         "sp500_return": fr["sp500_ret"].round(8), "active_return": (net_total - bench).round(8), "gross_exposure": fr["gross_exposure"].round(6),
                         "net_exposure": fr["net_exposure"].round(6), "n_positions": fr["n_positions"]})


def rationale(panel, out: dict) -> pd.DataFrame:
    """One row per position-month: what every desk said, in plain fields, plus a one-line reason assembled by fixed templates (no free text generation)."""
    h = out["result"]["holdings"][["permno", "eom", "target_month", "ticker", "weight", "sector"]].copy()
    df = panel.df
    key = df[["permno", "eom"]].copy()
    G = out["comp"].extras["groups"]
    for g in G.columns:
        key["g_" + g] = G[g].to_numpy()
    key["factors_score"] = out["factors"].score
    key["dtc_raw"], key["si_ratio"] = out["si"]["dtc"], out["si"]["sir"]
    raw = out["raw"]
    for col, nm in (("txt_v2_novelty_max", "novelty"), ("txt_v2_novel_negative_max", "novel_x_neg"), ("txt_v2_neg_mean", "neg_share"), ("txt_n_filings", "n_filings")):
        key[nm] = raw[col]
    for nm in ("novel_distress", "hard_abrupt", "litigation"):
        key["flag_" + nm] = TD.flag_array(panel, raw, [nm])
    key["text_score"] = out["text"].score
    m = h.merge(key, on=["permno", "eom"], how="left")
    gcols = ["g_" + g for g in G.columns]

    def reason(r):
        side = "LONG" if r["weight"] > 0 else "SHORT"
        gs = r[gcols].astype(float)
        ordered = gs.sort_values(ascending=(r["weight"] < 0))  # for a long: strongest positive first; for a short: most negative first
        top = ", ".join(f"{k[2:]} {v:+.2f}" for k, v in ordered.head(2).items())
        drag = ", ".join(f"{k[2:]} {v:+.2f}" for k, v in ordered.tail(1).items())
        parts = [f"{side} {100 * abs(r['weight']):.2f}% NAV", f"factors {r['factors_score']:+.2f} (for: {top}; against: {drag})"]
        if np.isfinite(r["si_ratio"]):
            parts.append(f"short interest {100 * r['si_ratio']:.1f}% of shares, days-to-cover {r['dtc_raw']:.1f}")
        fl = [n for n in ("novel_distress", "hard_abrupt", "litigation") if r["flag_" + n]]
        if fl:
            parts.append("text flags (advisory, not traded): " + ", ".join(fl))
        elif r["n_filings"] and r["n_filings"] > 0:
            parts.append(f"8-K month: novelty {r['novelty']:.2f}" if np.isfinite(r["novelty"]) else "8-K month")
        return "; ".join(parts)

    m["reason"] = m.apply(reason, axis=1)
    m["Date"] = (pd.to_datetime(m["target_month"]) - pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m-%d")
    return m.drop(columns=["eom", "target_month"]).sort_values(["Date", "weight"], ascending=[True, False]).reset_index(drop=True)
