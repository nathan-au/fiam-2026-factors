"""Deterministic PM, decision layer: turns desk outputs into the LP inputs (score `pred`, `veto_long`, `veto_short`, `w_mult`). No fitting, no randomness,
no LLM: every mode is a fixed rule of desk outputs available at the formation month, so a decision can always be re-derived by hand from the rationale table.

Modes (all pre-declared in docs/DESKS.md sec 4; each is a separate experiment):
  factors_only   pred = factors score                                                  (baseline; the frozen composite)
  blend          pred = (7 * factors + w * text) / (7 + w)                              (text as an 8th equal-weight group)
  veto_long      longs barred where the text ADVERSE flag fired                         (risk overlay; pred unchanged)
  tilt           pred = factors - lam * flag                                            (adverse flag pushes a name toward the short side)
  dial           w_mult = 1 + kappa * agree, agree = sign(factors) * sign(text) in {-1, 0, 1} (desks agreeing -> larger cap, disagreeing -> smaller)
  judge          a short must be 'convicted': factors agreement >= a_min of 7 groups OR text flag; acquitted shorts are barred (mirror rule for longs off by default)
  agree_dial     w_mult = lo + (hi - lo) * factor_group_agreement  (desk-internal confidence, no text)
Optional add-ons applied on top of any mode: si_cap (shorts only where the FINRA short-interest ratio <= 10%, adopted in experiments/feat_si_followup).
"""

import numpy as np

from . import config as C
from .factors_desk import DeskOutput


def decide(factors: DeskOutput, text: DeskOutput | None, mode: str = "factors_only", **kw) -> dict:
    n = len(factors.score)
    pred = factors.score.copy()
    veto_long, veto_short, w_mult = np.zeros(n, bool), np.zeros(n, bool), np.ones(n)
    log = {"mode": mode, **kw}
    needs_text = mode in ("blend", "veto_long", "tilt", "dial") or (mode == "judge" and kw.get("use_text", True))
    if needs_text and text is None:
        raise ValueError(f"mode {mode} needs a text desk output")
    if mode == "factors_only":
        pass
    elif mode == "blend":
        w = kw.get("w", 1.0)
        pred = (7.0 * factors.score + w * text.score) / (7.0 + w)
    elif mode == "veto_long":
        veto_long = text.extras["flag"].copy()
    elif mode == "tilt":
        pred = factors.score - kw.get("lam", 0.25) * text.extras["flag"].astype(float)
    elif mode == "dial":
        agree = np.sign(factors.score) * np.sign(text.score)
        w_mult = 1.0 + kw.get("kappa", 0.5) * agree
    elif mode == "judge":
        agr = np.nan_to_num(factors.extras["agreement"], nan=0.0)
        convicted = (agr * 7 >= kw.get("a_min", 5)) | (text.extras["flag"] if kw.get("use_text", True) else np.zeros(n, bool))
        veto_short = (factors.score < 0) & ~convicted
    elif mode == "agree_dial":
        agr = np.nan_to_num(factors.extras["agreement"], nan=0.5)
        w_mult = kw.get("lo", 0.5) + (kw.get("hi", 1.5) - kw.get("lo", 0.5)) * agr
    else:
        raise ValueError(mode)
    return {"pred": pred, "veto_long": veto_long, "veto_short": veto_short, "w_mult": w_mult, "log": log}


def lp_frame(panel, decision: dict, period: str, extras: dict | None = None):
    """Predictions frame for the LP: desk decision columns (+ optional `sir` for the short-interest cap)."""
    ex = {"veto_long": decision["veto_long"], "veto_short": decision["veto_short"], "w_mult": decision["w_mult"]}
    ex.update(extras or {})
    return panel.preds_frame(decision["pred"], period, ex)


def cfg_with_si_cap(cfg: dict | None = None, cap: float = C.SI_CAP) -> dict:
    return {**(cfg or C.LP_BASE), "short_cap_val": cap}
