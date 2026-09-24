"""
FIAM 2026 - Rare severe 8-K events as a long-side veto, DEV period only (desk_04_text_rare_event_veto). Test window not read.

FIAM section 4 calls auditor changes / non-reliance (4.01, 4.02) "two of the strongest distress signals in the corpus". They are too rare to work as a ranking
group (desk_01 / feat_8k_meta), but a veto only needs them to select tail losers among names the composite wants to be LONG. Pre-registered flag:
severe = any of items 4.02, 4.01, 2.06 (impairment), 2.05 (exit/restructuring costs) in month t, plus a 3-month persistence version (any in t-2..t).
Question: do flagged names in the universe have a fatter next-month left tail / lower mean, especially among the composite's top-quintile (long candidates)?
Statistics: month-clustered flagged-minus-unflagged tail rate (y <= -15%) and mean return, t over months (months with >= 3 flagged).
Run:  .venv/bin/python experiments/desk_04_text_rare_event_veto/desk_04_text_rare_event_veto.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
PERIOD = "dev"


def stat(P, flag, sel, name, thr=-0.15, min_n=3):
    m = P.mask(PERIOD) & sel
    d = pd.DataFrame({"m": P.months[m], "f": flag[m], "tail": (P.y[m] <= thr).astype(float), "y": P.y[m]})
    g = d.groupby(["m", "f"]).agg(tail=("tail", "mean"), y=("y", "mean"), n=("y", "size")).unstack("f")
    if True not in g["n"].columns or False not in g["n"].columns:
        return {"set": name, "months_ok": 0}
    ok = g["n"][True].fillna(0) >= min_n
    dt, dy = (g["tail"][True] - g["tail"][False])[ok], (g["y"][True] - g["y"][False])[ok]
    return {"set": name, "months_ok": int(ok.sum()), "flagged_per_month": float(g["n"][True][ok].mean()), "tail_flagged": float(g["tail"][True][ok].mean()),
            "tail_unflagged": float(g["tail"][False][ok].mean()), "tail_diff": float(dt.mean()), "tail_diff_t": E.tstat(dt), "mean_diff_pct": float(100 * dy.mean()), "mean_diff_t": E.tstat(dy)}


def main():
    P = Panel()
    comp = FD.composite(P)
    v1, v2 = TD.load_blocks()
    raw = TD.raw_arrays(P, v1, v2)
    sev = sum(np.nan_to_num(raw[c]) for c in ("txt_item_4_02", "txt_item_4_01", "txt_item_2_06", "txt_item_2_05"))
    f1 = (sev > 0) & P.u
    # 3-month persistence: any severe filing in t-2..t (causal rolling max over calendar months)
    df = pd.DataFrame({"permno": P.df["permno"].to_numpy(), "eom": P.df["eom"].to_numpy(), "x": (sev > 0).astype(float)})
    W = df.pivot(index="eom", columns="permno", values="x")
    W = W.reindex(pd.date_range(W.index.min(), W.index.max(), freq="ME")).fillna(0.0).rolling(3, min_periods=1).max()
    f3 = (W.to_numpy()[W.index.get_indexer(df["eom"]), W.columns.get_indexer(df["permno"])] > 0) & P.u
    q = pd.Series(comp.score).where(P.u).groupby(P.df["eom"]).transform(lambda s: s.rank(pct=True)).to_numpy()
    long_cand, short_cand = (q >= 0.8) & P.u, (q <= 0.2) & P.u
    rows = []
    for lab, flag in (("severe_t", f1), ("severe_3m", f3)):
        for sel_name, sel in (("universe", P.u), ("composite top quintile (long candidates)", long_cand), ("composite bottom quintile (short candidates)", short_cand)):
            r = stat(P, flag, sel, f"{lab} | {sel_name}")
            rows.append(r)
    df_ = pd.DataFrame(rows)
    df_.to_csv(OUT / "rare_event.csv", index=False)
    print(df_.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    E.dump(OUT / "results.json", rows)


if __name__ == "__main__":
    main()
