"""FINRA short-interest features aligned to panel rows (adopted, conditional, in experiments/feat_short_interest + feat_si_followup): the short-interest ratio `sir` = FINRA short
position / shares outstanding, and `dtc` = FINRA days-to-cover, from the mid-month settlement file of the characteristic month (published before month-end, so causal). Files exist
from 2020-06 only; earlier months are NaN = "no information" (the LP short cap then does not bind; the dtc block is 0). Uses the local cache only (no network)."""

import numpy as np
import pandas as pd

from . import config as C

FINRA_DIR = C.CACHE / "finra_short_interest"


def _mid_month(month_end):
    d = pd.Timestamp(month_end.year, month_end.month, 15)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _file_for(month_end):
    d = _mid_month(month_end)
    for back in range(5):
        f = FINRA_DIR / f"shrt{(d - pd.tseries.offsets.BDay(back)):%Y%m%d}.csv"
        if f.exists():
            return f
    return None


def features(panel) -> dict:
    df = panel.df
    sir, dtc = np.full(len(df), np.nan), np.full(len(df), np.nan)
    months = [t for t in pd.date_range("2020-06-30", "2026-07-31", freq="ME")]
    used = 0
    eom = df["eom"].to_numpy()
    for t in months:
        f = _file_for(t)
        if f is None:
            continue
        rows = np.flatnonzero(eom == np.datetime64(t))
        if len(rows) == 0:
            continue
        s = pd.read_csv(f, sep="|", usecols=["symbolCode", "currentShortPositionQuantity", "daysToCoverQuantity"], dtype={"symbolCode": str}).drop_duplicates("symbolCode")
        sub = df.iloc[rows][["ticker", "shares"]].reset_index()
        sub = sub.drop_duplicates("ticker").merge(s, left_on="ticker", right_on="symbolCode", how="inner")
        sub = sub[sub["shares"] > 0]
        sir[sub["index"].to_numpy()] = (sub["currentShortPositionQuantity"] / (sub["shares"] * 1e6)).to_numpy()
        dtc[sub["index"].to_numpy()] = sub["daysToCoverQuantity"].to_numpy()
        used += 1
    return {"sir": sir, "dtc": dtc, "files_used": used}
