"""FINRA short interest aligned to panel rows, extended back to 2018-01 (the first month FINRA's public CDN serves; earlier files return 403).

Identical logic to fiam_desks/si.py (mid-month settlement file of the characteristic month, published ~8 business days later and therefore before
month-end; join on the period ticker; sir = short position / shares outstanding; dtc = FINRA days-to-cover) — only the month range differs.
The 2018-01..2020-05 files were downloaded on 2026-09-24 into cache/finra_short_interest/ (experiments/desk_14_si_history_extension).
"""

import numpy as np
import pandas as pd

from fiam_desks import si as SI

START = "2018-01-31"


def features(panel, start: str = START, end: str = "2026-07-31") -> dict:
    df = panel.df
    n = len(df)
    sir, dtc, adv = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    chg = np.full(n, np.nan)
    eom = df["eom"].to_numpy()
    used, match = 0, []
    for t in pd.date_range(start, end, freq="ME"):
        f = SI._file_for(t)
        if f is None:
            continue
        rows = np.flatnonzero(eom == np.datetime64(t))
        if len(rows) == 0:
            continue
        s = pd.read_csv(f, sep="|", usecols=["symbolCode", "currentShortPositionQuantity", "previousShortPositionQuantity", "daysToCoverQuantity",
                                            "averageDailyVolumeQuantity"], dtype={"symbolCode": str}, quoting=3).drop_duplicates("symbolCode")
        sub = df.iloc[rows][["ticker", "shares"]].reset_index()
        sub = sub.drop_duplicates("ticker").merge(s, left_on="ticker", right_on="symbolCode", how="inner")
        sub = sub[sub["shares"] > 0]
        i = sub["index"].to_numpy()
        sir[i] = (sub["currentShortPositionQuantity"] / (sub["shares"] * 1e6)).to_numpy()
        dtc[i] = sub["daysToCoverQuantity"].to_numpy()
        adv[i] = sub["averageDailyVolumeQuantity"].to_numpy()
        prev = sub["previousShortPositionQuantity"].to_numpy(dtype=float)
        chg[i] = np.where(prev > 0, sub["currentShortPositionQuantity"].to_numpy() / np.where(prev > 0, prev, 1) - 1, np.nan)
        used += 1
        match.append((str(t.date()), len(rows), len(i)))
    return {"sir": sir, "dtc": dtc, "adv": adv, "si_chg_half_month": chg, "files_used": used, "match": match}
