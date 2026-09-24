"""Panel variants for the audit (experiments/desk_15_data_audit).

PanelX is fiam_desks.panel.Panel with ONE change: rows whose next-month return is missing are NOT dropped before the universe is formed (the frozen
Panel drops them, which lets target availability - i.e. whether the stock survives month t+1 - define the prediction universe, a mild look-ahead the
FIAM brief warns about). Those rows get y = `fill` (default 0.0) and a boolean `delist` column. Everything else is the frozen logic, which is
patched in by filtering pandas.read_parquet output rather than copying the class (so any later change of Panel is inherited)."""

import numpy as np
import pandas as pd

from fiam_desks import config as C
from fiam_desks import panel as FP


def PanelX(fill: float = 0.0, smoke: bool = False):
    orig = pd.read_parquet

    def rp(*a, **k):
        df = orig(*a, **k)
        if C.TARGET_COL in df.columns:
            df["__delist"] = df[C.TARGET_COL].isna() & (pd.to_datetime(df["eom"]) < pd.Timestamp("2026-08-31"))
            df.loc[df["__delist"], C.TARGET_COL] = fill
        return df

    FP.pd.read_parquet = rp
    try:
        P = FP.Panel(smoke=smoke)
    finally:
        FP.pd.read_parquet = orig
    P.delist = P.df["__delist"].to_numpy(bool)
    return P
