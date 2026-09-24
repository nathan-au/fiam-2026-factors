"""The stock-month table every desk reads and every desk writes back to (one row per stock-month that has a next-month return).

Row identity: (permno, eom). `eom` is the month of the characteristics, `target_month` the month of `ret_exc_lead1m`.
Desk outputs are float arrays aligned to `Panel.df` rows (NaN = "this desk has no view"); nothing is ever fitted on a row of a
later month than the one it scores, and the only cross-sectional operations are within one `eom`.
"""

import numpy as np
import pandas as pd

from . import config as C


class Panel:
    def __init__(self, smoke: bool = False, floor: float = C.FLOOR_MCAP):
        cols = (list(C.FEATURES) + ["permno", "eom", "ticker", "company_name", "me", "gics", "prc", "dolvol_126d", "beta_60m",
                                    "betabab_1260d", "shares", "ret_1_0", "ret_60_12", "ret_12_1", "resff3_12_1", C.TARGET_COL])
        cols = list(dict.fromkeys(cols))
        df = pd.read_parquet(C.CHARS_FILE, columns=cols)
        df["eom"] = pd.to_datetime(df["eom"])
        df = df[df[C.TARGET_COL].notna()].copy()
        if smoke:
            df = df[df["permno"] % 3 == 0]
        df = df.sort_values(["eom", "permno"]).reset_index(drop=True)
        df["raw_prc"], df["raw_dolvol_126d"], df["raw_me"] = df["prc"], df["dolvol_126d"], df["me"]
        df["raw_beta_60m"], df["raw_betabab_1260d"] = df["beta_60m"], df["betabab_1260d"]
        df["raw_ret_1_0"], df["raw_ret_60_12"], df["raw_ret_12_1"] = df["ret_1_0"], df["ret_60_12"], df["ret_12_1"]
        df["raw_resff3_12_1"] = df["resff3_12_1"]
        df["sector"] = df["gics"].astype("string").str[:2].fillna("NA").astype(str)
        df["target_month"] = (df["eom"] + pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
        df["target_month"] = pd.to_datetime(df["target_month"]) + pd.offsets.MonthEnd(0)
        df["y"] = df[C.TARGET_COL]

        # Frozen rank transform of experiments/largecap (median fill over ALL stocks of the month, dense rank to [-1, 1]).
        g = df.groupby("eom")
        for v in C.FEATURES:
            df[v] = df[v].fillna(g[v].transform("median")).fillna(0.0)
        g = df.groupby("eom")
        for v in C.FEATURES:
            r = g[v].rank(method="dense")
            rmax = r.groupby(df["eom"]).transform("max")
            df[v] = np.where(rmax > 0, (r / rmax) * 2 - 1, 0.0)

        # LP neutraliser columns of experiments/neutral_dial (all-stock rank-transformed characteristics; size = log market cap)
        r = df["ret_12_1"].fillna(df.groupby("eom")["ret_12_1"].transform("median")).fillna(0.0)
        rk = r.groupby(df["eom"]).rank(method="dense")
        df["x_mom"] = np.where(rk.groupby(df["eom"]).transform("max") > 0, (rk / rk.groupby(df["eom"]).transform("max")) * 2 - 1, 0.0)
        df["x_size"] = np.log(df["raw_me"].to_numpy(dtype=float))
        df["x_vol"], df["x_qual"] = df["ivol_capm_21d"], df["qmj"]
        self.floor = floor
        self.df = df
        self.u = ((df["raw_prc"].abs() >= C.MIN_PRICE) & (df["raw_me"] >= floor) & (df["raw_dolvol_126d"] >= C.MIN_DOLVOL)
                  & df["raw_beta_60m"].notna() & df["raw_betabab_1260d"].notna()).to_numpy()
        tm = df["target_month"]
        self.period = np.where((tm >= C.DEV[0]) & (tm <= C.DEV[1]), "dev", np.where((tm >= C.TEST[0]) & (tm <= C.TEST[1]), "test", "none"))
        self.months = df["target_month"].to_numpy()
        self.y = df["y"].to_numpy()
        # size terciles inside the universe (per month), used by robustness checks
        me = df["raw_me"].where(self.u)
        t = me.groupby(df["eom"]).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if s.notna().sum() >= 3 else np.nan)
        self.terc = t.fillna(-1).to_numpy().astype(int)
        self.sector = df["sector"].to_numpy()

    # ---- selection helpers -------------------------------------------------------------------------------------------------
    def mask(self, period: str, universe: bool = True) -> np.ndarray:
        m = (self.period == period) if period in ("dev", "test") else (self.period != "none")
        return m & self.u if universe else m

    def meta_cols(self):
        return ["permno", "eom", "target_month", "ticker", "company_name", "sector", "raw_prc", "raw_dolvol_126d", "raw_me",
                "raw_beta_60m", "raw_betabab_1260d", "x_size", "x_mom", "x_vol", "x_qual", C.TARGET_COL]

    def preds_frame(self, score, period: str, extra: dict | None = None) -> pd.DataFrame:
        """Frame the LP consumes: universe-screen columns + `pred`. Rows of the period only (LP applies its own screens)."""
        m = self.mask(period, universe=False)
        out = self.df.loc[m, self.meta_cols()].copy()
        out["pred"] = np.asarray(score, dtype=float)[m]
        for k, v in (extra or {}).items():
            out[k] = np.asarray(v)[m]
        return out.reset_index(drop=True)

    def join_wide(self, other: pd.DataFrame, cols: list[str], fill_zero=(), key=("permno", "eom")) -> dict:
        """Left-join columns of a (permno, eom)-keyed table onto the panel rows; return {col: array}. Absent rows -> NaN, or 0 for
        `fill_zero` columns (counts). The panel row count never changes."""
        o = other[list(key) + cols].copy()
        o["eom"] = pd.to_datetime(o["eom"])
        m = self.df[list(key)].merge(o, on=list(key), how="left", validate="one_to_one")
        assert len(m) == len(self.df)
        out = {}
        for c in cols:
            v = m[c].to_numpy(dtype=float)
            if c in fill_zero:
                v = np.nan_to_num(v, nan=0.0)
            out[c] = v
        return out

    def truncated(self, t) -> "Panel":
        """A copy holding only rows with eom <= t (look-ahead audit: every desk output at t must be identical when built from this)."""
        import copy
        t = pd.Timestamp(t)
        keep = (self.df["eom"] <= t).to_numpy()
        new = copy.copy(self)
        new.df = self.df.loc[keep].reset_index(drop=True)
        for a in ("u", "period", "months", "y", "terc", "sector"):
            setattr(new, a, getattr(self, a)[keep])
        return new
