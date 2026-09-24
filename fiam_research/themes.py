"""JKP (Jensen, Kelly & Pedersen 2023) themes with published signs, built on our universe. No fitting: cluster membership and the long direction
come from github.com/bkelly-lab/ReplicationCrisis (cached in cache/jkp/). `intrinsic_value` (our panel's name) = JKP `ival_me`.

signed_ranks(ctx)  -> DataFrame (rows = ctx.P.df rows, cols = characteristics) of direction * universe percentile rank in [-1, 1] (0 = missing / outside
                      universe), cached per panel fingerprint in cache/research/.
theme_scores(ctx, agg) -> DataFrame of 13 theme scores (mean of member signed scores).
"""

import hashlib

import numpy as np
import pandas as pd

from fiam_desks import config as C
from fiam_desks.factors_desk import urank

from . import core

JKP_DIR = C.CACHE / "jkp"
RCACHE = C.CACHE / "research"
ALIAS = {"ival_me": "intrinsic_value"}


def jkp_map() -> pd.DataFrame:
    d = pd.read_csv(JKP_DIR / "factor_details.csv").dropna(subset=["abr_jkp"])
    c = pd.read_csv(JKP_DIR / "cluster_labels.csv")
    m = c.merge(d[["abr_jkp", "direction"]], left_on="characteristic", right_on="abr_jkp", how="left")
    m["col"] = m["characteristic"].map(lambda x: ALIAS.get(x, x))
    avail = set(pd.read_csv(C.FIAM_DIR / "factor_char_list.csv")["variable"])
    m = m[m["col"].isin(avail)].drop_duplicates("col")
    return m[["col", "cluster", "direction"]].reset_index(drop=True)


def _fp(ctx) -> str:
    df = ctx.P.df
    return hashlib.sha256((str(len(df)) + str(df["permno"].sum()) + str(int(ctx.P.u.sum()))).encode()).hexdigest()[:10]


def raw_chars(ctx) -> pd.DataFrame:
    RCACHE.mkdir(parents=True, exist_ok=True)
    f = RCACHE / f"raw_chars_{_fp(ctx)}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    X = core.load_chars(ctx, list(jkp_map()["col"]))
    X.to_parquet(f)
    return X


def _zscore(values, ctx, clip=3.0):
    """Within-month z-score over universe rows, winsorised at +-clip sd; 0 outside universe / missing."""
    P = ctx.P
    v = pd.Series(np.asarray(values, float))
    m = P.df["eom"].to_numpy()
    ok = P.u & np.isfinite(v.to_numpy())
    s = v[ok]
    g = s.groupby(m[ok])
    # rank-based robust version would be urank; here: clip raw at 1st/99th pct then standardise
    lo, hi = g.transform(lambda x: x.quantile(0.01)), g.transform(lambda x: x.quantile(0.99))
    s = s.clip(lo, hi)
    g = s.groupby(m[ok])
    z = ((s - g.transform("mean")) / g.transform("std")).clip(-clip, clip) / clip
    out = np.zeros(len(v))
    out[z.index.to_numpy()] = z.fillna(0.0).to_numpy()
    return out


def signed_scores(ctx, kind: str = "rank") -> pd.DataFrame:
    RCACHE.mkdir(parents=True, exist_ok=True)
    f = RCACHE / f"signed_{kind}_{_fp(ctx)}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    J = jkp_map()
    X = raw_chars(ctx)
    out = {}
    for _, r in J.iterrows():
        x = X[r["col"]].to_numpy(dtype=float)
        out[r["col"]] = r["direction"] * (urank(x, ctx.P) if kind == "rank" else _zscore(x, ctx))
    D = pd.DataFrame(out)
    D.to_parquet(f)
    return D


def theme_scores(ctx, kind: str = "rank", exclude=()) -> pd.DataFrame:
    J = jkp_map()
    D = signed_scores(ctx, kind)
    T = {}
    for th, g in J.groupby("cluster"):
        cols = [c for c in g["col"] if c not in exclude]
        if cols:
            T[th] = D[cols].mean(axis=1).to_numpy()
    return pd.DataFrame(T)


def composite_from_themes(T: pd.DataFrame, weights: dict | None = None, extra: dict | None = None) -> np.ndarray:
    cols = list(T.columns)
    w = pd.Series({c: 1.0 for c in cols} if weights is None else weights).reindex(cols).fillna(0.0)
    s = (T[cols] * w).sum(axis=1).to_numpy()
    tot = w.sum()
    for k, v in (extra or {}).items():
        s = s + np.asarray(v)
        tot += 1
    return s / tot
